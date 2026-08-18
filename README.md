# rmath

A SIMD `libm` for `f64` — modular, and configured with a builder.

Vectorising elementwise `f64` math usually stalls on the same wall: LLVM will
not vectorise a loop containing a call to `exp` or `log`, because the call is
opaque to it. The loop stays scalar no matter how well the rest of it would
have vectorised. `rmath` provides those functions as vector code, so the loop
can vectorise.

```rust
use rmath::prelude::*;

// Bit-identical to the platform libm, safe on any input.
let f = Exp::new();
assert_eq!(f.eval(1.0_f64), 1.0_f64.exp());

// Or configured: cheaper algorithm, and the caller vouches for the inputs.
let quick = Exp::builder().accuracy(Fast).domain(Finite).build();

// One object, any width — and whole buffers.
let src: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
let mut out = vec![0.0; src.len()];
f.eval_slice(&src, &mut out);       // widest vectors, scalar tail
```

## The two questions it makes you answer

A vector math library forces two decisions a scalar `libm` never asks about,
and getting either wrong is a silent bug. So they are the two axes of the
builder, and both are compile-time types.

**How accurate?** Vector math libraries typically answer "about 1 ulp" and
move on. That is fine until it isn't: 1 ulp of `exp` amplified through a
derivative expression can be 1e-12 of the result, which is the difference
between agreeing with your reference implementation and not. The default here
is `BitExact` — not close to the platform `libm`, *identical* to it. Swapping
`rmath` in cannot change a result, which is what makes the swap reviewable.

**Which inputs?** Handling infinities, NaN and subnormals correctly costs a
test per call. Often the caller already knows the data is well-behaved — a grid
of physical quantities, a normalised buffer — and re-establishing that per call
is waste. `FullRange` is the default and is always safe; `Finite` removes the
test, and the guarantee becomes yours to keep.

|              | `FullRange` (default) | `Finite` |
|---|---|---|
| **`BitExact`** (default) | identical to scalar `libm`, safe on anything | identical inside the domain, wrong outside it |
| **`Fast`** | ≤ 2 ulp, safe on anything | ≤ 2 ulp inside the domain |

## Measured

AMD Ryzen AI 7 350 (Zen 5, AVX-512), 1 M elements, best of 12,
`RUSTFLAGS="-C target-cpu=native"`. Baseline is the scalar `f64::` method
you would otherwise call. Reproduce with
`cargo run --release --example bench`.

| function | scalar | `BitExact` | + `Finite` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|--:|
| `exp`  | 2.11 ns | 0.93 (**2.26x**) | 0.77 (2.73x) | 0.55 (3.83x) | 0.52 (**4.08x**) |
| `exp2` | 1.58 ns | 0.94 (**1.69x**) | 0.84 (1.87x) | 0.49 (3.23x) | 0.45 (**3.48x**) |
| `ln`   | 1.89 ns | 0.81 (**2.32x**) | 0.69 (2.75x) | 0.62 (3.06x) | 0.55 (**3.45x**) |
| `cbrt` | 8.71 ns | 4.91 (**1.77x**) | 4.99 (1.75x) | — | — |
| `sqrt` | 0.42 ns | 0.41 (1.00x) | — | — | — |

Two of those deserve comment rather than burial:

- **`sqrt` gains nothing**, and cannot. It is one instruction, LLVM already
  vectorises loops containing it, and `rmath` has nothing to add. It is
  included so a pipeline of these functions does not have to break its pattern
  for one member — not because it is faster.
- **`cbrt` barely moves with `Finite`.** Its cost is per-lane exponent
  surgery at both ends, not the range check, so removing the check saves
  nothing measurable.

`Fast` was measured at **2 ulp** worst case for `exp`, `exp2` and `ln` over
300k inputs each; `tests/policy.rs` asserts those bounds, so if a change moves
them, the test fails and this table is wrong.

## What "bit-exact" rests on

Every IEEE-754 operation rounds identically regardless of vector width, so
running the reference algorithm's schedule eight lanes at a time gives the same
bits as running it once. The work is in reproducing that schedule exactly —
including where the compiled reference fuses a multiply-add and where it rounds
twice. That is *not* visible in the C source, and it is not uniform:

- glibc's `exp` runs its `_fma` ifunc variant on any FMA-capable x86-64, so its
  schedule is fused. **`exp2` has no `_fma` variant**, so what runs is the
  baseline SSE2 build, with two roundings where `exp` has one. `rmath::exp2`
  therefore uses no fused multiply-adds at all. Fusing them would be *more*
  accurate and would stop matching.
- Even within one function the choice differs: in `exp`'s special-case handler,
  the `k > 0` arm is fused and the `k < 0` arm is not.

Every placement was read from a disassembly of the compiled library, not
inferred. It follows that bit-exactness is a claim about a platform, not a
universal one — so it is tested, not asserted. `tests/bit_exact.rs` compares
every lane against the host's reference over ~7 M inputs per function: branch
boundaries with their two neighbouring representable values either side,
subnormals, specials, and uniformly random bit patterns (which land on NaN and
infinity far more often than any realistic distribution). It fails loudly on a
platform whose library differs rather than silently degrading.

### `cbrt` matches Rust, not C

For `cbrt` — and only `cbrt` — those are different functions. Rust's `std` does
not forward `cbrt` to the platform `libm`; it uses its own port of the
correctly-rounded core-math routine. glibc's `cbrt` is a much cruder
frexp-plus-one-Halley-step algorithm, and the two disagree by an ulp on roughly
half of a random sweep.

`rmath` matches **Rust's `f64::cbrt`**, because the point of `BitExact` is that
substituting `rmath` for the call you were already making cannot change a
result — and in a Rust program, that call is `f64::cbrt`. If you need to
reproduce a C program's `cbrt`, this is not the function for it.

## Requires FMA for full speed

`wide` silently degrades `mul_add` to `a * b + c` when the target has no `fma`
feature — two roundings where the contract requires one. That breaks
bit-exactness on roughly one input in two thousand, which is exactly the kind
of failure that survives casual testing.

`rmath` does not accept that: without `fma` it substitutes a genuine per-lane
FMA, which is correct but several times slower. **Build with
`-C target-cpu=native` (or `-C target-feature=+fma`)** to get the vector
instruction. The benchmark prints a warning if you did not.

## Layout

| path | role |
|---|---|
| `src/simd/` | the `Simd` trait every kernel is written against, plus `f64` and `wide` backends |
| `src/policy.rs` | `BitExact` / `Fast`, `FullRange` / `Finite` — the typestate |
| `src/kernels/` | the vector implementations, one module per function |
| `src/reference.rs` | self-contained scalar ports: the definition of "bit-exact", and the fallback for rare lanes |
| `src/tables/` | generated data tables |
| `src/function.rs` | the builder machinery, one `math_fn!` line per function |
| `tools/gen_tables.py` | regenerates `src/tables/` from upstream C |

Three properties fall out of that split:

- **A kernel is generic over the vector type**, so one implementation serves
  `f64`, `f64x2`, `f64x4` and `f64x8`. Adding a backend means implementing one
  trait and inheriting every function. Adding a width is one line.
- **A function object is zero-sized** and not tied to a lane count. `size_of`
  is 0, `build()` compiles to nothing, and one built object handles every
  width — because width is a property of the data, not of the configuration.
- **The hard cases live in one place.** A vector kernel repairs rare lanes —
  overflow, subnormals, NaN — by calling `reference` for those lanes only, so
  correctness at the edges never depends on the vector code getting them right.
  `eval_slice`'s ragged tail runs the same kernel at one lane, so the tail
  cannot drift from the body.

### Adding a function

1. Add `src/kernels/<name>.rs` exposing
   `pub fn eval<V: Simd, A: Accuracy, D: Domain>(x: V) -> V`.
2. Add a scalar port to `src/reference.rs` — **read the schedule out of a
   disassembly**, not out of the C source.
3. Add one `math_fn!` block to `src/function.rs`.
4. Add the corpus and the assertion to `tests/bit_exact.rs`.

## Status

Covered: `exp`, `exp2`, `ln`, `cbrt`, `sqrt`.

Known limitations, in rough order of how much they would be missed:

- **Only five functions.** `log2`, `log10`, `pow`, `tanh`, `erf` and the
  trigonometric family are not implemented. `log2` needs its own table (glibc
  uses a different, N=64 one); `pow` is the genuinely hard one.
- **Requires `std`**, for exactly two operations: `f64::mul_add` and
  `f64::sqrt`, neither of which is in `core`. Supporting `no_std` needs a
  correctly-rounded software FMA — evaluating `a * b + c` instead would break
  every guarantee above. Nothing else in the crate uses `std`.
- **Bit-exactness is verified on x86-64 + glibc.** The test suite is the
  arbiter on any other platform, and is designed to fail rather than mislead.
- **`Fast` has no `cbrt` variant**; the accuracy axis is accepted but ignored
  there, since the reference algorithm has no cheaper form worth substituting.

## Licence

MIT. Algorithms and tables for `exp`, `exp2` and `ln` are ported from
[ARM optimized-routines](https://github.com/ARM-software/optimized-routines)
(MIT OR Apache-2.0 WITH LLVM-exception), the same code glibc uses;
`cbrt` is ported from [`libm`](https://github.com/rust-lang/compiler-builtins)
(MIT), itself a port of core-math's `cbrt.c`, Copyright (c) 2021-2022 Alexei
Sibidanov.
