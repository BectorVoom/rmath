# rmath

A SIMD `libm` for `f64` and `f32` — modular, and configured with a builder.

Vectorising elementwise floating-point math usually stalls on the same wall:
LLVM will not vectorise a loop containing a call to `exp` or `log`, because the
call is opaque to it. The loop stays scalar no matter how well the rest of it
would have vectorised. `rmath` provides those functions as vector code, so the
loop can vectorise.

```rust
use rmath::prelude::*;

// Bit-identical to the platform libm, safe on any input.
let f = Exp::new();
assert_eq!(f.eval(1.0_f64), 1.0_f64.exp());

// Or configured: cheaper algorithm, and the caller vouches for the inputs.
let quick = Exp::builder().accuracy(Fast).domain(Finite).build();

// One object, any precision, any width — and whole buffers.
assert_eq!(f.eval(1.0_f32), 1.0_f32.exp());

let src: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
let mut out = vec![0.0; src.len()];
f.eval_slice(&src, &mut out);       // widest vectors, scalar tail
```

## What is covered

Around forty functions, in four groups that differ in what they can promise.
The distinction is not bookkeeping — it decides whether the default policy is
faster than the call it replaces, so it is stated up front rather than buried.

| group | functions | `BitExact` | `Fast` |
|---|---|---|---|
| **Exact** | `floor` `ceil` `round` `trunc` `copysign` `fmod` `remainder` `frexp` `ldexp` `nextafter` `sqrt` | vectorised, exact on every platform | same code |
| **Ported** | `exp` `exp2` `expm1` `ln` `log2` `pow` `cbrt` `sinh` `cosh` `tanh` | vectorised, replays the platform schedule | separate table-free vector path |
| **Delegating** | `sin` `cos` `sincos` `tan` `asin` `acos` `atan` `atan2` `asinh` `acosh` `atanh` `log10` `log1p` `hypot` | one lane at a time: bit-exact, but only at parity | vectorised, measured ulp bound |
| **Own** | `lgamma` `tgamma` | — | one implementation, measured bound |

**Exact** needs no caveat. IEEE-754 pins those results down completely, so any
correct implementation is bit-identical, on any platform, forever; both policy
axes are genuine no-ops.

**Ported** is the crate's headline case: the vector code replays the platform
routine's operation schedule, so it is both bit-exact and several times faster.

**Delegating** is the honest part. glibc implements those families with the IBM
Accurate Portable Math Library routines, whose schedule turns on tables of
several hundred entries and on per-expression fused-multiply-add placement that
the compiler chooses and the C source does not show. That has not been
reproduced here, so under `BitExact` those kernels call the platform routine
per lane — still bit-exact, so substituting `rmath` cannot change your result,
but no faster than what it replaces. `Fast` is where their vector path lives,
and it is worth a lot: **20x** for the trigonometric family.

**Own** is `lgamma` and `tgamma`. Rust has no `f64::tgamma`, so there is no
call for `BitExact` to be bit-exact *to*; both policies run one vectorised
implementation, described by its measured error.

## Both precisions

`f64` and `f32`, at every width the backend provides — `f64x2`, `f64x4`,
`f64x8`, `f32x4`, `f32x8`. One function object serves all of them: precision
and width are properties of the data, not of the configuration.

Single precision is not the double-precision code with narrower lanes. The
platform's `expf`, `logf` and `log2f` are separate algorithms, and they do
their arithmetic *in `double`* over a small table — which is exactly what lets
those kernels widen `f32x8` to `f64x8`, replay the schedule lane-parallel, and
round once. Bit-exact and vectorised at the same time.

`f32` also admits a far stronger test than `f64` does. There are only 2^32
inputs, so `tests/single.rs` checks **every one of them** rather than a sample.
That is not ceremony; it earned its place twice:

- `expf` differed from the platform on exactly **2 inputs out of 4294967296**.
  The cause was one fused multiply-add: glibc computes `r` as a fused
  `InvLn2N*x - kd`, so the product is never rounded to a double of its own.
  Reading it out of the C source gives the wrong answer, and no sampled test
  would ever have found it.
- An earlier design computed the correctly-rounded `f32` functions in `f64` and
  rounded once. That is the standard trick and it is *very nearly* right — it
  failed on 1 input in 4e9 for `log10f` and 2 for `sinhf`, where the `f64`
  result lands within its own error of an `f32` rounding boundary. Those
  functions now delegate instead, which is both exact and faster.

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
| **`Fast`** | measured ulp bound, safe on anything | same bound inside the domain |

`Fast`'s bound is per function, measured against the platform and asserted in
`tests/accuracy.rs`, so a kernel change that loosens one fails the build rather
than quietly invalidating this file. Most are within **4 ulp**; the inverse
trigonometric and inverse hyperbolic families within **8**; `pow` within
**40**, because `y` multiplies the error in `log2 x` as well as its value.

## Measured

AMD Ryzen AI 7 350 (Zen 5, AVX-512), 1 M elements, best of 8,
`RUSTFLAGS="-C target-cpu=native"`. Baseline is the scalar routine you would
otherwise call. Reproduce with
`RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench`.

Numbers are speedups over that baseline; above 1.00x is faster than the `libm`
call it replaces.

### Ported — bit-exact *and* faster

| function | scalar | `BitExact` | + `Finite` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|--:|
| `exp`   |  2.11 ns | **2.22x** | 2.67x | 3.73x | **3.84x** |
| `exp2`  |  1.62 ns | **1.67x** | 1.89x | 3.41x | **3.43x** |
| `expm1` |  8.18 ns | **2.48x** | 3.78x | 9.52x | **10.73x** |
| `ln`    |  1.99 ns | **1.90x** | 2.15x | 3.01x | **3.46x** |
| `log2`  |  2.27 ns | **1.91x** | 2.06x | 3.66x | **4.46x** |
| `pow`   | 12.59 ns | **1.47x** | — | 2.26x | — |
| `cbrt`  |  8.62 ns | **1.67x** | 1.76x | — | — |
| `sinh`  | 12.22 ns | **3.77x** | 3.86x | 11.49x | **13.17x** |
| `cosh`  |  3.12 ns | **1.70x** | 1.96x | 4.27x | **4.68x** |
| `tanh`  | 11.04 ns | **2.67x** | 2.88x | 11.01x | **11.37x** |
| `sqrt`  |  0.42 ns | 1.00x | — | — | — |

### Delegating — parity by default, and a large win under `Fast`

| function | scalar | `BitExact` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|
| `sin`   | 13.18 ns | 1.00x | 19.93x | **20.98x** |
| `cos`   | 13.67 ns | 1.00x | 20.73x | **21.33x** |
| `tan`   | 15.87 ns | 0.98x | 21.26x | **22.06x** |
| `asin`  | 11.24 ns | 1.00x | 9.01x  | 9.86x |
| `acos`  | 11.21 ns | 0.99x | 9.06x  | 9.83x |
| `atan`  |  6.57 ns | 0.97x | 8.55x  | 9.51x |
| `atan2` | 13.66 ns | 0.99x | 8.23x  | — |
| `asinh` | 10.28 ns | 1.00x | 4.41x  | 5.94x |
| `acosh` |  9.62 ns | 0.97x | 2.00x  | 9.27x |
| `atanh` | 14.09 ns | 1.00x | 11.94x | 14.95x |
| `log10` |  4.93 ns | 0.98x | 7.64x  | 9.63x |
| `log1p` |  8.87 ns | 0.99x | 10.97x | 13.90x |
| `hypot` |  3.26 ns | 0.92x | 4.61x  | — |
| `lgamma`| 32.56 ns | **4.05x** (no second policy) | | |

### Single precision

| function | scalar | `BitExact` | `Fast` |
|---|--:|--:|--:|
| `expf`  | 1.12 ns | **1.41x** | **3.47x** |
| `exp2f` | 1.07 ns | **1.40x** | **3.46x** |
| `logf`  | 1.36 ns | **1.53x** | **3.36x** |
| `log2f` | 1.33 ns | **1.51x** | 1.51x |
| `sqrtf` | 0.13 ns | 1.10x | 1.11x |
| `cbrtf` | 2.49 ns | 0.96x | 0.96x |
| `tanhf` | 2.82 ns | 0.93x | 1.99x |

Several of those deserve comment rather than burial:

- **The delegating column is 1.00x on purpose.** Those kernels run the
  platform's own scalar routine once per lane, so parity is the ceiling, and
  the small shortfall on some rows is the cost of packing and unpacking the
  lane array around it. If you want those functions vectorised, that is what
  `Fast` is for — and it is a 4x to 22x difference, not a rounding one.
- **`sqrt` gains nothing**, and cannot. It is one instruction, LLVM already
  vectorises loops containing it, and `rmath` has nothing to add. It is
  included so a pipeline of these functions does not have to break its pattern
  for one member — not because it is faster. The same is true of `floor`,
  `ceil` and `trunc`.
- **`acosh` gains far more from `Finite` than anything else** (1.97x to 8.95x).
  Its domain test is two comparisons *and* a NaN-producing branch for `x < 1`,
  which is expensive relative to the small amount of arithmetic that follows.
- **`cbrtf` and `tanhf` sit just under parity.** Their single-precision
  routines are cheap enough that widening into the double-precision kernel
  costs more than it saves, so both policies delegate; see
  `src/kernels/single/`. A native single-precision `cbrt` approximation would
  be a real `Fast` path, and is not written yet.

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
- In `expf`, glibc fuses *both* uses of the same product, so `InvLn2N * x` is
  never rounded to a double of its own — the C source computes it into a
  variable and reuses it, and transcribing that faithfully is wrong on two
  inputs out of four billion.

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
| `src/simd/` | the `Real` and `Simd` traits every kernel is written against, plus the scalar and `wide` backends |
| `src/policy.rs` | `BitExact` / `Fast`, `FullRange` / `Finite` — the typestate |
| `src/kernels/exact.rs` | the functions IEEE-754 pins down: one implementation, generic over precision *and* width |
| `src/kernels/double/` | double-precision kernels, one module per family |
| `src/kernels/single/` | single-precision kernels |
| `src/reference/` | scalar references: the definition of "bit-exact", and the fallback for rare lanes |
| `src/tables/` | generated data tables and `Fast` coefficients |
| `src/function.rs` | the builder machinery |
| `src/function_defs.rs` | the catalogue: one `math_fn!` block per function |
| `tools/gen_tables.py` | regenerates the ported tables from upstream C |
| `tools/gen_poly.py` | regenerates the `Fast` coefficients by Remez, at 200 bits |

Four properties fall out of that split:

- **A kernel is generic over the vector type**, so one implementation serves
  every width of its precision. Adding a backend means implementing one trait
  and inheriting every function. Adding a width is one line.
- **A function object is zero-sized** and tied to neither a lane count nor a
  precision. `size_of` is 0, `build()` compiles to nothing, and one built
  object handles `f64`, `f32` and every width — because both are properties of
  the data, not of the configuration.
- **The hard cases live in one place.** A vector kernel repairs rare lanes —
  overflow, subnormals, NaN — by calling `reference` for those lanes only, so
  correctness at the edges never depends on the vector code getting them right.
  `eval_slice`'s ragged tail runs the same kernel at one lane, so the tail
  cannot drift from the body.
- **Nothing numeric is transcribed.** Both the ported tables and the `Fast`
  coefficients are generated, so the provenance of every constant is a script
  rather than a claim.

### Adding a function

1. Add a module under `src/kernels/double/` (and `single/`) exposing
   `pub fn eval<V: Simd<Elem = f64>, A: Accuracy, D: Domain>(x: V) -> V`.
2. Add a scalar reference to `src/reference/`. If you are porting, **read the
   schedule out of a disassembly**, not out of the C source — see above for why.
3. Add one `math_fn!` block to `src/function_defs.rs` and re-export from
   `src/lib.rs`.
4. Add the corpus and the assertion: `tests/bit_exact.rs` for a port,
   `tests/delegating.rs` for a delegating kernel, `tests/accuracy.rs` for the
   `Fast` bound, `tests/single.rs` for `f32`.

## Testing

| file | what it establishes |
|---|---|
| `tests/bit_exact.rs` | the ported kernels match the platform, at every width, over millions of inputs |
| `tests/delegating.rs` | `BitExact` really is bit-exact for the delegating kernels too — a kernel that quietly took its `Fast` path would still look fine to an accuracy test |
| `tests/single.rs` | every `f32` function against the platform; `--ignored` runs all 2^32 inputs per function |
| `tests/accuracy.rs` | the `Fast` ulp bounds this file quotes, so loosening one fails the build |
| `tests/policy.rs` | `Finite` agrees with `FullRange` inside the domain, and the builder is zero-cost |

```sh
cargo test --release
cargo test --release -- --ignored     # the exhaustive f32 sweeps, ~90 s
```

## Status

Covered: the exact, ported, delegating and Gamma groups listed at the top —
around forty functions, in both precisions.

Known limitations, in rough order of how much they would be missed:

- **Fourteen functions are delegating, not ported.** Their `BitExact` path is
  correct but not fast. The trigonometric family is what is most worth porting
  next, and it is the largest remaining job: glibc uses the IBM Accurate
  Portable Math Library routines there, with a 440-entry table and a separate
  reduction for huge arguments. `log10` is a smaller one — it is not the
  fdlibm composition on `log`, so it needs its own schedule read out.
- **`Fast` `pow` is the loosest kernel here**, at 40 ulp, because `y`
  multiplies the error in `log2 x` as well as its value. That is inherent to
  having no table — which is what `Fast` is for. `BitExact` `pow` carries the
  logarithm in double-double over glibc's table and is exact.
- **`lgamma` on the negative half-line** is a difference of comparable terms
  near its zeros, where relative error is unbounded for any implementation. The
  bound there is stated as absolute (below 1e-12), not relative.
- **`tgamma` above 18** is `exp(lgamma(x))`, so its relative error grows with
  the argument — near the overflow threshold it reaches some 2000 ulp. Below
  18 the recurrence reaches `[1, 2]` directly and it is within 16.
- **No native single-precision `cbrt`**, so `cbrtf` delegates under both
  policies rather than offering a `Fast` path.
- **Requires `std`**, for exactly two operations: `mul_add` and `sqrt`, neither
  of which is in `core`. Supporting `no_std` needs a correctly-rounded software
  FMA — evaluating `a * b + c` instead would break every guarantee above.
- **Bit-exactness is verified on x86-64 + glibc.** The test suite is the
  arbiter on any other platform, and is designed to fail rather than mislead.

## Licence

MIT. Algorithms and tables for `exp`, `exp2`, `ln`, `log2`, `pow` and their
single-precision counterparts are ported from
[ARM optimized-routines](https://github.com/ARM-software/optimized-routines)
(MIT OR Apache-2.0 WITH LLVM-exception), the same code glibc uses; `cbrt` is
ported from [`libm`](https://github.com/rust-lang/compiler-builtins) (MIT),
itself a port of core-math's `cbrt.c`, Copyright (c) 2021-2022 Alexei Sibidanov.
The `Fast` coefficients are this crate's own, generated by `tools/gen_poly.py`.
