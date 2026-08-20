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

Around sixty functions, in six groups that differ in what they can promise.
The distinction is not bookkeeping — it decides whether the default policy is
faster than the call it replaces, so it is stated up front rather than buried.

| group | functions | `BitExact` | `Fast` |
|---|---|---|---|
| **Exact** | `floor` `ceil` `round` `rint` `trunc` `copysign` `fmod` `remainder` `remquo` `modf` `frexp` `ldexp` `scalbn` `ilogb` `fdim` `fmin` `fmax` `nextafter` `sqrt` | vectorised, exact on every platform | same code |
| **Correctly rounded** | `erf` `erfc` | vectorised; the nearest representable value, on every platform | same arithmetic without the rounding test, below 0.51 ulp |
| **Ported** | `exp` `exp2` `exp10` `expm1` `ln` `log2` `log10` `log1p` `hypot` `pow` `cbrt` `sinh` `cosh` `tanh` `atanh` `sin` `cos` `sincos` `tan` `atan` `atan2` `asin` `acos` | vectorised, replays the platform schedule | separate table-free vector path |
| **Mixed** | `j0` `j1` `y0` `y1` | vectorised below \|x\| = 2, delegating above it | fully vectorised, 2.4x–3.1x |
| **Delegating** | `asinh` `acosh` `jn` `yn` | one lane at a time: bit-exact, but only at parity | vectorised, measured ulp bound |
| **Own** | `lgamma` `lgamma_r` `tgamma` | — | one implementation, <= 1 ulp measured bound |

**Exact** needs no caveat. IEEE-754 pins those results down completely, so any
correct implementation is bit-identical, on any platform, forever; both policy
axes are genuine no-ops.

**Correctly rounded** is the strongest guarantee here, and stronger than
"bit-exact" as the rest of this README uses the term. glibc computes `erf` and
`erfc` with CORE-MATH's routines, which return the representable value
*nearest the true result* for every input. That is a property of the answer,
not of the route to it, so any two correctly-rounded implementations agree —
which makes `BitExact` for this pair a claim about mathematics rather than
about the host's `libm`. It is reached the way CORE-MATH reaches it: a
double-double fast path with a proven error bound, a test asking whether the
bound settles the last bit, and a scalar accurate path for the roughly one
input in thirty thousand where it does not. The fast path vectorises; the
accurate path is the rare-lane repair every kernel here already has. **4x**,
bit-exact, with no policy to opt into.

**Ported** is the crate's headline case: the vector code replays the platform
routine's operation schedule, so it is both bit-exact and several times faster.
`sin`, `cos`, `sincos` and `tan` are the largest members of this group:
glibc's IBM Accurate Mathematical Library schedule for them — a 440-entry
table (and `tan`'s own 186-row `xfg`), degree-11 near-zero Taylor bands, a
six-band control flow, and fused-multiply-add placement that had to
be read out of a disassembly (`objdump -d` against
`__sin_fma`/`__cos_fma`/`__tan_fma`, not inferred from the C source, which
gets the fusion pairing wrong in more than one place) — is replayed
lane-parallel rather than one lane at a time. `sincos` benefits most because
it shares one argument reduction across both outputs, the same asymmetry the
platform routine itself exploits. `tan`'s kernel keeps the family's six-band
shape but adds the cotangent's compensated-division path, and hands `|x| >
1e8` to the same `patch_lanes` repair as the other three; it measures **2.77x
`BitExact`**, a step under `sin`/`cos`'s 3.0x for exactly that extra work.
The newest members are `asin`/`acos`, whose five-degree table schedule is
replayed from a single per-lane gather; their `BitExact` gain is smaller
than the group's headline — measured 0.83-0.85x, just under parity — because
that 13-slot
gather is a per-lane loop (the hardware-gather backend `ROADMAP.md`'s A5 entry
prototyped for `exp` is not rolled out yet), where `atan`/`atan2`'s lighter
seven-slot rows land at 1.28x / 2.74-3.03x. Their `Fast` path still measures
**8.4-8.5x**.

**Mixed** is the order-0 and order-1 Bessel functions, and the split is forced
by their shape. Below `|x| = 2` they are rational functions of `x^2` and
vectorise completely. Above it they are `sqrt(2/(pi x))` times a combination of
`sin x`, `cos x` and `cos 2x` — three delegated calls, where the scalar routine
gets away with two by sharing one argument reduction. So under `BitExact` a
vector with any lane at or above 2 is handed to the scalar routine: a
"vectorised" version of that branch measures *slower* than the call it
replaces, and shipping it would have been a loss dressed up as an
optimisation. `Fast` vectorises both branches.

**Delegating** is the honest part. glibc implements those families with the IBM
Accurate Portable Math Library routines, whose schedule turns on tables of
several hundred entries and on per-expression fused-multiply-add placement that
the compiler chooses and the C source does not show. That has not been
reproduced here, so under `BitExact` those kernels call the platform routine
per lane — still bit-exact, so substituting `rmath` cannot change your result,
but no faster than what it replaces. `Fast` is where their vector path lives,
and it is worth a lot: the hyperbolic inverses measure **8-13x**. `jn` and
`yn` are here for a different reason: they choose between three algorithms on
`n` against `x`, and one of them runs a continued fraction whose length is
decided at run time, so there is no vector shape to give them.

**Own** is the Gamma family. Rust has no `f64::tgamma`, so there is no call for
`BitExact` to be bit-exact *to*; both policies run one vectorised
implementation, described by its measured error. `lgamma_r`'s *sign* output is
exact and is checked against the platform bit for bit, including glibc's
conventions at the poles — which are not the ones the parity rule alone would
give.

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
than quietly invalidating this file. Most are within **4 ulp** — including the
whole inverse trigonometric family, `asin` through `atan2`; `asinh` and `pow`
are within **8**. `pow` is the function that has to work for that number — `y`
multiplies the error in
`log2 x` as well as its value, so its table-free logarithm is carried in
double-double: the division's residue, the scale constants and the first two
series terms are all kept exact, and the corpus pinned at the edge of the
vector domain (`|y log2 x|` near 1020) measures no worse than the moderate
one.

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
| `exp`   |  2.17 ns | **2.20x** | 2.73x | 3.82x | **3.96x** |
| `exp2`  |  1.62 ns | **1.67x** | 1.89x | 3.41x | **3.43x** |
| `expm1` |  8.18 ns | **2.48x** | 3.78x | 9.52x | **10.73x** |
| `ln`    |  1.99 ns | **1.90x** | 2.15x | 3.01x | **3.46x** |
| `log2`  |  2.27 ns | **1.91x** | 2.06x | 3.66x | **4.46x** |
| `pow`   | 12.59 ns | **1.47x** | — | 2.05x | — |
| `cbrt`  |  8.62 ns | **1.67x** | 1.76x | — | — |
| `sinh`  | 12.22 ns | **3.77x** | 3.86x | 11.49x | **13.17x** |
| `cosh`  |  3.12 ns | **1.70x** | 1.96x | 4.27x | **4.68x** |
| `tanh`  | 11.04 ns | **2.67x** | 2.88x | 11.01x | **11.37x** |
| `sqrt`  |  0.42 ns | 1.00x | — | — | — |
| `sin`   | 13.58 ns | **2.97x** | 2.63x | 19.60x | **20.35x** |
| `cos`   | 14.20 ns | **3.01x** | 2.58x | 20.45x | **21.22x** |
| `sincos`| 17.97 ns | **3.59x** | 3.72x | 19.11x | **25.55x** |
| `tan`   | 16.86 ns | **2.77x** | 2.70x | 19.63x | **20.97x** |
| `atan`  |  7.05 ns | **1.28x** | 1.16-1.18x | 9.08x | **9.42x** |
| `atan2` | 14.98 ns | **2.74-3.03x** | 8.03-8.36x | — |
| `asin`  | 12.7 ns | 0.83-0.84x | 0.87x | 8.36-8.45x | **10.2x** |
| `acos`  | 12.8 ns | 0.84-0.85x | 0.86x | 8.44-8.51x | **10.2x** |
| `log10` |  5.50 ns | **2.61x** | 2.27x | 8.26x | **10.03x** |
| `log1p` |  9.99 ns | **1.13x** | 1.31x | 11.39x | **14.36x** |
| `atanh` | 14.12 ns | **1.29x** | 1.29x | 13.73x | **14.95x** |
| `hypot` |  6.70 ns | **1.93x** | — | 4.82x | — |

`tan`/`atan`/`atan2`/`asin`/`acos`/`atanh`/`log10`/`log1p`/`hypot` all have genuine vector `BitExact` schedules.
`asin`/`acos`'s exact rows sit just under parity (0.83-0.90x) rather than above it: their table
bands need a 13-slot per-lane gather and the platform's scalar routine never
pays for more than one band — the one case in this table where the vector
`BitExact` path is not the faster one, and the exact cost `ROADMAP.md`'s A5
hardware-gather backend targets. Their `Fast` rows are the win, as elsewhere.
`log10` reuses `ln`'s exact table walk rather than deriving a new one. `log1p`
replays `__log1p_fma` lane-parallel, `atanh` vectorises `0.5 * ln_1p(2x/(1-x))`, and `hypot` replays the modern Borges
"MyHypot3" compensated correction with unfused vector arithmetic.

### Delegating — parity by default, and a large win under `Fast`

| function | scalar | `BitExact` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|
| `asinh` | 10.28 ns | 1.00x | 4.41x  | 5.94x |
| `acosh` |  9.62 ns | 0.97x | 2.00x  | 9.27x |
| `lgamma`| 32.56 ns | **4.05x** (no second policy) | | |


### Special functions — bit-exact *and* several times faster

`erf` and `erfc` are the clearest case the crate makes: correctly rounded, so
the result is not merely "the same as glibc here" but the same as any correct
implementation anywhere, and four times the throughput under the default
policy. `erfc`'s baseline is high because glibc computes it in double-double
with a run-time rounding test — the same work this crate does, one lane at a
time.

| function | scalar | `BitExact` | + `Finite` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|--:|
| `erf`   | 22.11 ns | **3.93x** | 4.36x | 4.14x | **4.94x** |
| `erfc`  | 60.59 ns | **3.98x** | 4.14x | 4.32x | **4.29x** |
| `exp10` |  2.21 ns | **1.88x** | 2.13x | 3.29x | **3.58x** |

### Bessel — parity by default, 2.4x to 3.1x under `Fast`

| function | scalar | `BitExact` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|
| `j0` | 37.89 ns | 0.95x | 3.07x | **3.19x** |
| `j1` | 37.84 ns | 0.95x | 3.05x | **3.00x** |
| `y0` | 38.02 ns | **1.10x** | 2.45x | 2.44x |
| `y1` | 37.79 ns | **1.11x** | 2.35x | **2.40x** |

The `BitExact` column is the point of the design rather than a disappointment:
it is what an honest vectorisation of a function whose cost is delegated
trigonometry looks like. `Fast` replaces that trigonometry with rmath's own
and the vector reappears.

### Single precision

| function | scalar | `BitExact` | `Fast` |
|---|--:|--:|--:|
| `expf`  | 1.13 ns | **1.37x** | **3.40x** |
| `exp2f` | 1.10 ns | **1.42x** | **3.51x** |
| `logf`  | 1.35 ns | **1.51x** | **3.32x** |
| `log2f` | 1.33 ns | **1.49x** | **3.44x** |
| `sqrtf` | 0.14 ns | 1.12x | 1.15x |
| `cbrtf` | 2.50 ns | 0.96x | **2.86x** |
| `tanhf` | 2.62 ns | 0.94x | **4.99x** |
| `sinf`  | 8.19 ns | 1.00x | **17.94x** |
| `cosf`  | 8.08 ns | 1.00x | **17.49x** |
| `tanf`  | 3.46 ns | 0.96x | **7.40x** |
| `exp10f`| 1.30 ns | **1.77x** | **2.22x** |
| `erff`  | 5.66 ns | **1.62x** | 1.62x |
| `erfcf` | 6.26 ns | **1.41x** | 1.40x |

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
- **`cbrtf` and `tanhf` sit just under parity under `BitExact`.** Their
  single-precision routines are cheap enough that widening into the
  double-precision kernel costs more than it saves, so the bit-exact path
  delegates; see `src/kernels/single/`. Both have native `Fast` paths instead
  of the widened one: `cbrtf`'s is a bit-pattern seed and three Newton steps,
  within 1 ulp; `tanhf`'s is a direct polynomial below `|x| = 1` and
  `1 - 2/(e^{2x}+1)` above it (via the native `f32` `exp2`), within 2 ulp,
  routing the tail past `|x| = 9` — where `tanh` is within one `f32` ulp of
  `+-1`, and rounds to it exactly from about 9.02 — to the scalar reference
  rather than paying for either formula there. That is where 2.84x and 4.97x
  come from.

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
| `tools/gen_special_tables.py` | regenerates the `erf`, `erfc` and Bessel tables from glibc's C |
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
   `Fast` bound, `tests/single.rs` for `f32`, `tests/glibc.rs` if `std` has no
   method to compare against.

## Testing

| file | what it establishes |
|---|---|
| `tests/bit_exact.rs` | the ported kernels match the platform, at every width, over millions of inputs |
| `tests/delegating.rs` | `BitExact` really is bit-exact for the delegating kernels too — a kernel that quietly took its `Fast` path would still look fine to an accuracy test |
| `tests/single.rs` | every `f32` function against the platform; `--ignored` runs all 2^32 inputs per function |
| `tests/accuracy.rs` | the `Fast` ulp bounds this file quotes, so loosening one fails the build |
| `tests/glibc.rs` | the functions `std` has no method for — `erf`, `erfc`, the Bessel family, `exp10`, `remquo`, `modf`, `ilogb`, `fdim`, `fmin`, `fmax`, `rint`, `scalbn`, `lgamma_r` — against the platform `libm` through `extern "C"`, at every width; `--ignored` runs all 2^32 `f32` inputs |
| `tests/policy.rs` | `Finite` agrees with `FullRange` inside the domain, and the builder is zero-cost |

```sh
cargo test --release
cargo test --release -- --ignored     # the exhaustive f32 sweeps, ~90 s
```

## Status

Covered: the exact, correctly-rounded, ported, mixed, delegating and Gamma
groups listed at the top — around sixty functions, in both precisions.

Known limitations, in rough order of how much they would be missed:

- **Sixteen functions are delegating, not ported.** Their `BitExact` path is
  correct but not fast. The trigonometric family is what is most worth porting
  next, and it is the largest remaining job: glibc uses the IBM Accurate
  Portable Math Library routines there, with a 440-entry table and a separate
  reduction for huge arguments. `log10` is a smaller one — it is not the
  fdlibm composition on `log`, so it needs its own schedule read out.
- **`lgamma` on the negative half-line** is a difference of comparable terms
  near its zeros, where relative error is unbounded for any implementation. The
  bound there is stated as absolute (below 1e-12), not relative.
- **`tgamma` above 18** is `exp(lgamma(x))`, with `lgamma` carried in
  double-double so the value handed to `exp` is the correctly rounded `f64`
  nearest the truth — verified against the platform's own `lgamma` exactly.
  What is left (measured at 512-513 ulp near the overflow threshold, down
  from over 2000) is not fixable by computing harder: composing through any
  single correctly-rounded logarithm before exponentiating discards
  information the true value carried, and glibc's own `tgamma` is provably
  not `exp` of its own correctly-rounded `lgamma` either. Below 18 the
  recurrence reaches `[1, 2]` directly and it is within 16.
- **The single-precision Bessel functions are scalar under `BitExact`.** glibc
  repairs them near each zero with one of 64 tabulated polynomials, and beyond
  the 64th zero with an asymptotic form behind a 192-bit Payne-Hanek
  reduction; which of the three runs is decided *after* the rational fit, on
  how small its bracket came out. Blending that would cost every lane more than
  the scalar call it replaces. `Fast` widens to the double-precision vector
  kernel, which is both faster and — since glibc's own bound for `j0f` is 9
  ulps — more accurate, just not identical.
- **`jn` and `yn` are scalar under both policies.** The backward recurrence
  runs a continued fraction whose length is decided at run time by iterating
  until a convergent exceeds `1e9`; a vector would run the longest lane's loop
  for every lane.
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
`erf` and `erfc` are ported from glibc's copies of the
[CORE-MATH](https://core-math.gitlabpages.inria.fr/) routines, Copyright (c)
2022-2025 Alexei Sibidanov, Paul Zimmermann, Tom Hubrecht and Claude-Pierre
Jeannerod (MIT); the Bessel family from glibc's fdlibm, Copyright (c) 1993 Sun
Microsystems, Inc. The `Fast` coefficients are this crate's own, generated by
`tools/gen_poly.py`.
