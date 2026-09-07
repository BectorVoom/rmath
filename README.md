# rmath

A SIMD `libm` for `f64` and `f32` — modular, and configured with a builder.

Vectorising elementwise floating-point math usually stalls on the same wall:
LLVM will not vectorise a loop containing a call to `exp` or `log`, because the
call is opaque to it. The loop stays scalar no matter how well the rest of it
would have vectorised. `rmath` provides those functions as vector code, so the
loop can vectorise.

It implements them, rather than wrapping them: the crate does not link a
single routine from the library it exists to replace, in either precision, on
any input.

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

Around sixty functions, in four groups that differ in what they can promise.
The distinction is not bookkeeping — it decides whether the default policy is
faster than the call it replaces, so it is stated up front rather than buried.

| group | functions | `BitExact` | `Fast` |
|---|---|---|---|
| **Exact** | `floor` `ceil` `round` `rint` `trunc` `copysign` `fmod` `remainder` `remquo` `modf` `frexp` `ldexp` `scalbn` `ilogb` `fdim` `fmin` `fmax` `nextafter` `sqrt` | vectorised, exact on every platform | same code |
| **Correctly rounded** | `erf` `erfc` `asinh` `acosh`, and in `f32` most of the library | vectorised where a kernel exists; the nearest representable value, on every platform | same arithmetic without the rounding test, below 0.51 ulp |
| **Ported** | `exp` `exp2` `exp10` `expm1` `ln` `log2` `log10` `log1p` `hypot` `pow` `cbrt` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `sin` `cos` `sincos` `tan` `atan` `atan2` `asin` `acos` `j0` `j1` `y0` `y1` `jn` `yn` | vectorised, replays the platform schedule | separate table-free vector path |
| **Own** | `lgamma` `lgamma_r` `tgamma` | — | one implementation, <= 2 ulp measured bound |

**Exact** needs no caveat. IEEE-754 pins those results down completely, so any
correct implementation is bit-identical, on any platform, forever; both policy
axes are genuine no-ops.

**Correctly rounded** is the strongest guarantee here, and stronger than
"bit-exact" as the rest of this README uses the term. glibc has been moving
family by family onto CORE-MATH's routines, which return the representable
value *nearest the true result* for every input — `erf` and `erfc` in double
precision, along with `asinh` and `acosh`, and in single precision nearly
everything that is not an exponential or a logarithm. That is a property of the answer,
not of the route to it, so any two correctly-rounded implementations agree —
which makes `BitExact` for these a claim about mathematics rather than
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
1e8` to the same `patch_lanes` repair as the other three.

All four also *guard* their narrow bands rather than blending them. The tiny,
polynomial and mid bands nest, so one `any()` decides whether any of the three
is reachable at all, and a vector of large arguments — every asymptotic Bessel
argument, most of any wide corpus — runs the table path alone instead of
evaluating three more kernels and discarding them. Blending is still what
happens *within* a mixed vector; the guard only skips work no lane wants. On
the default corpus that took `sin` from 2.45x to **5.67x**, `cos` from 2.44x
to **5.77x** and `sincos` from 2.75x to **6.90x**; on `--corpus=in-domain`,
whose angles are small enough to sit inside the bands, it changes nothing
either way (2.05x / 1.93x / 2.32x), which is the shape a guard should have.
`asinh` (2.45x), `acosh` (1.01x-2.05x), `log1p` (2.57x-2.69x), and `atanh` (3.53x-3.69x)
are fully vectorised.
`asin`/`acos` are the pair this round rebuilt. Their table is really indexed by the
argument's top eight bits — `e_asin.c`'s six band-index formulas are six spellings of
`floor(256*|x|) - 32`, and its band centre is always `(row + 32.5)/256` — so the whole
per-lane index computation becomes three vector instructions and the band centre is
computed rather than read. Repacked at one stride with the unused coefficient slots
zeroed, a row is twelve adjacent doubles, which a single block-load-and-transpose
gather (`Simd::gather_run`) reads as two 512-bit loads instead of twelve scalar ones.
`BitExact` went from **0.75x to 4.8-5.5x**, with no scalar repair left on any input;
`Fast` delivers **9.2-10.1x**.

**Ported** used to have two exceptions, and the reasons they stopped being
exceptions are the two most interesting things in this crate's recent
history. Both were the Bessel family.

The first was `j0`/`j1`/`y0`/`y1`. Below `|x| = 2` they are rational functions
of `x^2` and vectorise completely; above it they are `sqrt(2/(pi x))` times a
combination of `sin x`, `cos x` and `cos 2x`, which is about ninety per cent
of their cost. While glibc's `__sin`/`__cos` were only available here one lane
at a time, vectorising that branch meant *three* scalar trigonometric calls
per lane where the routine being replaced makes two, and it measured 0.8x. So
`BitExact` handed any vector with a lane at or above 2 to the scalar routine
and settled for parity — the honest answer at the time, and stale the moment
the IBM Accurate Portable Math schedule was ported. The three calls are now
one shared argument reduction and two kernel evaluations plus a second
reduction for `cos 2x`, all packed: **3.8-4.0x** under the default policy,
against 0.95-1.11x before.

The second was `jn`/`yn`, which choose between three algorithms on `n` against
`x`, one of them a continued fraction whose length is decided at run time. The
standing argument was that a vector would have to run the longest lane's loop
for every lane. That is true and it is not the objection it sounds like: the
scalar routine runs the *sum* of the lanes' trip counts, so the longest is
already the cheaper of the two. Each branch is a masked recurrence in which a
finished lane freezes and the loop ends when all of them have — **3.0-4.7x**,
bit-exact, where the README used to record parity.

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
  functions run a native scalar `f32` port once per lane instead, which is
  both exact and faster.

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
whole inverse trigonometric family, `asin` through `atan2`, and `cbrt`; `asinh` and `pow`
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
call it replaces. The `sqrt`, `asin` and `acos` rows were re-measured in a later
session on the same machine (the interleaved A/B in `ROADMAP.md`'s Cycle 6 entry);
the `scalar` column drifts a little between sessions, the speedups are within-run
ratios and do not.

One caveat that applies to every branchy scalar baseline here and bites `asin`/`acos`
hardest: measure them on a *small* buffer replayed many times and glibc's branch
predictor memorises the whole sequence of band branches, which is worth a factor of
several to it and to nothing in this crate. These are 1 M-element rows for that reason.
`asin` reads 1.6x at 4096 elements and 5.5x at 1 M, on the same code and the same data
distribution.

### Ported — bit-exact *and* faster

| function | scalar | `BitExact` | + `Finite` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|--:|
| `exp`   |  2.17 ns | **3.02x** | 3.47x | 4.34x | **4.45x** |
| `exp2`  |  1.62 ns | **1.72x** | 2.02x | 3.48x | **3.46x** |
| `expm1` |  8.28 ns | **5.75x** | 6.31x | 9.42x | **11.04x** |
| `ln`    |  1.99 ns | **2.62x** | 2.94x | 3.49x | **3.70x** |
| `log2`  |  2.27 ns | **2.67x** | 2.96x | 3.72x | **4.70x** |
| `pow`   | 12.59 ns | **1.84x** | — | 2.59x | — |
| `cbrt`  | 11.46 ns | **2.18x** | 2.15x | 8.17x | **9.26x** |
| `sinh`  | 12.49 ns | **6.35x** | 6.38x | 10.00x | **10.93x** |
| `cosh`  |  3.48 ns | **3.06x** | 3.56x | 5.57x | **6.08x** |
| `tanh`  | 11.09 ns | **5.48x** | 5.94x | 9.10x | **10.42x** |
| `asinh` |  4.65 ns | **2.38x** | 2.44x | 5.49x | **5.82x** |
| `acosh` |  4.54 ns | **1.90x** | 2.01x | 7.50x | **8.32x** |
| `atanh` |  3.88 ns | **3.54x** | 3.62x | 13.82x | **14.87x** |
| `sqrt`  |  0.52 ns | 1.03x | 1.05x | 1.05x | 1.05x |
| `sin`   | 13.58 ns | **2.05x** (5.67x def) | 2.12x | 16.99x | **17.07x** |
| `cos`   | 14.20 ns | **1.93x** (5.77x def) | 1.85x | 15.41x | **15.22x** |
| `sincos`| 17.97 ns | **2.32x** (6.90x def) | 2.36x | 16.03x | **18.09x** |
| `tan`   | 16.86 ns | **2.34x** (2.89x def) | 2.47x | 17.60x | **18.47x** |
| `atan`  |  7.05 ns | **1.92x** (8.52x coh) | 1.87x | 11.64x | **13.48x** |
| `atan2` | 14.98 ns | **1.58x** (2.74-3.03x def) | — | 12.40x | — |
| `asin`  | 13.39 ns | **5.49x** | 5.47x | 9.23x | **9.84x** |
| `acos`  | 13.56 ns | **4.82x** | 4.99x | 9.26x | **10.06x** |
| `log10` |  5.50 ns | **3.07x** | 3.39x | 7.52x | **9.43x** |
| `log1p` |  9.99 ns | **1.97x** (2.69x def) | 2.18x | 9.05x | **11.37x** |
| `hypot` |  6.70 ns | **2.81x** | — | 5.09x | — |

`expm1`/`sinh`/`cosh`/`tanh` are the four rows with no scalar repair left in them at all: their
`BitExact` paths answer every input — the overflow band, the infinities, the NaNs — in vector form.
That is worth more than the table above shows, because the table is one corpus. On the adversarial
`--corpus=boundary` mix, where roughly one lane in seven is at or past the overflow threshold and a
single such lane used to drag all eight through the scalar reference, they moved from
`expm1` 0.60x, `sinh` 0.58x, `cosh` 0.34x to **1.30x / 1.28x / 1.13x** — from well under the `libm`
call they replace to above it. Every corpus the harness ships now clears parity for all four.

`tan`/`atan`/`atan2`/`asin`/`acos`/`asinh`/`acosh`/`atanh`/`log10`/`log1p`/`hypot` all have genuine vector `BitExact` schedules.
`asin`/`acos`'s exact rows used to sit under parity at 0.75-0.85x, on the reasoning that their
table bands need a per-lane gather while the platform's scalar routine never pays for more than
one band. Half of that was right: the gather was the cost, but it was the cost of the table's
*shape*. Indexed by one affine expression, repacked at one stride with one polynomial degree, and
read a row at a time, the same schedule is **4.8-5.5x** (see `ROADMAP.md`'s Cycle 6). Their `Fast`
rows deliver 9.2-10.1x.
`log10` reuses `ln`'s exact table walk. `log1p` replays `__log1p_fma` lane-parallel with vectorised reduction and Horner FMA tail, `asinh`/`acosh`/`atanh` evaluate vectorised band structures with `V::select` blending, and `hypot` replays Borges "MyHypot3" compensated correction with unfused vector arithmetic.

### Gamma

| function | scalar | `rmath` |
|---|--:|--:|
| `lgamma`| 32.56 ns | **4.37x** (no second policy) |


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

### Bessel — four times the platform, bit for bit

| function | scalar | `BitExact` | + `Finite` | `Fast` | + `Finite` |
|---|--:|--:|--:|--:|--:|
| `j0` | 35.40 ns | **4.00x** | 4.18x | 6.73x | **7.17x** |
| `j1` | 35.48 ns | **3.98x** | 4.13x | 6.80x | **7.15x** |
| `y0` | 35.64 ns | **3.78x** | 3.97x | 6.55x | **7.09x** |
| `y1` | 35.35 ns | **3.78x** | 3.97x | 6.58x | **7.04x** |
| `jn` (one order per buffer) | 71.34 ns | **3.45x** | | 4.26x | |
| `jn` (order varying per lane) | 71.00 ns | **3.02x** | | 3.70x | |
| `yn` (one order per buffer) | 66.54 ns | **4.68x** | | 6.45x | |
| `yn` (order varying per lane) | 66.24 ns | **3.98x** | | 5.25x | |

The `BitExact` column read 0.95-1.11x until the trigonometric family was
ported, and the honest description of it then was "what an honest
vectorisation of a function whose cost is delegated trigonometry looks like".
That was true of the code and false of the problem: the cost was delegated
because the schedule had not been read out yet, not because it could not be.

Three things carry the current number. The three trigonometric calls share one
argument reduction and two kernel evaluations
(`trig::sincos_split`, which replays `__sin` and `__cos` rather than
`__sincos` — they are not the same computation and fdlibm's Bessel routines
are bit-exact against the separate pair). The four-row `P`/`Q` coefficient
table is a three-`select` blend ladder rather than a twenty-three-slot
per-lane gather. And each of the two branches hands the lanes it is *not*
answering for a stand-in argument, so a mixed vector does not open the
trigonometry's narrow bands — or, worse, send `y0`'s nested `j0` down its own
asymptotic branch — for an answer that is about to be discarded.

`jn` and `yn` are masked recurrences over the same `j0`/`j1` and `y0`/`y1`,
computed as a *pair* so the order-0 and order-1 evaluations share their
trigonometry too. Their two rows are the same code on two corpora: one order
across the whole buffer, and a different order in every lane. The gap between
them is the cost of running each lane's loop to the longest lane's trip count,
which is the price of the design and still well inside the scalar routine's
sum-of-all-lanes.

The `Fast` gains came along for free — the same structural work, with rmath's
own trigonometry underneath.

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

- **The 1.00x column is 1.00x on purpose.** Those kernels run a *scalar*
  routine once per lane — this crate's own, not the platform's, since nothing
  here calls `libm` any more — so parity is the ceiling, and the small
  shortfall on some rows is the cost of packing and unpacking the lane array
  around it. If you want those functions vectorised, that is what `Fast` is
  for — and it is a 4x to 22x difference, not a rounding one.
- **`sqrt` gains nothing**, and cannot. IEEE-754 *requires* square root to be
  correctly rounded, every target here has one instruction that does it, and
  glibc lowers `sqrt` to that same instruction — so there is no schedule to
  replay and no accuracy to trade. LLVM also vectorises loops containing
  `sqrt` on its own, which is precisely what it will not do for `exp`, so the
  baseline this row is measured against is already `vsqrtpd`: it compares two
  identical instruction streams. `sqrt` is here so a pipeline of these
  functions does not have to break its pattern for one member, not because it
  is faster. The same is true of `floor`, `ceil`, `round` and `trunc`. At
  cache-resident sizes they run at parity (~1.00–1.08x); at 1 M elements
  (16 MB of source and destination traffic, past L2) memory bandwidth
  dominates and the row reads ~1.0x either way. What did have slack was the
  buffer walk around the instruction, not the instruction: `eval_slice` used
  to leave a bounds check and a branch-to-panic inside its own loop, which is
  nothing next to `exp`'s hundred-instruction body and was two of the seven
  instructions in `sqrt`'s. It is written with `chunks_exact` now, and LLVM
  unrolls it.
- **`acosh` gains far more from `Finite` than anything else** (1.97x to 8.95x).
  Its domain test is two comparisons *and* a NaN-producing branch for `x < 1`,
  which is expensive relative to the small amount of arithmetic that follows.
- **`cbrtf` and `tanhf` sit just under parity under `BitExact`.** Their
  single-precision routines are cheap enough that widening into the
  double-precision kernel costs more than it saves, so the bit-exact path
  walks the lanes over the scalar `f32` port instead; see
  `src/kernels/single/`. Both have native `Fast` paths instead
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
   `tests/delegating.rs` for a scalar-loop kernel, `tests/accuracy.rs` for the
   `Fast` bound, `tests/single.rs` for `f32`, `tests/glibc.rs` if `std` has no
   method to compare against.

## Testing

| file | what it establishes |
|---|---|
| `tests/bit_exact.rs` | the ported kernels match the platform, at every width, over millions of inputs |
| `tests/delegating.rs` | `BitExact` really is bit-exact for the kernels that walk the lanes rather than replaying a schedule — one that quietly took its `Fast` path would still look fine to an accuracy test |
| `tests/single.rs` | every `f32` function against the platform; `--ignored` runs all 2^32 inputs per function |
| `tests/accuracy.rs` | the `Fast` ulp bounds this file quotes, so loosening one fails the build |
| `tests/glibc.rs` | the functions `std` has no method for — `erf`, `erfc`, the Bessel family, `exp10`, `remquo`, `modf`, `ilogb`, `fdim`, `fmin`, `fmax`, `rint`, `scalbn`, `lgamma_r` — against the platform `libm` through `extern "C"`, at every width; `--ignored` runs all 2^32 `f32` inputs |
| `tests/policy.rs` | `Finite` agrees with `FullRange` inside the domain, and the builder is zero-cost |

```sh
cargo test --release
cargo test --release -- --ignored     # the exhaustive f32 sweeps, ~90 s
```

## Status

Covered: the exact, correctly-rounded, ported and Gamma groups listed at the
top — around sixty functions, in both precisions.

**The crate no longer links `libm` at all.** Not in `f64`, not in `f32`, not
on any input: every reference is this crate's own port, and the compiled rlib
carries no undefined libm symbol. The only things it still takes from outside
`core` are `mul_add` and `sqrt`, which are IEEE-754 arithmetic rather than
library functions. In `f64` every `BitExact` path is additionally a genuine
vector schedule, the Bessel family included; in `f32` several are still a
scalar loop over the port, which is a speed gap and not a provenance one.

Known limitations, in rough order of how much they would be missed:

- **`lgamma` on the negative half-line** is a difference of comparable terms
  near its zeros, where relative error is unbounded for any implementation. The
  bound there is stated as absolute (below 1e-12), not relative.
- **`tgamma` above 18** uses `stirling_dd` to compute `ln(Gamma(x))` in
  double-double, passing `(hi, lo)` directly to the vectorised `pow_exp`
  kernel — keeping the high precision all the way through exponentiation
  rather than compressing to single `f64` beforehand. Measured at **<= 2 ulp**
  for large arguments and **<= 1 ulp** near the overflow threshold over 100M
  samples (`tests/ulp_scan.rs::scan_gamma`). Below 18 the recurrence reaches
  `[1, 2]` directly.
- **The single-precision Bessel functions are scalar under `BitExact`.** glibc
  repairs them near each zero with one of 64 tabulated polynomials, and beyond
  the 64th zero with an asymptotic form behind a 192-bit Payne-Hanek
  reduction; which of the three runs is decided *after* the rational fit, on
  how small its bracket came out. Blending that would cost every lane more than
  the scalar call it replaces. `Fast` widens to the double-precision vector
  kernel, which is both faster and — since glibc's own bound for `j0f` is 9
  ulps — more accurate, just not identical.
- **The `f64` Bessel family's vector path stops at `|x| = 52707175`.** That is
  the trigonometry's limit, not theirs: above it `sin`/`cos` need glibc's
  `__branred` Payne-Hanek reduction, which the scalar reference reproduces but
  the vector kernel does not. Under `FullRange` such a lane goes to the scalar
  reference, which is the same three scalar trigonometric calls the repair
  would have made and none of the vector work around them; under `Finite` the caller has vouched for the bound and
  gets a wrong number if they were wrong. `jn`/`yn` additionally take the
  vector path only for `|n| <= 2^24`, past which fdlibm's own `int` arithmetic
  is what the reference reproduces and the recurrence is sixteen million steps
  long however it is spelled.
- **Requires `std`**, for exactly two operations: `mul_add` and `sqrt`, neither
  of which is in `core`. Both are IEEE-754 *arithmetic* rather than library
  functions, and neither reaches `libm.so` in practice — `sqrt` lowers to one
  instruction and `mul_add` resolves against `compiler-builtins`. Supporting
  `no_std` needs a correctly-rounded software FMA: evaluating `a * b + c`
  instead would break every guarantee above, since half the schedules here
  turn on exactly which multiplies fuse.
- **Bit-exactness is verified on x86-64 + glibc.** The test suite is the
  arbiter on any other platform, and is designed to fail rather than mislead.

## Licence

MIT. Algorithms and tables for `exp`, `exp2`, `ln`, `log2`, `pow` and their
single-precision counterparts are ported from
[ARM optimized-routines](https://github.com/ARM-software/optimized-routines)
(MIT OR Apache-2.0 WITH LLVM-exception), the same code glibc uses; `cbrt` is
ported from [`libm`](https://github.com/rust-lang/compiler-builtins) (MIT),
itself a port of core-math's `cbrt.c`, Copyright (c) 2021-2022 Alexei Sibidanov.
`erf`, `erfc`, `asinh` and `acosh`, together with most of the
single-precision library — `tanf`, `asinf`, `acosf`, `atanf`, `atan2f`, the
hyperbolic family, `expm1f`, `log1pf`, `log10f`, `cbrtf` — are ported from
glibc's copies of the
[CORE-MATH](https://core-math.gitlabpages.inria.fr/) routines, Copyright (c)
2022-2025 Alexei Sibidanov, Paul Zimmermann, Tom Hubrecht and Claude-Pierre
Jeannerod (MIT); the Bessel family from glibc's fdlibm, Copyright (c) 1993 Sun
Microsystems, Inc.; and `sinf`, `cosf`, `sincosf` again from ARM
optimized-routines. The double-precision trigonometric and
inverse-trigonometric families, including `__branred`'s Payne-Hanek reduction,
follow glibc's own IBM Accurate Mathematical Library sources, Copyright (C)
2001-2026 Free Software Foundation, Inc., `LGPL-2.1-or-later` — see the
per-table headers in `src/tables/double/` for exactly which constants that
covers. The `Fast` coefficients are this crate's own, generated by
`tools/gen_poly.py`.
