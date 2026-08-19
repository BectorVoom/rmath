# rmath optimisation roadmap

A detailed plan for the next rounds of speed and precision work, written against
the measured state of the crate on 2026-08-19 (AMD Ryzen AI 7 350, Zen 5,
AVX-512, glibc x86-64, `-C target-cpu=native`). Every number quoted here was
measured on that machine with `examples/bench.rs` (best-of-8 over 1M elements)
or the ulp methodology of `tests/accuracy.rs`.

## 1. Where we stand

The round of work just completed closed three gaps:

- `Fast` `pow` tightened from **40 ulp to 8 asserted / ≤4 measured** (the
  table-free logarithm is now carried in double-double throughout), at the cost
  of 2.30x → 2.07x throughput.
- `log2f` `Fast` got a native table-free path: **1.57x → 3.38x**, ≤3 ulp.
- `cbrtf` `Fast` got a native seed-plus-Newton kernel: **0.96x → 2.86x**, ≤1 ulp.

What remains, grouped by what it costs the user today:

| gap | today | ceiling |
|---|---|---|
| Trig family `BitExact` delegates | `sin`/`cos`/`tan` at 0.98–1.02x | ~20x is proven by the `Fast` path; a bit-exact port would land lower but well above parity |
| Inverse trig + hyperbolic inverses `BitExact` delegate | 0.94–1.01x | `Fast` reaches 8–13x |
| `log10`, `log1p`, `hypot`, `atan2` `BitExact` delegate | 0.92–1.02x | `Fast` reaches 4.7–14x |
| Table gathers are per-lane scalar loops | `exp` `BitExact` 2.36x vs `Fast` 4.02x | hardware gathers could close part of that spread |
| `Fast` `pow` floor is `Fast` `exp2`'s ~2 ulp | pow ≤4 measured | ~1 ulp if `exp2`'s fast path is tightened |
| `tgamma` above 18 | up to ~2000 ulp near overflow | <64 with a direct double-double Stirling route |
| `f32` `Fast` kernels that widen to `f64` | `tanhf` 2.20x, `erff` 1.57x, `sinf` via f64 | native f32 arithmetic is 2–4x cheaper |

## 2. Constraints that shape every change

These are the crate's contracts; the plan treats them as hard.

1. **`BitExact` means identical to the platform.** A bit-exact port replays the
   schedule read from a *disassembly* — FMA placement is not visible in C
   source and is not uniform (see `exp2` vs `exp` in the README).
2. **Nothing numeric is transcribed.** New tables come from
   `tools/gen_tables.py` pinned to an upstream source; new `Fast` coefficients
   from `tools/gen_poly.py` (Remez at 200 bits). Both are deterministic —
   regenerating must reproduce committed tables byte-for-byte.
3. **Every `Fast` bound is asserted in `tests/accuracy.rs`** and quoted in the
   README; a change that loosens one must fail the build, not slip through.
4. **No silent algorithm changes.** Where a policy deliberately runs one
   algorithm (f64 `cbrt`), changing that is a design decision to surface, not
   an optimisation to sneak in.
5. **Rare-lane repairs stay in `reference/`.** Vector main paths never own the
   hard cases; `patch_lanes` + the scalar reference does.

## 3. Workstream A — `BitExact` throughput

The delegating rows are the largest remaining value: `BitExact` is the default
policy, and sixteen functions currently gain nothing under it.

### A1. Trigonometric family port (`sin`, `cos`, `sincos`, `tan`) — effort XL

The headline job, and explicitly the largest. glibc computes these with the
IBM Accurate Portable Math Library routines.

Plan of attack:

1. **Scope the schedule first.** Disassemble the host's `__sin_fma` /
   `__cos_fma` / `__tan` (confirm which ifunc variant actually dispatches on
   this CPU — same trap as `exp` vs `exp2`). Map the branch structure: the
   tiny-argument shortcut, the polynomial band, the table band (440-entry
   sincos table), and the huge-argument reduction (`__branred`, a Payne-Hanek
   style double-double reduction).
2. **Extend `tools/gen_tables.py`** to extract the sincos table and the
   reduction constants from the glibc source tree, same discipline as the
   `exp`/`log` tables.
3. **Port in bands.** The polynomial and table bands vectorise cleanly (the
   table gather is the per-lane loop every other kernel already uses).
   `__branred` is rare in real corpora — make it a `patch_lanes` repair first,
   measure, and only vectorise it if profiles say it matters.
4. **`sincos` first, then `sin`/`cos` as projections, then `tan`** (its
   payoff is the same reduction feeding a different rational).
5. **Corpus:** every branch boundary ±2 representable neighbours, the
   quadrant boundaries at multiples of π/2, subnormals, random bit patterns —
   the existing `tests/bit_exact.rs` shape, ~7M inputs per function.

Expected: `BitExact` sin/cos from 1.0x to an estimated 2–4x (the gather and
the reduction bound it; the 20x `Fast` number is not the target). Risk: high —
FMA placement across four branches, and glibc version drift. Mitigation: land
`sincos` alone first; the test suite fails loudly rather than shipping wrong.

### A2. `log10` and `log1p` ports — effort L for `log10` (corrected), M for `log1p`

- **`log10` disassembled and it is not the `ln`/`log2` shape.** Entry point
  `__log10_finite` at `0x51170` opens with an integer-domain fast/special-case
  dispatch (bit tests on the raw `u64` pattern, not the usual float compares),
  then a *second*, much larger block at `0x512a0` with a two-row, 16-byte-stride
  table gather (`[rax+0x9]<<4` / `[rax+0x49]<<4` indexing into what looks like
  a ~64-entry table of coefficient pairs) feeding roughly a dozen `mulsd`/`addsd`
  operations across nine-plus registers before the block was cut off mid-trace.
  This is at least as large as the `pow` kernel's table path, not a "smaller"
  job — the README's own description undersold it. Needs the same disassembly
  budget as A3, and the same warning: do not port under time pressure.
- `log1p` not yet disassembled this round; still expected fdlibm-shaped
  (branchy around 0) per the original plan, but given how wrong the `hypot`
  and `log10` estimates turned out to be, disassemble before estimating.
- `log1p` is fdlibm-shaped (branchy around 0). Vectorising it bit-exactly
  means computing both the small-`x` and reduced paths and selecting — worth a
  prototype to confirm the select overhead still beats parity before
  committing.

### A3. `hypot` port — effort L, corrected from M after disassembly

**This section's original effort estimate was wrong**, found by actually
disassembling `/lib64/libm.so.6`'s `hypot@@GLIBC_2.35` (== `__hypot_finite`,
no separate FMA ifunc — glibc 2.43 on the reference machine has one `hypot`
implementation for x86-64). It is not a short FMA sequence. It is a
**compensated (2Sum/2Product) algorithm that specifically avoids FMA** —
every multiply and add is a separate, non-fused `mulsd`/`addsd`/`subsd`,
because the error-free-transformation technique it uses (Dekker-style: split
`o² + t²` into a head and a compensated tail, √ it, then correct with one
Newton-style division) depends on IEEE separately-rounded arithmetic to
extract the rounding error each step discards. Fusing any of it would not
just "be more accurate and stop matching" (the crate's usual FMA story) —
it would break the error-extraction identities outright.

Shape found so far (entry point `0x48380`, reference disassembly, not yet
fully traced): a fast path for `min/max` within a moderate exponent range
computing `o² + t²`'s head via `addsd`/`mulsd` then correcting with a
2Sum-derived tail (`0x483fe`-`0x48468`); a scaled-down path for very large
inputs guarded by a `2^-...` pre-scale (from `0x48530`, constant at
`0xa5198` = `5.551115123125783e-17` = `2^-54`); a scaled-up path for inputs
where the smaller argument is negligible relative to the larger
(`0x485f0`); and at least one more branch (`0x484e0`) not yet interpreted.
Each scaled branch re-derives the same compensated identity at a different
scale and un-scales the result, which is where most of the remaining
disassembly work is.

This is not a shape to port under time pressure: a subtly wrong constant or
branch threshold here would silently produce a plausible-looking but
non-bit-exact result — exactly the failure mode the crate's whole design
exists to prevent, and worse than not porting it at all. Treat this as
comparable in difficulty to the crate's `erf`/`erfc` double-double work, not
to the `exp`/`ln` table ports. Whoever picks this up next: the constant
addresses above are a real head start; budget accordingly.

### A4. Inverse trig ports (`asin`, `acos`, `atan`, `atan2`) — effort L each

Same IBM-routine family as A1 (tables `doasin`/`atnat` etc.). Do these only
after A1 proves the workflow; they share its tooling and its corpus
discipline. Expected parity → 2–3x each.

### A5. Hardware-gather backend experiment — **prototyped on `exp`, accepted; wider rollout paused**

Every `BitExact` table kernel pays a per-lane scalar loop for its gathers.
AVX-512 (`vpgatherqq`, via `_mm512_i64gather_epi64`) does this in one
instruction; AVX2 has a narrower equivalent (`_mm256_i64gather_epi64`).

**Done:** `Simd::gather_bits(table, idx)` (`src/simd/mod.rs`), an `unsafe fn`
with a safe, checked-indexing default — the same per-lane loop every table
kernel already had — overridden for `f64x8` behind
`cfg(target_feature = "avx512f")` and `f64x4` behind `avx2`
(`src/simd/wide_backend.rs`). `#![deny(unsafe_code)]` added to `src/lib.rs`
with `#[allow(unsafe_code)]` only on the trait default and the two overrides;
every call site goes through a tiny safe per-kernel wrapper (`exp.rs`'s
`gather_tab`) whose doc comment carries the index-bounds argument next to the
arithmetic that makes it true. Verified independently before touching any
kernel: 1M randomized indices against both overrides, exact match every time.

Wired into `exp`'s `BitExact` path (both table reads). Measured
(`examples/dbg_gather_bench.rs`, a throwaway A/B harness, both
implementations timed in one process to remove cross-run noise): **f64x8
+22-23%, f64x4 +11-12%, f64x2 and scalar ~unchanged** (within ±1%, from the
gather step being restructured out of the original single loop into three —
index computation, two gathers, post-processing — which has no hardware
gather to amortize that restructuring against at those widths).
`examples/bench.rs`'s end-to-end `exp` `BitExact` row (which exercises
`Real::Widest`, `f64x8` on this hardware) measured consistently at 2.51-2.63x
against a ~2.20x-2.36x baseline — clears the 15% bar. Bit-exactness
unaffected by construction (same bits, different instruction) and confirmed
by the full `tests/bit_exact.rs`/`tests/glibc.rs` suites, unchanged.

**Rollout to `ln`/`log2`/`pow`/`erf`/`erfc` not attempted this round** — a
deliberate pause, not a rejection. The `exp` prototype answered the
go/no-go question (below) with real numbers; each further kernel needs its
own restructuring (`pow` gathers twice, from two different tables) and its
own A/B verification, which is real, uncompressible work per kernel rather
than a mechanical copy. Left as the next concrete step.

## 4. Workstream B — `Fast`-path speed (mostly `f32`)

The pattern proven by `log2f`/`cbrtf` this round: native f32 arithmetic beats
widening whenever the f64 kernel does more than the f32 result needs.

- **B1. Native `tanhf`** — effort S/M. Currently widens (2.20x). Route:
  `tanh(x) = 1 − 2/(e^{2x}+1)` on the `expf`-fast core, or a rational on the
  folded half-line; target ≤2 ulp, ~3–4x. Exhaustive 2³² sweep is cheap (~30 s)
  and should be added for it, not just sampled checks.
- ~~**B2. Native `erff` / `erfcf`**~~ — **withdrawn.** `src/kernels/single/erf.rs`
  and `erfc.rs` each explicitly document *why* both policies run the same
  code: already correctly rounded, branch-free vector arithmetic, with "no
  cheaper approximation worth having." That is a considered decision, in the
  same category as f64 `cbrt`'s single-algorithm stance — not a gap this
  roadmap should close unilaterally. Moved to §10 as an open decision instead
  of executed.
- **B3. Native f32 trig `Fast`** — effort M. `sinf`/`cosf`/`tanf` `Fast`
  currently widen into the f64 vector trig. An f32 Cody-Waite reduction
  (`PIO2` split already exists in `tables/single/poly.rs`) with the existing
  `SIN`/`COS` f32 polynomials avoids the widen/narrow round-trip entirely.
  Prototype `sinf` first and measure against the widened path honestly.
- **B4. `exp10f` corpus artifact** — effort S, documentation only. The bench's
  ±40 corpus crosses the ±38 main-path limit, so ~5% of lanes take the scalar
  repair under *both* policies; the kernel is fine. Either annotate the bench
  row or add a second in-domain corpus line so the table stops implying a
  regression. (Optionally: asymmetric limits −37.9/+38.5 recover a sliver of
  domain; low value.)

## 5. Workstream C — precision

- ~~**C1. Sub-ulp `Fast` exponentials**~~ — **done**. `exp2`'s degree-1 term
  carried as a double-double (see §6a); measures 1 ulp, matching `exp`. `pow`
  asserts 4 on its two moderate-domain corpora; its extreme-exponent corpus
  stays at 8 (measured a stable 5, not 4 — the amplification there is worst by
  construction, see `pow.rs::fast`'s doc). `exp10`'s bound was not touched —
  it still delegates to the shared 128-entry table's tail and measures 2 ulp,
  already asserted at 2.
- ~~**C2. `tgamma` above 18**~~ — **done**, out of original sequence (no glibc
  disassembly needed — Gamma is "own" work). `stirling_dd` carries
  `ln(Gamma(z))` in double-double via `pow`'s own `pow_log`; `hi + lo`
  verified byte-for-byte against the platform's `lgamma`. 2037 → 512-513 ulp,
  stable at 100M samples. The <64 target in the original estimate was not
  reachable: `exp` of the platform's own correctly-rounded `lgamma` already
  differs from the platform's own `tgamma` by this much — glibc's `tgamma`
  is provably not `exp(lgamma(x))` composed through one rounding, so closing
  the remaining gap needs a from-scratch large-argument algorithm, not a more
  careful logarithm. README caveat updated to state this rather than delete
  it.
- ~~**C3. `lgamma` negative half-line`**~~ — **checked, left alone.** Measured
  4.3e-14 absolute at 100M samples — already an order of magnitude inside the
  asserted 1e-12, with headroom rather than sitting at the edge. Not worth
  the double-double `sin(πx)` work this predicted; the existing bound is
  honest as stated.
- ~~**C4. `asin`/`acos` 8 → 4 ulp**~~ — **done** (Phase 1, see §6a); this entry
  was stale — left in place until now only because nothing had re-swept the
  workstream list after that phase landed.
- ~~**C5. `log2f` 3 → 2 ulp**~~ — **done**. Compensated the `m + 1` rounding
  in `s = (m - 1)/(m + 1)` the same way `pow.rs::log2_dd` compensates its own
  division, at `f64` (see `src/kernels/single/log2.rs`). Verified by an
  exhaustive sweep of every positive-normal `f32` — not a sample — via the
  new `tests/ulp_scan.rs::scan32_exhaustive` helper: worst case 2 ulp over all
  2,130,706,432 positive normals. The asserted bound in `tests/accuracy.rs`
  stays at 4 (it already had 2x headroom over the new measured worst, the
  same ratio the crate's `cbrt` precedent uses).

## 6. Workstream D — infrastructure that keeps the above honest

- **D1. Permanent ulp-scan harness** — effort S. This round's throwaway probe
  (worst-ulp over configurable corpora, 30M samples) gets rebuilt every time
  it's needed. Add `tests/ulp_scan.rs` behind `#[ignore]` with env-var corpus
  selection, so "measure before asserting" is one command.
- **D2. Bench regression tracking** — effort S. Teach `examples/bench.rs` an
  optional machine-readable output (CSV on a flag) plus a tiny diff script in
  `tools/`, so a 2.30x → 2.07x trade like this round's `pow` is a recorded
  decision, not archaeology.
- **D3. `gen_tables.py` extension for the trig tables** — effort S/M,
  prerequisite of A1/A4. Same provenance discipline as the existing tables.
- **D4. Corpus hardening** — effort S. The `pow` extreme corpus added this
  round caught what the wide corpus missed. Add the analogous edges elsewhere:
  subnormal-result `exp`/`exp2`/`exp10`, `sinh`/`cosh` near ±710, `hypot` near
  overflow, trig near the `TRIG_LIMIT` reduction boundary.

## 6a. Progress log

- **Phase 0 done.** `tests/ulp_scan.rs` (permanent, `#[ignore]`d, env-var
  sample count), `tools/bench_diff.py` + `examples/bench.rs --csv`, and the
  `exp10f` bench corpus fixed to stay in-domain. Also established this
  machine's bench noise floor empirically: ~10-12% run-to-run on identical
  binaries for the small/cheap rows (`floor`, `round`), so single-run bench
  diffs under that are not signal — the sequencing note in the process below
  reflects this.
- **Phase 1 done.** `exp`/`exp2`/`exp10` `Fast` paths restructured so their
  series carries no leading 1 and the final combine with `scale` is one true
  FMA instead of a separate rounding (same or fewer operations, verified
  against the noise floor); `exp` measures 2 -> 1 ulp. Added the f64
  `exp`/`exp2` `Fast` accuracy assertions that did not previously exist at
  all — a real gap, now closed at bound 4. `asin`/`acos` tightened from an
  asserted 8 ulp (measured 3/2, a stable structural worst case at the fold
  boundary, not a rare outlier) to 4. `pow` re-verified within its own
  session's new bound of 8 (measured <=5) after the exp2 change. `asinh`
  checked and left alone — its 8-ulp bound already has honest headroom
  (measured 4).
- **Phase 2 done, with one item withdrawn.** Native `Fast` kernels added for
  `tanhf` (direct polynomial below `|x|=1`, `1-2/(e^{2x}+1)` above it via the
  native `f32` `exp2`, saturated tail past `|x|=9` to the reference — 4.97x)
  and `sin`/`cos`/`tan`f (a straight port of the `f64` kernel's Cody-Waite
  shape to native `f32` arithmetic — 17-18x / 7.4x). **B2 (native
  `erf`/`erfc`f) withdrawn**: both modules already state, as a considered
  decision, that no cheaper approximation is worth having — see §10 item 4.
  One real bug found and fixed along the way, twice: reusing `f64`'s
  `TRIG_LIMIT`-derivation formula naively for `f32` gave a domain
  (`|x|<6434`) where the reduction's own truncation error is only 36 bits
  against `f32`'s 24-bit mantissa — comparable to `f32`'s own ulp — which
  `tests/ulp_scan.rs` correctly caught as a multi-million-ulp "failure" near
  a zero of `sin`/`cos`. Fixed by narrowing `f32`'s reduction `trailing`
  parameter (12 -> 9 bits) for real headroom, not by weakening the kernel;
  `TRIG_LIMIT` is now ~804 instead of ~6434. The residual near-zero ulp
  blowup that remains at very high sample density is inherent to evaluating
  any trig function near its zeros (same phenomenon as `lgamma`'s), not a
  defect — documented in the kernel and the scan harness so it doesn't read
  as a regression later. Also caught and fixed a second, unrelated instance
  of the same *class* of bug: `tanhf`'s first cut benched at 0.87x (a
  regression) purely because the crate's existing `-40..40` bench corpus
  crosses tanh's saturation point (~9) — same failure shape as the `exp10f`
  fix from phase 0, now three instances of the same lesson.
- **Phase 3 attempted, redirected after disassembly.** Went looking for the
  "easy" `BitExact` ports (`hypot`, `log10`) per the original phase 3 plan.
  Disassembled both from `/lib64/libm.so.6` (glibc 2.43) and found the
  original effort estimates wrong in the same direction both times: `hypot`
  is a compensated 2Sum/2Product algorithm that specifically *avoids* FMA
  (fusing would break its error-extraction identities, not just change the
  last bit), with at least three scale-dependent branches; `log10` opens
  with an integer-domain bit-test dispatch and a ~64-entry two-row table gather
  at least as large as `pow`'s. Neither is the "friendliest shape" or
  "smaller job" the original plan assumed — both are genuinely comparable to
  `erf`/`erfc`'s double-double work. Rather than rush a port of delicate
  compensated arithmetic under time pressure — where a subtle error produces
  a plausible-looking but silently non-bit-exact result, the one failure mode
  this crate's whole design exists to prevent — this round stopped after
  reconnaissance and recorded the findings (§ A2, A3) rather than shipping an
  under-verified port. Redirected the remaining time to lower-risk work
  verifiable with existing tooling.
- **C2 (`tgamma` above 18) done, out of order — no glibc disassembly needed**,
  since Gamma is "own" work with no bit-exactness claim to risk. Added
  `stirling_dd`, computing `ln(Gamma(z))` in double-double by reusing `Fast`
  `pow`'s own `pow_log` (already exercised by `pow`'s bit-exactness tests,
  not a second unvalidated implementation) rather than `stirling`'s
  single-rounded `ln`. Verified byte-for-byte against the platform's own
  `lgamma` at the worst point found (`hi + lo` compressed to `f64` equals
  `libm::lgamma(x)` exactly). One real bug on the way: initially tried
  feeding `(hi, lo)` through `pow`'s internal accurate exponential
  (`pow_exp`) directly, which produced `NaN` above `z ~= 132` — `pow_exp` is
  only ever called by `pow` itself for exponents under 512, a precondition
  `pow`'s own code enforces but the function itself does not check, and
  `tgamma`'s domain needs up to ~710. Fixed by recognizing `tgamma` (unlike
  `pow`) has no further amplifying factor to carry `lo` *through* — it only
  needs `exp` of one accurate value — so `hi + lo` compresses to a single
  correctly-rounded `f64` and feeds the ordinary `exp` kernel, which already
  covers the whole domain via its own `FullRange` fallback. Net result:
  worst case 2037 -> 512-513 ulp, stable at 100M samples. The remainder is
  provably not fixable by this route: `exp(hi+lo)` computed with the
  platform's own `exp` exactly equals `exp` of the platform's own `lgamma`,
  yet both differ from the platform's own `tgamma` — glibc's `tgamma` is not
  `exp(lgamma(x))` composed through one rounding, and neither can this
  crate's be without a from-scratch large-argument algorithm (out of scope
  here). C3 (`lgamma` negative half-line) checked and left alone: measured
  4.3e-14 absolute at 100M samples, already an order of magnitude inside the
  asserted 1e-12.

- **Review pass over everything above.** Load-bearing numerical claims were
  re-derived rather than taken on trust, and verified computationally where
  that was possible:
  - **`trig.rs` reduction exactness — verified.** `n * PIO2[i]` is exact for
    every `|n| <= 512`, boundary included (15-bit parts times a 9-bit `n`
    fits f32's 24-bit significand with a bit to spare; `|n| = 512` is a power
    of two and exact regardless). Residual reduction error 2.7e-12, inside
    the claimed `2^-32`.
  - **`cbrt.rs` — verified.** Seed error 0.03424 against the documented
    0.0343; three Newton steps land at exactly 1.00 ulp, confirming the
    stated bound and that the residual is the final narrowing, as claimed.
  - **Discarded branches — verified safe.** Both `tanh` bands compute
    unconditionally before `select`; the polynomial evaluated far outside its
    fit interval stays finite (2.5e9 at `|x| = 9`), and the large-branch
    `exp2` argument on a small lane stays deep inside its `Finite` domain.
    No inf/NaN can poison a discarded lane.
  - **Double-double preconditions — checked by hand.** Every Fast2Sum in
    `log2_dd` and `stirling_dd` holds over its actual domain. Two are worth
    recording: `log2_dd`'s final `e + v` has `e = 0` for `x` in
    `[1/sqrt2, sqrt2)`, which violates `|a| >= |b|` but is still exact
    (`hi = v`, residual 0); and `stirling_dd`'s `HALF_LN_2PI - (s + z)`
    recovers the exact residual only because `s + z` is itself exact by
    Sterbenz over `z >= 18`.
  - **Two defects found and fixed.** `examples/bench.rs`'s recorder doc
    claimed a `RefCell` while the code used a `Mutex`. And the `tanhf` docs
    (kernel + README) said `tanh` is "indistinguishable from +-1" past
    `|x| = 9`; it actually saturates at ~9.02, and at exactly 9 is still one
    ulp short — the code was always correct, since that range goes to the
    reference, but the prose overstated it.
  - **One test weakness found and fixed, which then caught a real
    measurement error.** The f32 trig accuracy corpus was `+-1e4`, so ~92% of
    its lanes landed past `TRIG_LIMIT` and returned bit-exact from the scalar
    reference — barely testing the new kernel. Adding an in-domain corpus
    immediately reported `cos` at 5 ulp, over its bound of 4. Investigated
    rather than relaxed: at `x = -733.56`, `cos` is 5.0e-6 — near a zero —
    and the absolute error is 2.3e-12, far better than an f32 ulp at 1.0.
    The bound was not wrong; the *metric* was. `sin`/`cos` are now held to an
    absolute bound (4 ulp of 1.0) in-domain via a new `check32_abs`, matching
    the precedent `fast_bessel_stays_within_bounds` and
    `lgamma_negative_is_bounded_in_absolute_error` already set for the same
    reason. `tan` keeps a relative bound, being unbounded.
- **C1's remaining lever closed.** `exp2`'s `Fast` degree-1 term, `r * ln(2)`,
  was the one place that kernel took a rounding `exp`'s reduction (by `ln(2)`
  itself) never needed. Carried it as a double-double via `a_mul`
  (`src/kernels/double/exp2.rs`): the exact high part re-enters the Estrin
  chain where the plain product was, the low part folds into the
  lowest-magnitude limb — one extra multiply and one extra FMA. `exp2`
  measures 1 ulp now (`tests/ulp_scan.rs::scan_exponentials`, 100M samples),
  matching `exp`; both asserted at 2 in `tests/accuracy.rs` (previously no
  bound existed for `exp2` past 4). This tightened `pow`'s two moderate-domain
  corpora from an asserted 8 to 4 (measured 2, stable to 400M samples), but
  *not* its extreme-exponent corpus, which measures a stable 5 ulp at 100M and
  400M samples — the amplification there is worst by construction (see
  `pow.rs::fast`'s doc), so it keeps its own bound of 8 rather than sharing
  the other two's. This machine's bench noise floor turned out to be far
  above the ~10-12% recorded in Phase 0 — repeated identical-binary runs swing
  entire unrelated rows (`floor`, `round`) by 20-70% — so `tools/bench_diff.py`
  could not cleanly gate this change here; `exp2`/`exp`/`pow`'s `Fast` rows
  were instead checked across several repeated runs and stayed in their
  pre-change bands (exp2 ~3-3.5x, exp ~3.8-4.6x, pow ~1.8-2.2x) with no visible
  collapse. A machine with a quieter bench floor should re-run
  `tools/bench_diff.py` for a precise number.
- **`log2f` 3 -> 2 ulp (C5) closed the same session.** See the C5 entry in
  §5, above — no separate write-up needed here beyond noting it used the same
  compensated-division technique as C1, at `f32` instead of `f64`, and the
  same "measure before asserting" discipline via a new permanent exhaustive
  `f32` sweep helper (`tests/ulp_scan.rs::scan32_exhaustive`), not a sample.
- **`bessel.rs`'s missing FMA explanation resolved by disassembly, not
  arithmetic change.** `rational_p`/`rational_q`/`j0_num`/`j0_den` had no
  comment explaining their absence of `mul_add`, unlike `exp2`/`exp10`'s
  explicit "glibc ships no `_fma` ifunc" note. Disassembled `__j0_finite` from
  the host's `libm.so.6` (glibc 2.43, `objdump -d`): zero `vfmadd`/`vfnmadd`/
  `vfmsub` instructions anywhere in the function, near-origin rational and far
  asymptotic branch alike — confirmed the same "no `_fma` ifunc" reason, not
  an oversight. Documented in the module doc; no arithmetic touched, per the
  original plan for this item.
- **Native `Fast` `cbrt` at `f64` landed, with an honest bound looser than
  first estimated.** A seed-plus-Newton kernel (`src/kernels/double/cbrt.rs`)
  resolving §10 item 3: bit-pattern seed for `|x|^(-1/3)` (constant found by
  the same search technique as the `f32` kernel's, scaled and refined
  numerically — see the kernel doc), four division-free Newton steps, a
  compensated final combine. Measures **7-8x against `BitExact`'s 1.7x** — well
  past the 3x acceptance bar. Its accuracy bound is **16 ulp** (measured worst
  8, stable from 100M to 500M samples), not the ~2-4 ulp first estimated: the
  `f32` kernel's equivalent gets its extra precision for free by running its
  Newton steps in widened `f64` lanes, but there is no wider type to widen an
  `f64` Newton iteration into, so its seed-plus-Newton `r` bottoms out at
  about one `f64` ulp of its own residual, and squaring that (`x * r²`) roughly
  doubles it before any rounding is even considered. A compensated final
  combine (`a_mul` twice) recovers only the combine's *own* rounding, moving
  the measured worst case from 9 to 8 ulp — real, but small next to the
  residual-doubling it cannot touch. Reaching the originally-hoped low-single-
  digit bound would need carrying `r` itself in double-double through the
  final Newton step, at a cost that erodes most of what makes this path fast
  in the first place; not attempted, since 7-8x at a documented, asserted 16
  ulp is already a legitimate, honestly-characterized trade — the same kind
  every other `Fast` bound in this crate makes.
- **`erff`/`erfcf` native `Fast` experiment, split outcome, and two real bugs
  found along the way.** The shared shape: below `ERF_SPLIT` (0.75), an odd
  minimax series `x P(x^2)` (`ERF_NEAR`); above it, `exp(-x^2) Q(z)` with
  `z = (x - A0)/(x + B0)`, mirroring the platform's own compression but
  refit directly in `f32` (`tools/gen_poly.py`'s `remez`, reused as-is).
  Three real precision problems surfaced and were fixed in turn, in order of
  how badly they broke things:
  1. **`exp2`f's `Fast` path returning `+-inf` or 0 near its own domain's
     subnormal-result edge.** `exp(-x^2)` for `x` near the far domain's upper
     end reduces to an `exp2` call whose argument approaches -127 — inside
     `exp2`'s own documented `|x| < 128` `Finite` range, but its scale's
     exponent field was built with `k.wrapping_add(127) << 23`, which only
     produces a *normal* result and wraps to garbage once the true result
     goes subnormal. Not hypothetical: it produced a 2*10^9-ulp "error"
     immediately. Confirmed the identical bug, by inspection, in `exp`f and
     `exp10`f — both share the construction, and both have their own narrow
     subnormal-adjacent bands inside their stated domains (`exp`f near
     `|x| = 87.3-88`, `exp10`f near `|x| = 37-38`; `exp2`f's own f64
     siblings never hit this, because their `Finite` limits were chosen
     conservatively enough to stay clear of it, unlike the `f32` ones). Fixed
     in all three by building `2^(k + 25)` (never subnormal, for any `k`
     these domains admit) and multiplying by the exact power of two `2^-25`
     separately — free of additional rounding, since both factors are exact.
     New permanent regression corpus:
     `tests/accuracy.rs::fast_single_precision_handles_subnormal_results`.
  2. **A coefficient-splitting bug in the new `gen_poly.py` code itself.**
     `exp(-x^2)`'s argument (`x^2 * log2(e)`) needed a compensated,
     double-single reduction to stay accurate once `x^2` reaches ~101 (`erfc`'s
     domain runs to `x ~= 10.05`) — a single `f32` rounding of the constant
     alone cost up to 119 ulp of the final result. The fix (`LOG2E_HI`/
     `LOG2E_LO`, `exp_neg_x2` in `erf.rs`) needed `HI` rounded to genuine `f32`
     precision before computing `LO` as its residual; the first version
     rounded to Python's native `float` (`f64`) instead, silently, which
     left `LO` correcting the wrong residual and made the error *worse*
     (2*10^9 -> 119 ulp only, not close to fixed). One-line fix once found
     (`to_f32`, a real `struct.pack`-based round-trip).
  3. **A single wide-domain minimax fit not evaluating well in `f32`, even
     though it fit well mathematically.** With (1) and (2) fixed, `erfc`'s
     far branch still measured up to 23 ulp: `Q`'s Remez error was ~9e-9 at
     degree 8, but represents a function falling by two orders of magnitude
     across its domain, and the f32-rounded coefficients did not evaluate
     that well in practice regardless of the fit's own mathematical quality.
     Split into two regions at `x = 2.5` (`ERFC_FAR_LO`/`ERFC_FAR_HI`),
     mirroring the platform's own two-Chebyshev-fit shape rather than
     reusing its split point — roughly halved the worst case (23 -> 11 ulp).
  With all three fixed, `erf`f measured 2 ulp *exhaustively* (every positive
  normal `f32`, `tests/ulp_scan.rs::scan_single_precision_exhaustive`) at
  4.1x — accepted, asserted in `tests/accuracy.rs::fast_erf_stays_within_bounds`.
  `erfc`f, sharing the same fixes, still only measured 11 ulp at 1.51x (its
  unfixed baseline was already 1.40x) — killed per the plan's own criteria
  (past the 4 ulp bound, and barely past the 1.5x speed bar), reverted to the
  shared/`BitExact` code, with the measurement recorded in its module doc
  rather than silently dropped. See §10 item 4.
- **`src/kernels/exact.rs` vectorisation attempted, and the premise it was
  chasing turned out false on inspection.** This was flagged (outside the
  original roadmap) as a cheap win: `frexp`, `nextafter`, `fmod` and
  `remainder` run per-lane, and the module's own doc claimed "writing them
  lane-at-a-time keeps them branch-free, which is what lets LLVM vectorise
  them anyway" — implying the scalar loop already compiles to real SIMD.
  Checked rather than trusted: disassembled the compiled `frexp` (`objdump
  -d` on a `#[no_mangle]` probe at `f64x4`) and found real per-lane branches
  and scalar `vmovsd` moves, not packed instructions — LLVM is not
  vectorising it. The deeper reason closes the door on a hand-written fix
  too: these functions extract or rebuild an IEEE exponent field, which
  needs an integer *shift*, and [`crate::simd::Simd`] has no packed integer
  shift anywhere in its surface — `and_bits`/`or_bits`/`xor_bits` are
  lane-wise bitwise ops on the float representation, and `Simd::Bits` is a
  plain `[Uint; LANES]` array with no arithmetic of its own. Every kernel
  elsewhere in the crate that needs a shift (table indices, exponent field
  construction in `exp`/`exp2`/etc.) pays for it with the exact same
  per-lane loop pattern; `frexp` and friends are simply *all* shift work,
  with no floating-point tail left over to vectorise once it is done.
  Genuinely fixing this would mean adding a shift primitive to `Simd` and
  implementing it across all six backends (`f64x2/4/8`, `f32x4/8`, scalar) —
  a real architectural change, not a kernel rewrite, and out of scope for
  what this item was budgeted as. The stale doc comment was corrected
  in-place (`src/kernels/exact.rs`) to state what was actually verified
  rather than assumed. `fmod`/`remainder`/`remquo`'s variable-trip reduction
  loop was not separately prototyped: it rests on the identical primitive
  gap, so the outcome would be the same. No code changed; the finding is the
  deliverable.
- **D4 corpus hardening, sinh/cosh only.** `src/kernels/double/hyper.rs`'s
  `sinh`/`cosh` have their own genuine vectorised bit-exact schedule (unlike
  the many functions here that simply delegate) but had no dedicated
  `tests/bit_exact.rs` corpus before this — only the generic `universal()`
  values and whatever `Fast`-only sampling elsewhere happened to cross. Added
  `corpus_hyper` (`tests/bit_exact.rs`), hardening the gap between `OVERFLOW`
  (710.0, the kernel's own domain boundary) and the true mathematical
  overflow threshold (~709.78) specifically — the band a boundary-placement
  mistake would hide in. Both pass at every width; no bug found, but a real
  coverage gap closed. `exp`/`exp2`'s subnormal-boundary gap was the same
  *shape* of gap and did have a bug (see the `erff`/`erfcf` entry above) — the
  other D4 items (`hypot` near overflow, trig around `TRIG_LIMIT`) were not
  attempted this round; trig's `BitExact` fully delegates to the scalar
  reference (`map_lanes`), so it is provably correct by construction and a
  dedicated corpus would mostly test `map_lanes` itself, already covered
  generically.
- **`tgamma`'s dual-compute blend replaced with a whole-vector guard.**
  `gamma.rs::positive` computed both the `TG_STEPS` recurrence and
  `stirling_dd` + `exp` unconditionally for every vector, mirroring
  `bessel.rs::branch`'s reasoning for why that is wasteful when the whole
  vector already agrees on which side of the cutoff it is on. Split into
  `direct`/`via_exp`, guarded on `is_direct.all()`/`.none()` with the mixed
  case still blending both exactly as before — a speed change only, and
  `tests/ulp_scan.rs::scan_gamma` confirms it: 512/513 ulp, unchanged to the
  last digit. New bench rows (`examples/bench.rs::tgamma_bench`, two corpora
  either side of `TG_DIRECT_LIMIT` so the guard's win shows directly rather
  than blending into one number) measure 8.4x on the recurrence-only corpus
  and 4.6x on the Stirling-only one.

## 7. Suggested sequencing

Ordered by value density; each phase is independently shippable and ends with
the full verification protocol (§8).

| phase | contents | effort | headline outcome |
|---|---|---|---|
| **0** | D1, D2, B4 | S | measurement tooling in place; bench table honest — **done** |
| **1** | C1, then re-tighten `pow`; C4; C5 if touched | M | `Fast` exponentials ~1 ulp, `pow` ≤4 asserted, inverse trig ≤4 — **done** |
| **2** | B1, B3 (B2 withdrawn — considered decision) | M–L | native f32 `Fast`: `tanhf` 4.97x, `sin`/`cos`f 17-18x, `tan`f 7.4x — **done** |
| **3** | A3, A2, D3 | **L, not M–L** | corrected after disassembly (§ A2/A3): both are genuinely hard, comparable to `erf`/`erfc`, not "easy targets" — **not attempted this round, see below** |
| **4** | A1 (`sincos` → `sin`/`cos` → `tan`), then A4 | XL | the trig family bit-exact *and* faster — the crate's biggest remaining claim |
| **5** | A5 gather experiment; ~~C2, C3~~ done | M | either a fleet-wide `BitExact` uplift or a documented negative result; `tgamma` 2037→512 ulp — **C2/C3 done** |

Effort legend: S ≈ under half a session, M ≈ a session, L ≈ several, XL ≈ a
multi-session project with its own intermediate milestones.

**Since this table, done in one extended session:** C1's remaining lever
(`exp2` → 1 ulp, `pow` → 4), C4/C5 confirmed done, native f64 `cbrt` `Fast`
and native `erf`f `Fast` (both §10 open decisions, both resolved — see §6a),
two real bugs found and fixed along the way (`exp`/`exp2`/`exp10`f subnormal
underflow; a `gen_poly.py` coefficient-splitting bug), the `exact.rs`
vectorisation question closed as a documented architectural limit rather than
attempted blind, `sinh`/`cosh` D4 corpus hardening, and the `tgamma`
dual-compute guard (8.4x/4.6x). None of this was on the table above when it
was written; §6a has the full accounting for each. A5 (gather) is next.

## 8. Verification protocol (applies to every item)

1. **Before:** record baseline bench rows and worst-ulp (D1 harness) for every
   touched function.
2. **Bit-exact ports:** corpus = branch boundaries ±2 neighbours, subnormals,
   specials, ≥7M random bit patterns, at every vector width; move the function
   from `tests/delegating.rs` to `tests/bit_exact.rs`; f32 ports additionally
   run the exhaustive 2³² sweep.
3. **`Fast` kernels:** measured worst over ≥30M samples per corpus, asserted
   bound = measured rounded up with power-of-two headroom, in
   `tests/accuracy.rs`; kernel doc and README state the same number.
4. **Tables/coefficients:** regenerate from the tool, confirm byte-identical
   for untouched arrays.
5. **After:** full `cargo test --release`, clippy, `cargo doc` warning-free;
   bench rows updated in the README only from a run on the reference machine.

## 9. Risks

- **Schedule-reading risk (A1/A4):** the FMA placement problem scales with
  branch count; budget for the disassembly step to dominate, and land one
  function at a time. The test suite is designed to fail loudly — trust it.
- **glibc version drift:** a port is bit-exact to *this* host's library;
  that is already the crate's stated contract (`tests/bit_exact.rs` arbitrates
  on other platforms), but each new port widens the surface. Keep the
  per-function corpus cheap enough to run everywhere.
- **Speed/precision trades:** C1 and A5 can each go either way; both carry an
  explicit accept/reject threshold above, so a regression is a decision, not
  a surprise.
- ~~**f64 `cbrt` `Fast`:** deliberately out of scope~~ — resolved, see §10
  item 3 and §6a.

## 10. Open decisions

1. **Priority weighting:** phases 1–2 (precision + f32 `Fast`) before the big
   `BitExact` ports, as sequenced — or is default-policy throughput (phases
   3–4) the user's actual pain point? The sequencing above assumes value
   density; a user running everything under `BitExact` should invert phases
   2 and 3/4.
2. ~~**A5 gather backend**~~ — **decided: go, on `exp` — wider rollout still
   open.** Prototyped, measured (+22-23% on `f64x8`, +11-12% on `f64x4`),
   and accepted; `#![deny(unsafe_code)]` added with the surface confined to
   the trait default and two overrides, per the original ask. Whether to
   extend it to `ln`/`log2`/`pow`/`erf`/`erfc` is the now-open follow-on
   question — see §5's A5 entry.
3. ~~**f64 `cbrt` `Fast`**~~ — **decided: offered.** A native seed-plus-Newton
   `Fast` path landed (`src/kernels/double/cbrt.rs`), measuring 7-8x against
   `BitExact`'s 1.7x. Its bound is 16 ulp (measured worst 8, stable from 100M
   to 500M samples at extreme magnitudes), looser than the ~4 ulp originally
   estimated: unlike the `f32` kernel, there is no wider type for the Newton
   iteration to run in, so the last squaring pays for `r`'s own one-ulp
   residual twice over, and a compensated final combine only recovers the
   combine's own rounding (9 -> 8 ulp), not that residual. See §6a and the
   kernel's own doc for the full accounting.
4. ~~**`erff` / `erfcf` `Fast`**~~ — **decided, split: `erff` offered, `erfcf`
   kept as-is.** Both got the same native attempt (a shared two-region
   minimax fit, `src/kernels/single/erf.rs`'s `exp_neg_x2`/`far_q`), and both
   were measured, not assumed. `erff`: 4.1x at 2 ulp worst case, exhaustive
   over every positive normal — accepted, asserted in `tests/accuracy.rs`.
   `erfcf`: only 1.51x (barely past its unchanged 1.40x) at 11 ulp — killed,
   the code reverted, `erfc.rs`'s module doc records the measurement so a
   future attempt does not re-derive it. The difference was domain, not the
   fit: `erfc`'s own `Finite` domain reaches ~10.05 with no saturating
   shortcut, so every lane pays the far branch's full cost, where `erf` only
   pays for it up to its own ~3.92 saturation point. See §6a for the fuller
   accounting, including two real, independent bugs this work found and fixed
   along the way (`exp`/`exp2`/`exp10`f's native `Fast` paths returning `+-inf`
   or 0 instead of a correct subnormal result, and a coefficient-splitting bug
   in `tools/gen_poly.py` that rounded to `f64` instead of `f32`).
5. **`jn`/`yn` and f32 Bessel `BitExact`:** the README's reasons for leaving
   them scalar remain valid; this plan proposes no work there unless a use
   case surfaces.
