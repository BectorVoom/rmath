# rmath optimisation roadmap

A detailed plan for the next rounds of speed and precision work, written against
the measured state of the crate on 2026-08-19 (AMD Ryzen AI 7 350, Zen 5,
AVX-512, glibc x86-64, `-C target-cpu=native`). Every number quoted here was
measured on that machine with `examples/bench.rs` (best-of-8 over 1M elements)
or the ulp methodology of `tests/accuracy.rs`.

## 1. Where we stand

The round of work just completed closed four gaps:

- `tan` `BitExact` stopped delegating: **0.98x → 2.77x**, bit-exact (the
  last member of the A1 trigonometric family, see §6a).
- `Fast` `pow` tightened from **40 ulp to 8 asserted / ≤4 measured** (the
  table-free logarithm is now carried in double-double throughout), at the cost
  of 2.30x → 2.07x throughput.
- `log2f` `Fast` got a native table-free path: **1.57x → 3.38x**, ≤3 ulp.
- `cbrtf` `Fast` got a native seed-plus-Newton kernel: **0.96x → 2.86x**, ≤1 ulp.

What remains, grouped by what it costs the user today:

| gap | today | ceiling |
|---|---|---|
| `tan` `BitExact` — ported **and** vectorised (§6a A1) | done: 2.77x | — |
| `atan`/`atan2` `BitExact` — ported **and** vectorised (§6a A4) | done: 1.28x / 2.74-3.03x | — |
| `asin`/`acos` `BitExact` — ported **and** vectorised (§6a A4); `acos`'s near-1 gap fixed, see §6a | done: 0.83-0.85x exact (just under parity — 13-slot per-lane gather is the cost), `Fast` 8.4-8.5x / 10.2x `+Finite` | parity still open — A5's hardware gather was tried and measured slower (see §3) |
| Hyperbolic inverses `BitExact` delegate | 0.97–1.00x | `Fast` reaches 8–13x |
| `log10` `BitExact` — ported **and** vectorised (§6a A2), reusing `ln`'s own table | done: 2.61x | — |
| `log1p`, `hypot` `BitExact` — ported **and** vectorised (§6a A2/A3) | done: `log1p` 1.13x (1.31x `+Finite`), `hypot` 1.93x | — |
| Table gathers are per-lane scalar loops | `exp` `BitExact` 2.36x vs `Fast` 4.02x; `asin`/`acos` `BitExact` 0.83-0.85x — the one case under parity, see §6a | hardware gathers could close part of that spread and push `asin`/`acos` past parity |
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

### A1. Trigonometric family port (`sin`, `cos`, `sincos`, `tan`) — **all four done**

The headline job, and it was the largest. glibc computes these with the IBM
Accurate Portable Math Library routines.

**Done** (see §6a for the full accounting): schedule read out of a
disassembly of `__sin_fma`/`__cos_fma` (glibc 2.43, x86-64), not the C source
— four separate FMA-placement bugs were found and fixed by iterating against
that ground truth, not by re-reading the source more carefully. Reference
port in `src/reference/double/trig.rs` (independently verified: 0 mismatches
against the platform across tens of millions of inputs). Vector kernel in
`src/kernels/double/trig.rs`'s `bit_exact` submodule: whole-band blending
(the `logx.rs` near-1 pattern), `__branred` (`|x| >= 105414350`) left as a
`patch_lanes` repair to `f64::sin`/`cos` rather than ported, per the plan's
own risk mitigation. Moved from `tests/delegating.rs` to `tests/bit_exact.rs`
with a dedicated corpus (branch boundaries, quadrant boundaries out to and
past the table limit, subnormals, adversarial bit patterns). Measured:
sin/cos `BitExact` **~2.97-3.01x** (from ~1.0x), `sincos` **~3.59-3.72x**
(higher because it shares one reduction across both outputs) — within the
plan's own 2-4x estimate.

**`tan` done too** (see §6a). The one wrinkle the deferral noted — `xfg`'s
declared shape (`xfg[186][4]`) vs. how `s_tan.c` indexes it (`xfg[i][0..2]`)
— resolved on the schedule-reading pass: `[3]` is a fourth column only the
*cotangent* rows populate (`FFi`), which `__tan_fma`'s compiled code never
loads, so the generator emits the table as `XFG: [u64; 558]` (`186 x 3`) and
the kernel never touches a phantom fourth column. Same discipline as the
other three: schedule read from the `__tan_fma` disassembly (glibc 2.43, the
FMA ifunc), scalar reference port brute-force-verified against the platform
(0 mismatches over ~58M inputs including 36M banded and all 186 table-index
boundaries), then vectorised with whole-band blending. `|x| > 1e8` (glibc's
own `__branred` Payne-Hanek reduction) stays a `patch_lanes` repair. Moved
to `tests/bit_exact.rs` with `corpus_tan`. Measured: `tan` `BitExact`
**2.77x** — inside the plan's 2-4x, a little under `sin`/`cos` because the
tan schedule carries the extra `DIV2` cotangent path and the heavier
reduction for `|x| > 25`.

### A2. `log10` and `log1p` ports — **`log10` done end-to-end; `log1p` scalar-only**

**The earlier "at least as large as `pow`'s table" finding was stale**,
based on an incomplete trace from a prior session that had broken at the
wrong address. Re-disassembled properly this round (`break main; run;
disassemble /rs __log10_finite`, glibc 2.43, x86-64, no separate FMA ifunc):
`__log10_finite` is only ~80 bytes and is a thin wrapper, not its own table
algorithm. It extracts the unbiased exponent `k` and a rounding-parity bit
`i` from the raw bits, forces `x`'s exponent field to `0x3ff - i` (a
`select` between two constants, not a variable shift), **calls straight into
`__ieee754_log_fma`** — confirmed live, by breaking at the call site and
stepping in — which is the exact table-walk this crate already ported
bit-exact and vectorised as `ln`'s `BitExact` kernel, and combines with
three genuinely unfused operations, `((ivln10 * log(reduced)) + y*log10_2lo)
+ y*log10_2hi`, in exactly the order `e_log10.c`'s own source grouping
reads (a first attempt at reading the disassembly swapped which spilled
product was `log10_2hi` vs `log10_2lo` — the compiler schedules the `lo`
product before the call and the `hi` one after, so reading instruction order
top-to-bottom without checking which literal address held which value gets
it backwards; caught by brute-force verification, not by a second read).
**Ported and vectorised**, reusing `ln`'s existing table rather than
re-deriving one: `src/kernels/double/logx.rs`'s `log10::bit_exact` calls
`crate::kernels::double::ln::bit_exact` (made `pub(super)`) directly on the
reduced argument. Measures **2.61x** `BitExact` (2.27x `+Finite`), verified
bit-exact against the live platform over 19M+ samples (30M random bit
patterns filtered to positive-normal-finite, plus a dense sweep across every
biased exponent). `reference::log10` itself is left as a platform delegate
(only used by the rare-lane `patch_lanes` repair and by tests) rather than
also ported — a deliberate scope call, since the vector kernel's own
independence from the platform is what the throughput claim rests on.

`log1p` disassembled too (`__log1p_fma`): it is genuinely the classic
fdlibm `s_log1p.c` shape as originally predicted — confirmed by finding
glibc's own `s_log1p-fma.c`, which is *the same source file*, `#define`d
and recompiled with `-mfma`; the source is untouched, only the compiler's
fusion choices differ, and those differ a lot (a straight, unfused
transliteration of the C source passed 42M+ random samples with only 63
mismatches, all 1 ulp, all traceable to specific fused operations the
disassembly shows and the source's own grouping does not: the `Lp[]`
polynomial's `R = R1 + z2*R2 + z4*R3 + z6*R4` is a genuine 3-step `fma`
chain — algebraically Horner in `z2`, not the flat sum the source's own
expression suggests — and the final `k*ln2_hi - (...)` combine is one fused
`vfmsub231sd`, not two roundings). **Scalar reference ported and verified**
(`src/reference/double/log1p.rs`), and **vector kernel ported and verified**
(`src/kernels/double/logx.rs`'s `log1p::eval`), extracting per-lane scalar
reduction state and running the FMA polynomial and tail evaluation lane-parallel.
`log1p` moved from `tests/delegating.rs` to `tests/bit_exact.rs` with its dedicated
multi-width corpus (`corpus_log1p`), verified 0 mismatches across 10M+ inputs
at widths 1, 2, 4, 8. Throughput measured: **1.13x exact (1.31x `+Finite`), 11.39x `Fast`**
(14.36x `Fast +Finite`).

Along the way, an existing gap in `log2` was uncovered and fixed: `log2`'s `NEAR_LO`
(`1.0 - 0x1.5b51p-5`) and `NEAR_HI` (`1.0 + 0x1.6ab2p-5`) constants in `src/kernels/double/logx.rs`
were previously typo'd as `0x3fef4a4ef0000000` and `0x3ff016ab20000000` (far too narrow a window),
causing near-1 values to fall through to `main`, and `near_one` had an unfused polynomial
schedule where glibc's `__log2_fma` used FMA Horner fusions. Both were aligned to match
glibc's `__log2_fma` disassembly exactly, resolving all failures in `tests/bit_exact.rs`.

### A3. `hypot` port — **done end-to-end (scalar reference & vector kernel)**

**The earlier "compensated 2Sum/2Product, Dekker-style, `dla.h`" finding was
also stale.** That was glibc's *old* `hypot`; this host's glibc 2.43 ships
the modern (2021+) Borges "MyHypot3" correction
(`sysdeps/ieee754/dbl-64/e_hypot.c`), a considerably smaller and
better-documented algorithm with named constants
(`SCALE = 2^-600`, `LARGE_VAL = 2^511`, `TINY_VAL = 2^-459`, `EPS = 2^-54`).
Confirmed no separate FMA ifunc (`__ieee754_hypot`/`__hypot` share one
address), and confirmed by full disassembly (950 bytes, self-contained, zero
calls out — not "cut off mid-trace" as the earlier session found, just
mis-addressed) that this host selects the algorithm's own `#else`
(non-FMA) branch: every multiply/add/sub is separate, matching the source's
own stated reason — the compensated correction's error-free-transformation
identities depend on separately-rounded arithmetic, and fusing any of it
would silently break the error extraction, not just round differently.

**Scalar reference and vector kernel ported** (`src/reference/double/hypot.rs`,
`src/kernels/double/hypot.rs`): the three branches (huge-`ax` pre-scale, tiny-`ay`
pre-scale, common-case `kernel()`) plus `kernel()`'s own two-branch compensated
correction, all strictly unfused using separate operations. The vector kernel
evaluates Borges "MyHypot3" lane-parallel across all SIMD lanes with branch
selection via `V::select` and `patch_lanes2` for non-finite inputs. Verified
against the live platform over 25M+ pairs across all widths (1, 2, 4, 8) with
0 mismatches. Moved from `tests/delegating.rs` to `tests/bit_exact.rs` (`corpus_hypot`).
Throughput measured: **1.93x exact, 4.82x `Fast`**.

### A4. Inverse trig ports (`asin`, `acos`, `atan`, `atan2`) — effort L each — **done end-to-end**

Same IBM-routine family as A1 (tables `asncs.x`/`cij`, from `asincos.tbl`
and `uatan.tbl`). All four are fully done: scalar reference
(`src/reference/double/invtrig.rs`) and vector kernel
(`src/kernels/double/invtrig.rs`'s `bit_exact` submodule) both ported and
disassembly-verified, measuring 1.28x / 2.74-3.03x / 0.83-0.85x — see §6a's
A4 entry for the full account, including four real bugs found and fixed along
the way (a missing fusion in `acos`'s small-`x` band, a NaN-vs-domain-error
conflation only `tests/delegating.rs` caught, a two-fusion gap in `atan`'s
own `D <= u < E` band only a much denser corpus caught, and `acos`'s near-1
band's Dekker split — `fma(c, 2^27, c)`, which the C source's literal
`y=(c+t24)-t24` does not show — read out of the disassembly and fixed as
`t27`). `asin`/`acos`'s vector kernel is the one in the group whose `BitExact`
path does not beat parity (0.83-0.85x, just under it): their 13-slot
per-lane table gather dominates, which is precisely the cost A5's
hardware-gather backend targeted — and the cost that experiment, measured
end-to-end, failed to remove (see A5's entry).

### A5. Hardware-gather backend experiment — **prototyped on `exp`, rolled out, measured, rolled back**

Every `BitExact` table kernel pays a per-lane scalar loop for its gathers.
AVX-512 (`vpgatherqq`, via `_mm512_i64gather_epi64`) does this in one
instruction; AVX2 has a narrower equivalent (`_mm256_i64gather_epi64`).

**Done (kept):** `Simd::gather_bits(table, idx)` (`src/simd/mod.rs`), an
`unsafe fn` with a safe, checked-indexing default — the same per-lane loop
every table kernel already had — overridden for `f64x8` behind
`cfg(target_feature = "avx512f")` and `f64x4` behind `avx2`
(`src/simd/wide_backend.rs`). `#![deny(unsafe_code)]` added to `src/lib.rs`
with `#[allow(unsafe_code)]` only on the trait default and the two overrides;
every call site goes through a tiny safe per-kernel wrapper (`exp.rs`'s
`gather_tab`) whose doc comment carries the index-bounds argument next to the
arithmetic that makes it true. Verified independently before touching any
kernel: 1M randomized indices against both overrides, exact match every time.

Wired into `exp`'s `BitExact` path (both table reads), where the prototype
measured a +22-23% gain on `f64x8` in an in-process A/B of the gather step
alone and the end-to-end row at 2.51-2.63x against a ~2.20x-2.36x baseline —
accepted on that basis, and still in place. Bit-exactness unaffected by
construction (same bits, different instruction).

**Rolled out this round to every other table kernel** — `ln`, `log2`, `pow`
(two tables, three gathers), `erf`/`erfc` (13-slot rows), and the `asin`/
`acos`/`atan`/`atan2` rows (`cij_row`, the 13-slot `asncs_row`, and
`near_one_root`'s two 1/sqrt tables), each restructured per the `exp`
pattern: per-lane index computation, one vector gather per slot, then the
degree/zeroing logic as vector selects. Bit-exact at every width from the
first compile (the full suite passed; one structural bug — the rewritten
`asncs_row`'s `outer`/`fin` select chains compared the degree against
`7..=10`/`8..=11` instead of `5..=8`, exactly the off-by-degree slip the
per-lane original could not make — was caught by
`tests/bit_exact.rs`'s at-every-width sweep before any measurement).

**Measured — negative, rolled back.** A/B against the pre-rollout code,
round-robin builds timed alternately (four runs each, medians; the same
`examples/bench.rs` end-to-end protocol the prototype's accept used), on the
reference machine (Ryzen AI 7 350, Zen 5, AVX-512):

| kernel | before | with hw gather | Δ |
|---|---|---|---|
| `exp` (kept prototype) | 2.23x | 2.21x | ~0 |
| `ln` | 1.80x | 1.58x | −12% |
| `log2` | 1.93x | 1.75x | −9% |
| `log10` | 2.45x | 2.37x | −3% |
| `pow` | 1.47x | 1.45x | ~0 |
| `erf` | 4.45x | 3.49x | −22% |
| `erfc` | 4.28x | 3.33x | −22% |
| `asin` | 0.84x | 0.74x | −12% |
| `acos` | 0.85x | 0.73x | −14% |
| `atan` | 1.36x | 1.06x | −22% |
| `atan2` | 2.96x | 2.74x | −7% |

Every row moved the wrong way; the biggest regressions track the biggest
slot counts (13 gathers for `erf`/`erfc`/`asin`/`acos`, 7 for `atan`), which
is the hardware gather's latency showing through: the tables all sit in L1,
and a pipelined per-lane load costs less than a `vpgatherqq` whose ~20-cycle
latency the surrounding FMA chain cannot hide. A control run of the same
restructured code with the hardware override disabled (per-lane fallback)
recovered most of the loss for `atan`/`atan2`/`ln`/`log10` but not
`erf`/`erfc` (−1.3x/−0.9x), so the restructuring itself also carries a cost
the `exp` prototype's in-process measurement never had to pay. Notably, even
`exp`'s accepted end-to-end margin does not reproduce on this machine today
(at best a wash); the prototype's +22-23% was a measurement of the gather
step in isolation, and the end-to-end row does not carry it.

**Verdict:** the go/no-go criterion was "clears the 15% bar"; the rollout
clears it in no direction. Reverted to the per-lane idiom across the board
(working tree restored to the pre-rollout kernels; nothing but this record
remains). `Simd::gather_bits` and the two overrides stay — they are the
correct surface for a target where gathers do win, and `exp`'s accepted use
keeps them exercised. The takeaway for any revisit is in the control run,
not the prototype: restructure to gather *or* not, but measure the end-to-end
row, on the target, before committing kernels to either idiom. `asin`/`acos`
parity remains open (0.83-0.85x) — the 13-slot per-lane gather is the cost,
and the hardware gather is not the cure on this hardware.

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

- **A1: `sin`/`cos`/`sincos`/`tan` ported — the trigonometric family
  complete.** The crate's stated-biggest remaining claim. Full pipeline, in
  order:
  - **Table/constant generation.** `tools/gen_tables.py` extended to fetch
    `usncs.h`, `branred.h`, `sincostab.c` and `s_sin.c` from glibc's own
    source tree (not ARM-optimized-routines, the source for every other
    table here) and emit `src/tables/double/trig.rs`: `S1`-`S5` (Taylor
    band), `SN3`/`SN5`/`CS2`/`CS4`/`CS6` (table-band polynomial),
    `BIG`/`HP0`/`HP1`/`MP1`/`MP2`/`PP3`/`PP4`/`HPINV`/`TOINT` (reduction
    constants), and the 440-entry `TAB`. Regenerates byte-identical; verified
    by re-running the generator mid-session (see the note on upstream drift
    below).
  - **Schedule reconnaissance was the dominant cost, as budgeted.** `break
    sin` never fired reliably on the shared library's ifunc-resolved load;
    fixed by computing the resolved runtime address from `info proc
    mappings` and breaking there directly, then reading FMA placement out of
    `objdump -d` on the compiled `__sin_fma`/`__cos_fma` (glibc 2.43,
    x86-64) — not the C source, which under-determines it: several steps
    fuse (or specifically do not) in a way the source's own expression
    grouping does not predict.
  - **Scalar reference port** (`src/reference/double/trig.rs`): `sin`,
    `cos`, `sincos` each reproduce their *own* C entry point's control flow
    rather than sharing one helper — `__sin`'s and `__sincos`'s mid-band
    compute the complementary angle differently (`do_cos(y, hp1)` directly
    vs. a compensated sum formed first), which is not a simplification
    opportunity, it is a genuine difference in what gets rounded when.
    `__branred` (`|x| >= 105414350`) is deliberately not ported — it calls
    straight through to `f64::sin`/`cos`, bit-exact by construction, the
    same "repair-first" choice the plan called for.
  - **Four FMA-placement bugs found by iterating against the gdb trace**,
    not by re-reading the source more carefully: `reduce_sincos`'s tail
    terms unfused, `do_cos`/`do_sin`'s final correction unfused, `taylor_sin`
    unfused, and — the one that would not have been found by pattern-matching
    the other three — `do_sin`'s `c = x*dx + xx*cos_inner(xx)` fuses the
    *opposite* pairing from what a left-to-right reading suggests
    (`xx*cos_inner` rounds separately, then `x*dx` fuses into the add).
    Verification counts after each fix, 5M samples per run:
    3407 → 2736 → 175 → (3, 1, 4) → **0** bad `sin`/`cos`/`sincos` results,
    then confirmed 0/60M and a separate 0/30M small-magnitude stress pass.
  - **Vector kernel** (`src/kernels/double/trig.rs`'s `bit_exact`
    submodule): every scalar primitive re-expressed generically over `Simd`,
    the four bands blended with `V::select` rather than branched (the
    `logx.rs` near-1 pattern) since `Simd` has no scalar control flow to
    branch on; `reduce_sincos`'s quadrant bit `n` is carried as two float
    flags (`n_bit0`/`n_bit1`) rather than an integer, because `Simd` has no
    packed integer arithmetic (the same constraint `exact.rs`'s progress-log
    entry documents). `__branred` and specials repaired via `patch_lanes` to
    the scalar reference. Found and fixed one panic this way: computing
    every band unconditionally means an adversarial/huge `x` still reaches
    the table-index arithmetic for a lane whose result will be discarded, so
    a defensive `% 110` was added to the per-lane index — not part of the
    algorithm, commented as such, and only reachable for inputs the blend
    already discards.
  - **Verified independently at 4 widths** (scalar, `f64x2`, `f64x4`,
    `f64x8`) against 8M adversarial/huge/dense/exact-quadrant inputs: zero
    mismatches. Tests moved from `tests/delegating.rs` to
    `tests/bit_exact.rs` (`corpus_trig`: every band threshold ±2
    neighbours, quadrant boundaries at multiples of `pi/2` both inside and
    pushed to the table-limit edge, subnormals through `1e300`, random bit
    patterns) — the shape `tests/bit_exact.rs`'s own module doc specifies.
  - **Measured** (`examples/bench.rs`, a new `row_pair!` macro added
    alongside `row!` for `sincos`'s two-output `FunctionPair` shape): `sin`
    `BitExact` **2.97x/2.63x** (`+Finite`), `cos` **3.01x/2.58x**, `sincos`
    **3.59x/3.72x** — `sincos` beats `sin`/`cos` taken separately because it
    shares one reduction across both outputs, same asymmetry the platform
    routine itself exploits. All three land inside the plan's 2-4x estimate
    (`tan` was still ~0.98x at that point, deferred — next bullet).
  - **`tan` done on the same pipeline.** The deferral's wrinkle resolved
    first: `xfg`'s declared `[186][4]` shape vs. the `xfg[i][0..2]` indexing
    is not a discrepancy in the algorithm — `[3]` (`FFi`) is only filled for
    the cotangent half-rows `__tan_fma` never reads, so the generator emits
    `XFG: [u64; 558]` (`186 x 3`) and no phantom fourth column is carried.
    `tools/gen_tables.py`'s trig emitter gained the tan section (fetches
    `utan.h`/`utan.tbl`, asserts `MP1`/`MP2`/`PP3`/`PP4`/`HPINV`/`TOINT`
    bit-identical to `usncs.h`'s copies). The scalar reference port
    (`src/reference/double/trig.rs::tan`, bands I-VI, six constants `g1`-`g5`
    plus `gy2`, the `DIV2` cotangent with its `+0.0` term kept, `n`-parity
    via the truncation bit) was brute-force-verified against the platform
    over ~58M inputs — 20M random bit patterns, 36M banded log-uniform, all
    five band boundaries ±8 ulps, every table-index edge `(k+15.5)/256`
    ±4 ulps, 5M dense in `(0.0608, 25)`, 2M subnormals — **0 mismatches**
    (the harness lives at `/tmp/opencode/tancheck`, outside the repo). The
    vector kernel (`bit_exact::tan`) computes all six bands unconditionally
    and blends, with the parity flag carried as a float and the reduction's
    residue (`cvttsd2si` truncation) as per-lane integer casts; `|x| > 1e8`
    stays a `patch_lanes` repair to `reference::tan`. Verified bit-exact at
    every width (scalar, `f64x2`, `f64x4`, `f64x8`) against 8M further
    inputs; tests moved from `tests/delegating.rs` to `tests/bit_exact.rs`
    (`corpus_tan`: the `g1`/`g2`/`g3`/`25`/`1e8` thresholds ±2 neighbours,
    all 186 table-index edges, quadrant parity flips at multiples of `pi/2`
    through `134201344*pi/2`, dense across both reductions' domains so the
    reduced-argument `gy2` split is hit, subnormals, random bit patterns).
    Measured: `tan` `BitExact` **2.77x** (`+Finite` 2.70x), `Fast` 19.63x —
    inside the plan's 2-4x estimate for `BitExact`, a little under
    `sin`/`cos` because the schedule also runs the cotangent's `DIV2` path
    and the second reduction for `|x| > 25`.
  - **Upstream-drift hazard found and worked around, not absorbed.**
    Re-running `tools/gen_tables.py` at the end of this phase (to confirm
    byte-identical regeneration, the standing gate) also re-fetched
    `exp_data.c`/`log_data.c`/`log2_data.c`/`pow_log_data.c` from
    `ARM-software/optimized-routines`'s `master` branch — unpinned, and it
    had genuinely changed upstream since those tables were last generated,
    producing a completely different (still presumably valid, but
    unverified) `TAB`/coefficient set for `exp`/`log`/`log2`/`pow`. That
    diff was **not** part of this phase's work and was reverted
    (`git checkout` on those seven files) rather than silently accepted —
    regenerating a table that was not the subject of the current change is
    not a side effect to absorb quietly. The glibc trig sources
    (`usncs.h`/`branred.h`/`sincostab.c`/`s_sin.c`) had not drifted, so
    `trig.rs` itself regenerated identically. This is a real gap in the
    tooling's reproducibility story — `gen_tables.py` should pin commits
    for every upstream source it fetches, not just document provenance after
    the fact — left as a follow-on rather than fixed under this phase's
    scope.

- **A4: `asin`/`acos`/`atan`/`atan2` — scalar reference ported and verified,
  then all four vectorised (see below for the full account).** Same
  schedule-reading discipline as A1: broke on
  the exported `asin`/`acos`/`atan`/`atan2` symbols, stepped through their
  ifunc-resolved trampolines with `stepi`, and disassembled the real
  entry points (`__ieee754_asin_fma`, `__ieee754_acos_fma`, `__atan_fma`,
  `__ieee754_atan2_fma`; glibc 2.43, x86-64) rather than reading
  `e_asin.c`/`s_atan.c`/`e_atan2.c` and guessing at FMA placement.
  - **Tables.** `tools/gen_tables.py` extended with three new emitters
    (`emit_atan_tables`, `emit_asincos_tables`, `emit_atan2_tables`),
    fetching `atnat.h`/`uatan.tbl` (`atan`'s 241-row `cij` table),
    `uasncs.h`/`asincos.tbl`/`root.tbl`/`powtwo.tbl` (`asin`/`acos`'s shared
    2568-entry `asncs.x` table plus the reciprocal-sqrt seed tables), and
    `atnat2.h` (`atan2`'s own `pi`/`3pi/4`/rescale constants) from glibc's
    raw source tree — same provenance discipline as the trig tables, no
    hand-transcribed numerics. One generator bug found along the way:
    `mynumber_bits`'s regex assumed no whitespace between a `mynumber`
    initialiser's two closing braces (`}}`); `atnat.h` writes `} }` (a
    space), which silently failed to match until the regex was loosened to
    `\}\s*\}`, a strict superset of the old pattern (verified: still matches
    every existing caller). `powtwo` was assumed to live in `root.tbl`
    alongside `inroot`; it is actually `powtwo.tbl`, a separate 28-entry
    (not 128-entry) table — caught by the generator's own assertion, not by
    a silent wrong answer.
  - **Scalar reference** (`src/reference/double/invtrig.rs`, new module):
    `atan` first (establishes the double-double primitives), then `atan2`
    (confirmed by disassembly to *not* call into `atan`'s own code — no
    `call` instruction anywhere in `__ieee754_atan2_fma` — so it
    reimplements the same table/Taylor shape inline, once per quadrant, with
    its own extra `du` division-residual term), then `asin`/`acos` together
    (their eight bands share one table and nearly all of one polynomial
    evaluator). `dla.h`'s `EMULV` macro, under `__FP_FAST_FMA` (true on this
    host), collapses to exactly this crate's `a_mul` two-product shape,
    reused rather than reinvented; `ESUB`'s magnitude-ordered branch was
    confirmed, not assumed, to fold away at compile time in the one call site
    that needs it (`atan`'s `D <= u < E` band, where `w = 1/u <= 1/16 < HPI`
    always holds).
  - **Verification: brute-force against the live platform, not sampling.**
    A throwaway per-function harness (`examples/verify_*.rs`, deleted after
    use — the permanent corpus lives in `tests/delegating.rs`) compared every
    ported function against the real `f64::asin`/`acos`/`atan`/`atan2` over
    20M random bit patterns plus dense boundary/domain sweeps, before any
    vector code was written, exactly as A1's own protocol specifies. `atan`
    and `atan2` matched **on the first attempt** (0/20M) — the disassembly
    reading was thorough enough to need no iteration. `asin` also matched
    immediately. `acos` did not: 5 mismatches in a 4,000,015-point dense
    sweep, all 1 ulp, all in the direct-Taylor band (`|x| < 0.125`). Root
    cause, found by disassembling that specific block: the C source's
    `t = (x2*x)*poly; cor = (...) - t;` compiles to *one* fused
    `vfnmadd132sd` — `t` is never separately rounded — where the initial
    translation had rounded it as its own step (extra rounding, not extra
    precision, but on the wrong side: it made the ported value the *less*
    accurate one, which is what a 1-ulp mismatch against a bit-exact target
    always means). Fixed by fusing the same way the disassembly does;
    re-run confirmed 0 mismatches. A second, unrelated correctness gap
    surfaced only by the *existing* `tests/delegating.rs` suite (not the
    throwaway harness, which only fed in-domain and finite-out-of-domain
    values): out-of-domain input (`|x| > 1`) and NaN input are not the same
    case. The exported `asin`/`acos` *wrapper* (confirmed by disassembly of
    the wrapper stub, not the `_fma` entry point) checks `|x| > 1` itself
    with an ordered `ucomisd` compare — true for finite out-of-range `x`,
    false (unordered) for NaN — so a genuine domain error never reaches
    `__ieee754_asin_fma` at all and returns a fixed canonical NaN through a
    shared error path, while a NaN *input* falls through into
    `__ieee754_asin_fma`'s own final `return (x-x)/(x-x);`, which
    propagates the input NaN's payload and sign rather than producing a
    fresh one. The initial port conflated the two (one `(x-x)/(x-x)`
    fallback for both), which was invisible to 20M random finite samples
    and to a domain sweep that used only `f64::NAN` itself, but not to
    `tests/delegating.rs`'s corpus, which includes structured NaN payloads.
    Fixed by adding the same explicit `if x.is_nan() { return x + x; }`
    early return `atan` already had (and had already gotten right, by the
    same reasoning, on the first pass) ahead of the range dispatch, and
    keeping a literal `f64::NAN` — not a division — for the true
    out-of-domain case. `cargo test --release`, `clippy` and `cargo doc`
    clean after both fixes; full corpus in `tests/delegating.rs`
    (`asin_is_bit_exact`, `acos_is_bit_exact`, `atan_is_bit_exact`,
    `binary_functions_are_bit_exact`'s `atan2` row) passes unmodified — the
    existing infrastructure needed no changes, only a correct reference to
    check.
  - **Wired in:** `reference::double`'s `delegate!` macro invocations for
    `asin`/`acos`/`atan`/`atan2` removed; `reference::double::invtrig`'s
    ports re-exported in their place. `crate::kernels::double::invtrig`'s
    `BitExact` path at that point still used `dispatch`'s `map_lanes` (one
    lane at a time) but called a genuine port instead of the platform,
    closing the gap between this crate's stated goal ("bit-exact
    without calling the libm you're replacing") and what these four
    functions actually did, independent of the vector work below; the
    `asin`/`acos` `BitExact` path has since been replaced by its own vector
    kernel too (see below).
  - **Vector kernel: `atan`/`atan2` done and verified; `asin`/`acos` deferred
    for a real reason found along the way, not a schedule squeeze.** A second
    pass added `src/kernels/double/invtrig.rs`'s `bit_exact` submodule,
    mirroring `trig.rs`'s shape exactly: every band evaluated unconditionally
    and blended with `V::select`, `cij` gathered per lane the same way
    `trig.rs`'s `TAB` and `ln.rs`'s exponent-field extraction already do (no
    packed integer shift anywhere in `Simd`, the same documented constraint),
    a defensive `.clamp(0, 240)` on the gather index for lanes whose real
    band is something else entirely (mirrors `trig.rs`'s `% 110`). Wired in
    the `ln.rs`/`logx.rs` way (`if A::BIT_EXACT { bit_exact(x) } else {
    fast(x) }`) rather than through `dispatch`, since `dispatch` under
    `BitExact` always maps to the scalar reference lane-by-lane — using it
    would have silently kept the "no speed change" state this entry
    previously described. `atan2`'s early-return lanes (not-normal
    arguments, the extreme-exponent-difference short circuit, the extreme-
    magnitude rescale bands) are excluded from the vector path and repaired
    via `patch_lanes2` to the reference, per this crate's own rule that rare
    lanes stay in `reference/`, not the vector main path — detected from raw
    bits per lane rather than a scaled float comparison, because at the very
    magnitudes those exist to guard, a float comparison like `ay >= ax *
    2^57` can itself silently overflow and under-count.
    - **Moved `tests/delegating.rs` → `tests/bit_exact.rs`**: `atan_is_bit_exact`
      and `atan2`'s row in `binary_functions_are_bit_exact` removed;
      `corpus_atan`/`corpus_atan2` added (branch boundaries `A`..`E` ±2,
      `atan2`'s own extreme-exponent and rescale edges, ≥1.6M random and
      adversarial samples), checked at every width this host has (scalar,
      `f64x2`, `f64x4`, `f64x8`) via a new `check_width2`/`check_all_widths2!`
      pair (this file had no two-argument checker before).
    - **A real bug found by the wider corpus, not by the original
      disassembly pass.** `atan_recip_taylor`'s (`D <= u < E`) final combine
      — C's `yy = ((HPI1+cor)-ww) - yy;` — has *two* fusions the earlier
      disassembly read correctly identified in prose but the landed code
      did not actually perform: `inner = (HPI1+cor) - s*w` and `combined =
      inner - wv*yy` were both plain two-rounding arithmetic instead of one
      `mul_add` each. Invisible to the original 20M-random-bit-pattern
      verification (uniform random doubles rarely land with any density in
      `[16, 5.805e15]`, and the miss is a rare rounding coincidence even
      within that band); caught immediately by `corpus_atan`'s dense
      `[-20, 20]` sweep — 1 mismatch in `atan_bit_exact_at_every_width`
      at width 1, `x = -19.315`. Root-caused with a corrected gdb
      methodology (the first attempt read registers *before* the
      instruction that defines them had executed, off by one throughout;
      fixed by reading only at breakpoints placed *after* each value is
      written) confirming `combined = (-wv).mul_add(yy, inner)` and `inner =
      (-s).mul_add(w, hpi1_cor)` against the live trace, then verified with a
      30M-sample sweep concentrated on every band boundary (0 mismatches,
      scalar reference and vector kernel both) before being accepted.
    - **The near-1 bug this entry previously documented as "found and *not*
      fixed" is now fixed and verified.** Root cause confirmed exactly as the
      earlier reading said: `e_asin.c`'s acos near-1 band uses `y = (t27*c+c)
      - t27*c` in **both** arms (a Dekker split via one fused `fma(c, 2^27, c)`,
      `dla.h`'s `CN`), while the port had wrongly transcribed asin's `t24`
      shape `(c + t24) - t24` into acos's positive-`x` arm. The fix is one
      line in `src/reference/double/invtrig.rs` (`y = T27.mul_add(c, c) -
      T27 * c`); `cor = cc + p*(y+cc)` stays unfused, which a first failed
      attempt (537 new mismatches across `[0.97, 0.99]` from fusing it) had
      already shown is required. The decisive check, and the reason the
      earlier "8-in-3M" number is now stronger rather than weaker: a
      three-way split at the last `2^22` ulps below 1 shows the buggy `t24`
      form vs the `t27` fix differ at 283 points, buggy vs platform 283,
      fixed vs platform 0 — the fix is *the* difference, and the corrected
      verification sweep (the first harness used `1u64 << 52` as 1.0's bit
      pattern, which is not it — 1.0 is `0x3ff0_0000_0000_0000` — so its
      "0 mismatches" had swept the wrong range; with the constant fixed, the
      buggy form shows 799 mismatches in the 1e-7-of-1 window and the fixed
      form 0 across ~46M points) confirms both directions. A hostile `t27`
      variant of the same check passed on the first run, unchanged.
    - **Measured** (`examples/bench.rs`, this machine, `RUSTFLAGS="-C
      target-cpu=native"`, two runs): `atan` `BitExact` **1.28x** (`+Finite`
      1.16-1.18x — smaller than plain `FullRange` here, unusual but
      reproduced across both runs; the per-lane gather cost likely already
      dominates over the repair check `Finite` removes), up from ~0.97x
      delegating. `atan2` `BitExact` **2.74-3.03x**, up from ~0.99x —
      squarely inside the "expected parity → 2-3x each" estimate this
      section originally made. `atan`'s own gain is smaller than that
      estimate but still a genuine, verified improvement over the previous
      ~1.0x.
    - **`asin`/`acos` vector kernel done, after the near-1 fix above
      landed first** — the order this entry prescribed. Their `bit_exact`
      submodule (same module, `src/kernels/double/invtrig.rs`) reuses
      `high32`, the near-1 machinery, and the blend discipline, but the six
      index formulas and five degrees of `asncs_band_index` are resolved per
      lane into a single `(n, degree)` pair, so the whole table band is one
      13-slot per-lane gather (`asncs_row`) feeding one unrolled polynomial
      (`asncs_poly`): the 11 coefficient slots beyond a lane's own degree
      are zeroed at gather time and fold as no-ops, so one Horner pipeline
      serves all five degrees — the mask-blend counterpart of the six
      gather-and-select arms the first draft carried, which measured 0.49x
      before this rewrite. The near-1 band keeps `asin`'s `t24` and `acos`'s
      `t27` splits, `|x| == 1` is a select (`+-pi/2` / `0` / `2*pi/2`), and
      the repair mask is NaN only — `|x| > 1` is the canonical NaN the
      reference's wrapper returns, reproduced in-vector, with `asin`'s
      `.copysign(x)` applied before the out-of-domain overwrite so
      `asin(-inf)` stays the positive canonical NaN. Wired in the
      `ln.rs`/`logx.rs` way (`if A::BIT_EXACT { bit_exact(x) } else {
      fast(x) }`).
      - **Moved `tests/delegating.rs` → `tests/bit_exact.rs`** for these two
        as well: `asin_is_bit_exact`/`acos_is_bit_exact` removed;
        `corpus_asincos(seed)` added (the six band boundaries and 0.96875
        threshold each ±2 ulp, 800K uniform `[-1,1]`, 400K dense
        `[0.96875, 1)`, 200K across the last 2^28 ulps below 1 on both
        signs, 200K log-uniform tiny, 300K random bit patterns), checked at
        every width via `reference_*_matches_platform_libm` (the corpus is
        also the reference's own verification now) and
        `*_bit_exact_at_every_width`, plus both functions added to the
        special-lane leak test.
      - **Measured** (same protocol as `atan` above): `asin` `BitExact`
        **0.83-0.84x** (`+Finite` 0.87x), `acos` **0.84-0.85x** (`+Finite`
        0.86x), `Fast` 8.36-8.51x / 10.17-10.27x `+Finite`. The exact rows
        are the one place in this crate's vector table where the `BitExact`
        path lands *just under* the ~1.0x delegating parity it replaces —
        honestly reported rather than dressed up. The cause is measurable,
        not speculative: an experiment removing the table band from the
        blend jumps the exact row to 5.9x, so the 13-slot per-lane gather is
        ~90% of the exact path's cost, and the platform's scalar routine
        never pays for more than one band per input. That is exactly the
        cost A5's hardware-gather backend (`Simd::gather_bits`, prototyped
        on `exp`) set out to remove, and it *is* ~90% of the exact path's
        cost — but A5's end-to-end measurement then showed the hardware
        gather replacing it with a slower instruction on this machine, and
        the rollout was rolled back (see §3's A5 entry); the per-lane idiom
        this pass deliberately stayed with remains the right one here.
        `Fast` is unaffected and still the win.

- **A2/A3: `log10` ported and vectorised; `log1p`/`hypot` ported at the
  scalar level.** Both A2 and A3's prior "genuinely hard, comparable to
  `erf`/`erfc`" findings turned out to be artifacts of incomplete traces
  from an earlier session, not accurate readings of the real algorithms —
  re-disassembled properly this round with the same live-gdb discipline A1
  established (`break main; run; disassemble /rs <symbol>` against a small
  probe binary calling each function, ASLR disabled by gdb's own default so
  addresses stay stable across separate invocations).
  - **`log10` is a thin wrapper, not its own table algorithm.**
    `__log10_finite` is ~80 bytes and calls straight into
    `__ieee754_log_fma` — confirmed live, by breaking at the call site and
    stepping in — which is exactly the table walk this crate already ported
    bit-exact as `ln`'s `BitExact` kernel. Ported as a reduction (exponent
    extraction, forced-near-1 argument) plus a call into
    `crate::kernels::double::ln::bit_exact` (made `pub(super)` for this)
    plus a three-term combine. One real bug: a first reading of the
    disassembly swapped which of the two spilled products was `log10_2hi`
    vs `log10_2lo` (the compiler schedules the `lo` product before the
    call and the `hi` one after; reading instruction order top-to-bottom
    without checking which literal address held which constant gets it
    backwards) — caught immediately by brute-force verification (31874 of
    300015 mismatches, all 1 ulp, every one traced to the swap) rather than
    shipped. Fixed, then verified clean over 19M+ positive-normal-finite
    samples (30M random bit patterns filtered, plus a dense sweep across
    every biased exponent) and again over a further 40M-sample NaN-inclusive
    sweep. **Vectorised and landed**: `BitExact` measures **2.61x** (2.27x
    `+Finite`), up from the ~0.98x delegating baseline — comfortably past
    the "`Fast` reaches 4.7-14x" ceiling this section originally quoted for
    the group as a whole, at zero net-new table cost since it rides `ln`'s.
  - **`log1p` really is the classic fdlibm shape** — confirmed by finding
    glibc's own `s_log1p-fma.c`, which is `#define __log1p __log1p_fma` plus
    a section-attribute wrapper around `#include
    <sysdeps/ieee754/dbl-64/s_log1p.c>`: the *same source*, recompiled with
    `-mfma`. A literal, unfused transliteration of that source
    (`src/reference/double/log1p.rs`) passed 42M+ random samples with only
    63 mismatches, every one 1 ulp and traceable to a specific compiler
    fusion the disassembly shows and the source's own expression grouping
    does not: the tiny-`|x|` shortcut's `x - x*x*0.5` is one fused
    `vfnmadd231sd`; the `Lp[]` polynomial's flat-looking `R = R1 + z2*R2 +
    z4*R3 + z6*R4` is compiled as a 3-step `fma` chain, algebraically
    Horner in `z2` but not the same roundings; and the `k != 0` tail's
    final `k*ln2_hi - (...)` is one fused `vfmsub231sd`. Fixed all four,
    re-verified clean over 60M further random samples plus a corpus
    specifically constructed to land in the rare `hu == 0, k != 0` branch
    (`x` chosen so `1 + x` normalises within a handful of ulp of a power of
    two, 831K points, 0 mismatches). One more bug, found only by
    `tests/delegating.rs`'s existing corpus (not the brute-force sweeps,
    whose own "want" came from the identical code path and so could not
    catch a difference from it): x86-64's `divsd` for a genuine runtime
    `0.0/0.0` returns `0xfff8000000000000`, sign bit set — its hardware
    default NaN — which is *not* Rust's own `f64::NAN` constant
    (`0x7ff8...`, unset, a compile-time-folded value with a different
    convention). The port's first draft wrote the source's own literal
    `(x - x) / (x - x)` and got this right by construction; replacing it
    with `f64::NAN` to silence clippy's `eq_op` lint silently introduced
    the bug — fixed by using the explicit correct bit pattern instead of
    either form. A second NaN subtlety, also found this way: a NaN
    *argument* (either sign) needed an explicit early check
    (`x.to_bits() | (1 << 51)`, preserving sign and payload, forcing the
    quiet bit) — the algorithm's own domain-error branch, taken literally
    for a negative-signed NaN input, would canonicalise it instead, which
    does not match the live platform's actual behaviour. **Scalar reference
    ported and verified**, replacing the platform delegate; no vector
    kernel yet (`BitExact` measures ~1.05x, essentially unchanged from
    delegating, since the lane-at-a-time fallback costs the same whether it
    calls into the platform or into this crate's own Rust).
  - **`hypot`'s prior finding was reading a different, older algorithm.**
    This host's glibc 2.43 ships the 2021+ Borges "MyHypot3" correction
    (`e_hypot.c`), not the Dekker/`dla.h`-style routine the earlier
    disassembly pass found — a smaller, better-documented shape with named
    constants (`SCALE`, `LARGE_VAL`, `TINY_VAL`, `EPS`). Confirmed no
    separate FMA ifunc, and confirmed by *complete* disassembly this time
    (950 bytes, self-contained, zero calls out — the earlier session's own
    "cut off mid-trace" was an address problem, not a real limit) that this
    host runs the algorithm's own non-FMA branch, zero `vfmadd` anywhere,
    matching the source's stated reason (fusing would break the
    compensated correction's error-free-transformation identities, not
    just round differently). **Scalar reference ported**
    (`src/reference/double/hypot.rs`), literal and unfused throughout,
    verified against the live platform over 25M+ pairs (20M random bit
    patterns, 5M ordinary-magnitude pairs at random exponents, dense
    sweeps around every scale-threshold and power-of-two boundary, and the
    full specials cross-product including signaling-vs-quiet NaN). Two
    real bugs, both hand-computed constants: `SCALE` (`2^-600`) and
    `TINY_VAL` (`2^-459`) were both wrong by large, silent margins from a
    bit-pattern arithmetic slip — not an algorithm misunderstanding —
    surfacing as spurious NaN for small-magnitude pairs that should have
    taken the up-scaled `kernel()` path but instead underflowed
    `ax*ax + ay*ay` to exactly `0.0` on the unscaled common-case path,
    then divided by it. Fixed by computing both constants programmatically
    (`2f64.powi(n).to_bits()`) rather than by hand — the same
    provenance discipline the crate already applies to tables, now applied
    to bare scalar constants too. No vector kernel yet, deliberately: this
    is the most delicate of A2/A3's three ports, and landing a solid,
    verified scalar port rather than rushing a vector kernel on top of it
    the same session is the same trade A4 made for `asin`/`acos`.
  - **`log1p` and `hypot` vector kernels completed.**
    `log1p` replays `__log1p_fma` extracting per-lane reduction parameters
    and evaluating the FMA Horner tail vectorially. `hypot` replays Borges
    "MyHypot3" lane-parallel with unfused arithmetic and `V::select` branch
    blending. Both verified bit-exact across all SIMD widths (1, 2, 4, 8)
    in `tests/bit_exact.rs`.

## 7. Suggested sequencing

Ordered by value density; each phase is independently shippable and ends with
the full verification protocol (§8).

| phase | contents | effort | headline outcome |
|---|---|---|---|
| **0** | D1, D2, B4 | S | measurement tooling in place; bench table honest — **done** |
| **1** | C1, then re-tighten `pow`; C4; C5 if touched | M | `Fast` exponentials ~1 ulp, `pow` ≤4 asserted, inverse trig ≤4 — **done** |
| **2** | B1, B3 (B2 withdrawn — considered decision) | M–L | native f32 `Fast`: `tanhf` 4.97x, `sin`/`cos`f 17-18x, `tan`f 7.4x — **done** |
| **3** | A3, A2, D3 | **L** | `log10` (2.61x), `log1p` (1.13x / 1.31x `+Finite`), `hypot` (1.93x) bit-exact vector kernels and scalar references — **all done, see §6a** |
| **4** | A1 (`sincos` → `sin`/`cos` → `tan`), then A4 | XL | the whole trigonometric family bit-exact *and* faster — **done, see §6a**: `sin`/`cos`/`sincos` 3.0–3.7x, `tan` 2.77x, A4's `atan`/`atan2` 1.28x/2.74-3.03x, `asin`/`acos` 0.83-0.85x — **all done** |
| **5** | A5 gather experiment; ~~C2, C3~~ done | M | **measured negative end-to-end and rolled back** — documented in §3, the hardware gather lost to the per-lane idiom on this machine; `tgamma` 2037→512 ulp — **C2/C3 done** |

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
was written; §6a has the full accounting for each. A5 was next; it has been
done and its negative verdict is recorded in §3 — see §10 for what remains.

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

- **Schedule-reading risk (`tan`, A4):** the FMA placement problem scales with
  branch count; budget for the disassembly step to dominate, and land one
  function at a time. The test suite is designed to fail loudly — trust it.
  Confirmed in practice by A1's `sin`/`cos`/`sincos` port (§6a): four separate
  FMA-placement bugs, each found only by iterating against a live gdb trace,
  not by re-reading the C source more carefully.
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
2. ~~**A5 gather backend**~~ — **decided: go — then measured negative and
   rolled back.** Prototyped, measured (+22-23% on `f64x8` in an in-process
   gather-step A/B), and accepted for `exp`; the wider rollout to every other
   table kernel was then done, A/B'd end-to-end on the reference machine, and
   reverted — the hardware gather lost to the per-lane idiom on this
   hardware. `Simd::gather_bits` and the two overrides remain (exercised by
   `exp`'s accepted use); see §3's A5 entry for the table and the takeaway.
   `asin`/`acos` parity stays open — the 13-slot per-lane gather remains the
   cost, and the hardware gather is not the cure on this machine.
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
