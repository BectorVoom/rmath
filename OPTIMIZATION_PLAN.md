# Speed and precision optimisation plan

Status: proposed  
Prepared: 2026-08-20  
Scope: the next optimisation cycle, after the completed work recorded in
[`ROADMAP.md`](ROADMAP.md)

## 1. Goal

Improve useful throughput without weakening the crate's numerical contracts.
Optimisation is successful only when it moves a measured speed/precision Pareto
frontier; a faster result with an undocumented error increase, or a more accurate
result with an unmeasured throughput loss, is not complete.

The primary outcomes are:

1. make the remaining default `BitExact` paths faster;
2. reduce the largest remaining documented error, `tgamma` at large positive
   arguments;
3. reduce buffer-processing and exceptional-lane overhead shared by many
   kernels; and
4. make regressions visible across input distributions, slice sizes, vector
   widths, and target features.

## 2. Contracts that must not change

- `BitExact` must continue to match the platform result bit-for-bit, including
  signed zero, infinities, and NaN behaviour. Ports must replay the actual
  operation schedule, including FMA placement; source-level algebraic
  equivalence is insufficient.
- `Fast` error bounds must remain asserted in `tests/accuracy.rs` and documented
  beside the kernel. Bounds may only be tightened after a larger measurement
  campaign; they may not be loosened as a hidden speed trade.
- `FullRange` must remain correct for all inputs. `Finite` may skip checks only
  within its documented per-function domain.
- New coefficients and tables must be generated reproducibly by the tools under
  `tools/`, using high-precision arithmetic or pinned upstream source. Do not
  hand-transcribe numeric data.
- Scalar repair code remains the authority for rare or unsupported lanes. A
  vector main path may change how lanes reach that repair, but not its results.
- An optimisation must be evaluated end-to-end. Isolated instruction or gather
  microbenchmarks are supporting evidence, not acceptance evidence.

## 3. Current evidence and remaining gaps

CodeGraph shows the following high-leverage surfaces:

- `Function::eval_slice`, `Function2::eval_slice`, the pair variants, and
  in-place evaluation already use `chunks_exact` plus one scalar tail, but each
  full chunk is still copied into an associated lane array, converted with
  `from_array`, converted back with `to_array`, and copied to the destination.
  The remaining traversal question is whether those copies disappear after
  inlining on every backend and slice shape.
- `patch_lanes` and `patch_lanes2` in `src/simd/mod.rs` are shared by more than
  sixty kernel call sites. They already have zero-mask and all-mask branches
  and iterate set bits with `to_bitmask`; remaining cost comes from computing a
  vector result that may be discarded, unpacking mixed vectors, and duplicate
  scalar work for pair-returning functions.
- `atanh` is now a genuine vector `BitExact` port and is the process template
  for the two inverse-hyperbolic gaps that remain: `asinh` and `acosh`.
  Their `Fast` paths demonstrate that lane-parallel work is available, but
  exact ports must follow the platform schedules rather than reuse formulas
  blindly.
- `asin` and `acos` remain below scalar parity under `BitExact` because their
  13-value per-lane table rows dominate execution.
- Large positive `tgamma` uses `stirling_dd` followed by `exp`. This improved the
  earlier result substantially, but the remaining error cannot be removed by
  refining `lgamma` alone; it needs a direct large-argument Gamma algorithm.
- Hardware gather was already prototyped and rolled out broadly on the reference
  Zen 5 machine. It regressed the affected kernels and was rolled back. Native
  `erfcf` `Fast` was also tried and rejected on its measured speed/error trade.
  Neither experiment should be repeated without a materially new hypothesis or
  a different target architecture.
- The repository already has strong foundations: deterministic named corpora,
  configurable benchmark sizes, dedicated traversal and repair suites,
  absolute ns/element CSV output with comparison checks, multi-width bit-exact
  tests, exhaustive `f32` scans, and a 30M-sample ignored ULP harness controlled
  by `RMATH_SCAN_N`.

## 4. Measurement gates

Every work item begins and ends with the same protocol.

### Performance

Record results for:

- scalar, `f64x2`, `f64x4`, `f64x8`, `f32x4`, and `f32x8` where supported;
- `BitExact`/`FullRange`, `BitExact`/`Finite`, `Fast`/`FullRange`, and
  `Fast`/`Finite` where meaningful;
- lengths `0`, `1`, `LANES - 1`, `LANES`, `LANES + 1`, `64`, `4096`, and
  `1 << 20`;
- in-domain, boundary-heavy, random-bit, sorted/coherent, and mixed-special
  corpora; and
- special-lane densities of 0%, one lane per vector, 25%, and 100% for kernels
  that use scalar repair.

Use native FMA results as the primary reference-machine numbers, then perform
correctness and smoke-performance runs for AVX2 and scalar/no-default-feature
builds. Compare alternating before/after runs on an otherwise idle machine.

Acceptance gates:

- at least 10% end-to-end improvement on the targeted workload, or at least 5%
  when the change affects a shared primitive and improves several kernels;
- no unrelated benchmark row regresses by more than 3%; and
- no target-specific implementation lands without a neutral-or-better result
  on that target.

The existing large-buffer command remains the common baseline:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=before.csv
# apply the candidate change
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --csv=after.csv
python3 tools/bench_diff.py before.csv after.csv --threshold 0.03
```

### Precision

For each touched function:

- test branch thresholds and the two adjacent representable values on each side;
- include zeroes, subnormals, normal extremes, infinities, NaNs, and random bit
  patterns;
- use log-uniform samples for wide-magnitude domains and focused samples near
  overflow, underflow, roots, poles, and cancellation points;
- check every supported SIMD width so blending and per-lane state cannot hide a
  width-specific error;
- exhaust all `2^32` inputs for a changed `f32` kernel; and
- run at least 30M deterministic samples for a changed `f64` approximation,
  increasing to 100M when setting a new public bound.

The ignored measurement harness is invoked with, for example:

```sh
RMATH_SCAN_N=30000000 cargo test --release --test ulp_scan -- --ignored --nocapture
```

An asserted `Fast` bound should be the measured worst case rounded up with
explicit headroom. Record the worst input and corpus, not only the final bound.

## 5. Workstream A: finish the benchmark gate

Priority: P0. Effort: small.

Selectable sizes and corpora, absolute ns/element, target/lane/FMA metadata,
traversal and repair suites, and metadata-aware diffing already exist. Complete
the gate rather than rebuilding it:

1. Add the selected suite, `rustc -Vv`, CPU model, repetition policy, and all
   effective code-generation flags to the CSV metadata. Make missing critical
   metadata an error, not silently comparable.
2. Add a small matrix runner for the canonical sizes and corpora in section 4,
   producing one manifest that links every CSV to a commit, target, and command.
3. Replace best-of-eight as the only statistic with warm-up plus enough samples
   to report median and dispersion. Keep the minimum as diagnostic data if it
   remains useful.
4. Complete the traversal suite with length `0`, `LANES + 1`, a large-buffer
   row, and an identity kernel that isolates traversal from math. Ensure unary,
   in-place, binary, scalar-second, and both pair-output shapes are represented.
5. Make `tools/bench_diff.py` gate direct ns/element movement as well as speedup
   movement. A simultaneous change in the scalar baseline must not hide a crate
   regression.

Deliverable: a reproducible baseline covering the workloads in section 4.

## 6. Workstream B: buffer traversal and tail cost

Priority: P1. Effort: small/medium.

The current implementation already has a full-chunk loop and one explicit
scalar tail. The hypothesis to test is narrower: associated-array
copy/convert/copy traffic may survive optimisation for cheap kernels, especially
on short and medium buffers.

1. Use the identity row to quantify the traversal floor for each API shape and
   size; compare it with representative cheap (`sqrt`/`floor`) and expensive
   (`exp`/trig) kernels.
2. Inspect release assembly for each supported width and confirm whether the
   `copy_from_slice -> from_array` and `to_array -> copy_from_slice` sequences
   collapse into unaligned vector loads/stores with no repeated bounds checks.
3. If they do not, prototype an internal backend load/store abstraction. Keep
   it alignment-independent and safe at the call site; isolate and document any
   unavoidable `unsafe` in the backend with exact length preconditions.
4. Do not vectorise the tail by evaluating padded extra lanes unless the public
   evaluation-count semantics are explicitly accepted. Preserve the current
   scalar tail and panic behaviour by default.
5. Apply a winning representation consistently to unary, binary,
   scalar-second, pair-output, two-argument pair-output, and in-place methods.

Acceptance: at least 10% improvement for one or more short/medium size bands,
neutral large-buffer throughput, identical output, and no panic-behaviour change
for length mismatches.

## 7. Workstream C: scalar-repair overhead

Priority: P1. Effort: medium.

`patch_lanes` already uses `mask.none()`, `mask.all()`, and a set-bit iterator.
Changes must therefore target work performed before repair and mixed-vector
pack/unpack cost without taxing the zero-mask case.

1. Extend the existing repair suite to report 0%, one-lane, 25%, 50%, and 100%
   masks through representative unary, binary, and pair-returning kernels.
2. In `dispatch`/`dispatch2` and selected exact kernels, prototype computing the
   special mask before the vector main path so an all-special vector can skip a
   result that will be discarded. Confirm neutral instruction scheduling for
   the common zero-mask case.
3. Prototype a pair-aware repair helper so `sincos`-style functions unpack once
   and invoke the scalar pair reference once per repaired lane, instead of
   repairing each output independently.
4. For mixed masks, compare the existing array round-trip with backend-specific
   lane extraction/insertion only where the backend exposes it cheaply. Keep the
   current bitmask API as the neutral implementation.
5. Investigate buffer-level batching of exceptional inputs only as a separate
   experiment. It must preserve output order and IEEE behaviour and must pay for
   temporary storage on realistic slice sizes.

Acceptance: zero-mask performance within noise, at least 10% gain for sparse or
dense repairs in end-to-end rows, and bit-identical repaired lanes.

## 8. Workstream D: exact inverse-hyperbolic ports

Priority: P1. Effort: large.

`atanh` has completed this process and now measures above scalar parity. Port
`asinh` and `acosh` one at a time, starting with whichever live disassembly
shows can reuse the already-ported exact `log1p`, `ln`, and square-root
schedules with the least extra table state.

For each function:

1. identify the actual platform symbol/ifunc selected on the reference machine;
2. trace its branch structure, constants, association, and FMA placement from
   disassembly;
3. write and independently brute-force a scalar reference port;
4. vectorise whole bands, repairing only genuinely rare or unsupported lanes;
5. add boundary-neighbour, random-bit, special-value, and multi-width corpora to
   `tests/bit_exact.rs`; and
6. retain delegation if the vector port does not beat it end-to-end.

Acceptance per function: zero bit mismatches across the full corpus at every
width, at least 1.25x over the scalar baseline under default policy, and no
change to the existing `Fast` bound.

## 9. Workstream E: `asin`/`acos` exact table cost

Priority: P2. Effort: medium, explicitly experimental.

Do not retry the rejected AVX2/AVX-512 gather rollout. Test data-layout and input
coherence hypotheses instead:

1. prototype a uniform-band fast path: when all lanes select one table row,
   splat that row once and bypass per-lane gathers;
2. compare array-of-structures and generated structure-of-arrays layouts for the
   existing scalar gather, accounting for instruction-cache and table-size cost;
3. benchmark random, sorted, and locally coherent inputs so a win on one corpus
   is not presented as universal; and
4. prototype on `asin` only, then share with `acos` only after an end-to-end win.

Acceptance: preserve exact bits, improve the intended coherent workload by at
least 15% or the standard random workload by at least 10%, and keep the other
workload within 3%. Otherwise document the result and retain the current path.

## 10. Workstream F: large-argument `tgamma` precision

Priority: P1 for precision. Effort: large.

The current `stirling_dd -> exp` composition has reached its architectural
accuracy limit. Replace it only with a direct, scaled Gamma computation that
controls mantissa and exponent separately and avoids rounding through
`ln(Gamma(x))`.

1. Establish and commit the current worst input and ULP result over the direct
   and high ranges, then reconcile the asserted test bound, kernel docs,
   `README.md`, and `ROADMAP.md`. At present those surfaces do not all describe
   the large-positive error with the same specificity.
2. Add an independent MPFR-class high-precision oracle for development
   measurements; keep platform `tgamma` comparison as a compatibility signal,
   not the sole truth. The oracle must not become a runtime dependency.
3. Derive a direct large-positive algorithm with an explicit error budget for
   range reduction, Stirling correction, polynomial evaluation, scaling, and
   final rounding.
4. Generate all coefficients at high precision and commit provenance plus a
   regeneration check.
5. Preserve the existing recurrence path below `TG_DIRECT_LIMIT`; compare the
   old and new paths around the hand-off and search for a better threshold only
   after both are stable.
6. Add dense tests around 18, binade boundaries, 170..overflow, and the final
   finite result, plus monotonicity checks on the positive domain.

Acceptance: target at most 64 ulp in the high positive range and at minimum a
4x reduction from the freshly recorded baseline; lower the asserted/documented
bound accordingly, keep the direct-path benchmark neutral, and limit a
high-range throughput loss to 10%. A larger speed trade requires an explicit
API/policy decision rather than silently changing the existing path.

## 11. Workstream G: targeted `Fast` precision tightening

Priority: P3, after the shared and Gamma work.

Use measured error attribution rather than increasing polynomial degree by
default. Candidate order:

1. `cbrt` near extreme magnitudes, where the measured worst case still has
   headroom below its asserted bound;
2. `pow` near `|y * log2(x)| ~= 1020`, where error amplification is largest; and
3. any function whose new 100M-sample scan lands close to its asserted bound.

For each candidate, isolate range-reduction error, coefficient approximation,
evaluation rounding, reconstruction, and final rounding. Try compensated
operations or a local hard-case repair before a global degree increase.

Acceptance: a strictly lower asserted bound with no more than 3% throughput
loss, or a documented Pareto option that requires an explicit policy decision.

## 12. Sequencing

| Phase | Work | Exit condition |
|---|---|---|
| 0 | A: finish benchmark gate and record Gamma baseline | Comparable size/corpus/suite/target baselines exist |
| 1 | B and C: shared traversal/repair experiments | Winning changes land; negative results are recorded |
| 2 | D: `asinh`, then `acosh` (order may swap after disassembly) | Each function independently passes exactness and speed gates |
| 3 | F: direct large `tgamma` | New measured and asserted bound, with speed trade quantified |
| 4 | E: coherent-row `asin` prototype | Go/no-go decision before touching `acos` |
| 5 | G: selective `Fast` tightening | Only measured Pareto improvements land |

Each phase is independently shippable. Do not combine a numerical algorithm
change with a traversal/backend rewrite in one benchmark comparison.

## 13. Completion checklist

For every accepted change:

- [x] before/after CSVs were produced from alternating runs on the same machine;
- [x] the targeted row clears its acceptance threshold and unrelated rows stay
      within 3%;
- [x] `BitExact` boundary, special, random-bit, and all-width tests pass where
      applicable;
- [x] changed `Fast` kernels pass the required 30M/100M or exhaustive scan;
- [x] the kernel documentation, `tests/accuracy.rs`, and README quote one
      consistent bound;
- [x] generated tables/coefficients reproduce byte-for-byte;
- [x] `cargo test --release` passes with default and no-default features;
- [x] clippy and rustdoc are warning-free; and
- [x] negative experiments leave a short record so they are not rediscovered.

## 14. Explicit non-goals for this cycle

- Repeating the broad hardware-gather rollout on the current reference CPU.
- Reintroducing the rejected native `erfcf` approximation without a new
  algorithm or target-specific reason.
- Reworking `atanh` as though it still delegated; it is now the reference
  process for the remaining inverse-hyperbolic ports.
- Porting `jn`/`yn` or the large-argument Bessel path without a concrete workload
  demonstrating that their dynamic control flow is worth the complexity.
- Weakening `BitExact`, `FullRange`, or a published `Fast` bound to make a
  benchmark number look better.
