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
  in-place evaluation all gather and scatter every full vector chunk in
  `src/function.rs`. The current 1M-element benchmark can hide overhead that
  matters for short and medium slices.
- `patch_lanes` and `patch_lanes2` in `src/simd/mod.rs` are shared by more than
  sixty kernel call sites. They are cheap when no mask bit is set, but unpack
  the complete vector and scan every lane as soon as one repair is needed.
- `asinh`, `acosh`, and `atanh` still delegate under `BitExact`; their `Fast`
  paths demonstrate that lane-parallel work is available, but exact ports must
  follow the platform schedules rather than reuse formulas blindly.
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
- The repository already has strong foundations: deterministic corpora,
  multi-width bit-exact tests, exhaustive `f32` scans, a 30M-sample ignored ULP
  harness controlled by `RMATH_SCAN_N`, benchmark CSV output, and a 3% diff
  checker.

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

## 5. Workstream A: harden the benchmark first

Priority: P0. Effort: small.

1. Extend `examples/bench.rs` with selectable slice sizes and named corpus
   profiles. Keep the current default output stable.
2. Add absolute nanoseconds per element to CSV alongside speedup. Speedup alone
   can conceal a noisy or changed scalar baseline.
3. Record target metadata in a comment/header or companion record: CPU, active
   lane widths, FMA state, Rust version, and relevant `RUSTFLAGS`.
4. Add focused rows for `eval_slice`, in-place, binary, pair-output, and tail
   handling, plus mixed-special corpora for representative cheap and expensive
   kernels.
5. Teach `tools/bench_diff.py` to reject incomparable metadata/corpora and to
   report both relative and absolute movement.

Deliverable: a reproducible baseline covering the workloads in section 4.

## 6. Workstream B: buffer traversal and tail cost

Priority: P1. Effort: small/medium.

The hypothesis is that `strided` plus generic gather/scatter is appropriate for
large buffers but leaves avoidable overhead for full chunks and short slices.
Test this before changing the API.

1. Profile `Function`, `Function2`, `FunctionPair`, and in-place evaluation by
   size, separating kernel time from traversal time with an identity/no-op test
   function.
2. Prototype a full-chunk loop with a single explicit tail, comparing it with
   the current closure-based `strided` path. Inspect generated assembly for
   bounds-check and copy elimination.
3. Prefer a safe implementation. Consider backend-specific or unsafe loads only
   if assembly and end-to-end measurements show the safe path cannot reach the
   target; any unsafe path needs alignment-independent handling and exhaustive
   length tests.
4. Apply the winning shape consistently to unary, binary, scalar-second,
   pair-output, and in-place methods.

Acceptance: at least 10% improvement for one or more short/medium size bands,
neutral large-buffer throughput, identical output, and no panic-behaviour change
for length mismatches.

## 7. Workstream C: scalar-repair overhead

Priority: P1. Effort: medium.

`patch_lanes` is a shared optimisation point, but the common no-repair path is
already a single `mask.none()` branch. Changes must therefore target sparse and
dense repair cases without taxing the zero-mask case.

1. Benchmark `patch_lanes`/`patch_lanes2` independently at each mask density and
   through representative kernels (`exp`, `erf`, trig, and a binary function).
2. Compare the current boolean-array scan with a backend-neutral mask-bit
   iterator. Add a mask-bit API only if all backends can implement it cheaply;
   do not expose backend representation details to kernels.
3. Add an `all()` path only when calling the scalar reference across all lanes is
   measurably cheaper than computing/packing the mixed result.
4. Investigate buffer-level batching of exceptional inputs only as a separate
   experiment. It must preserve output order and IEEE behaviour and must pay for
   its temporary storage on realistic slice sizes.

Acceptance: zero-mask performance within noise, at least 10% gain for sparse or
dense repairs in end-to-end rows, and bit-identical repaired lanes.

## 8. Workstream D: exact inverse-hyperbolic ports

Priority: P1. Effort: large.

Port one function at a time, starting with whichever live disassembly shows can
reuse the already-ported exact `log1p`, `ln`, and square-root schedules with the
least extra table state. `atanh` is the first candidate; `asinh` and `acosh`
follow only after its process is proven.

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

1. Add an independent high-precision oracle for development measurements; keep
   platform `tgamma` comparison as a compatibility signal, not the sole truth.
2. Derive a direct large-positive algorithm with an explicit error budget for
   range reduction, Stirling correction, polynomial evaluation, scaling, and
   final rounding.
3. Generate all coefficients at high precision and commit provenance plus a
   regeneration check.
4. Preserve the existing recurrence path below `TG_DIRECT_LIMIT`; compare the
   old and new paths around the hand-off and search for a better threshold only
   after both are stable.
5. Add dense tests around 18, binade boundaries, 170..overflow, and the final
   finite result, plus monotonicity checks on the positive domain.

Acceptance: reduce the measured high-range worst-ULP by at least 4x, lower the
asserted/documented bound accordingly, keep the direct-path benchmark neutral,
and limit a high-range throughput loss to 10%. A larger speed trade requires an
explicit API/policy decision rather than silently changing the existing path.

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
| 0 | A: benchmark hardening | Comparable size/corpus/target baselines exist |
| 1 | B and C: shared traversal/repair experiments | Winning changes land; negative results are recorded |
| 2 | D: `atanh`, then `asinh`/`acosh` | Each function independently passes exactness and speed gates |
| 3 | F: direct large `tgamma` | New measured and asserted bound, with speed trade quantified |
| 4 | E: coherent-row `asin` prototype | Go/no-go decision before touching `acos` |
| 5 | G: selective `Fast` tightening | Only measured Pareto improvements land |

Each phase is independently shippable. Do not combine a numerical algorithm
change with a traversal/backend rewrite in one benchmark comparison.

## 13. Completion checklist

For every accepted change:

- [ ] before/after CSVs were produced from alternating runs on the same machine;
- [ ] the targeted row clears its acceptance threshold and unrelated rows stay
      within 3%;
- [ ] `BitExact` boundary, special, random-bit, and all-width tests pass where
      applicable;
- [ ] changed `Fast` kernels pass the required 30M/100M or exhaustive scan;
- [ ] the kernel documentation, `tests/accuracy.rs`, and README quote one
      consistent bound;
- [ ] generated tables/coefficients reproduce byte-for-byte;
- [ ] `cargo test --release` passes with default and no-default features;
- [ ] clippy and rustdoc are warning-free; and
- [ ] negative experiments leave a short record so they are not rediscovered.

## 14. Explicit non-goals for this cycle

- Repeating the broad hardware-gather rollout on the current reference CPU.
- Reintroducing the rejected native `erfcf` approximation without a new
  algorithm or target-specific reason.
- Porting `jn`/`yn` or the large-argument Bessel path without a concrete workload
  demonstrating that their dynamic control flow is worth the complexity.
- Weakening `BitExact`, `FullRange`, or a published `Fast` bound to make a
  benchmark number look better.
