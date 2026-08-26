//! `asin`/`acos`'s band table, repacked for a vector gather.
//!
//! Not a transcription — every value here is copied out of
//! [`super::asincos_data::ASNCS`] by `const fn` at compile time, so there is
//! exactly one source of truth for the numbers and no generator step to keep
//! in sync. What changes is only the *shape*, and the shape is what the
//! vector kernel could not afford:
//!
//! * **One row index instead of six band formulas.** `e_asin.c` computes the
//!   row from `k = high32(|x|)` through six different expressions, one per
//!   band, each with its own mask, shift, multiplier and base. All six
//!   collapse to the single affine expression
//!
//!   ```text
//!   row = floor(256 * |x|) - 32
//!   ```
//!
//!   for every `|x|` in `[0.125, 0.96875)` — the table is really indexed by
//!   the top eight bits of the argument, and the six formulas are six ways of
//!   spelling that after the leading `1.` has been stripped. `tests/
//!   bit_exact.rs` proves the equality over every `k` in the band rather than
//!   asserting it here.
//!
//! * **One row stride instead of five.** Upstream rows are `4 + degree` wide
//!   with `degree` running 5..9, so a row's slots sit at an offset that
//!   depends on which band it came from. Here every row is [`STRIDE`] wide
//!   with the coefficient slots a lane's own degree does not use written as
//!   `+0.0`. A zero leading coefficient is a no-op in Horner's rule —
//!   `fma(xx, 0, c) == c` exactly — so one unrolled degree-9 fold evaluates
//!   every band, with no per-lane branch and, more importantly, no
//!   *conditional* store in the gather loop: LLVM can only drop the
//!   zero-initialisation of the gather's staging arrays when every slot is
//!   written unconditionally.
//!
//! * **No `x0` slot.** Upstream's slot 0 is the band centre, and it is always
//!   exactly `(row + 32.5) / 256` — again a consequence of the table being
//!   indexed by the top eight bits. The kernel computes it, exactly, from the
//!   `floor` it already has; it is not stored or gathered. `tests/
//!   bit_exact.rs` checks the identity for all 216 rows.
//!
//! * **Padding past the last real row.** Lanes outside the table band (tiny,
//!   near one, infinite, NaN) run the gather anyway under this crate's
//!   whole-vector-blend discipline, and their rows are discarded by the blend
//!   — but they still have to be in bounds. Sizing the table to one row past
//!   `0xfff` is what lets the kernel's row offset be a mask rather than a
//!   clamp, and lets that offset be computed in the same float arithmetic that
//!   already produced `floor(256*|x|)`: see the kernel's `asncs_key`.

use super::asincos_data::ASNCS;

/// Slots per packed row: `t1`, nine coefficient slots, `outer`, `final` — and
/// four slots of nothing, to round the row up to 128 bytes.
///
/// The padding is not waste, it is alignment. A block-loading
/// [`Simd::gather_run`](crate::simd::Simd::gather_run) reads each row as two
/// 64-byte vectors; at a 96-byte stride half the rows start mid-cache-line and
/// *every* one of those loads is split across two lines. At 128 bytes, with
/// the table itself 64-byte aligned, no load ever splits.
pub const STRIDE: usize = 16;

/// Slots a caller actually reads from a row: `t1`, nine coefficient slots,
/// `outer`, `final`. The rest of [`STRIDE`] is alignment padding.
pub const USED: usize = 12;

/// Slots in the table, counting the padding past the last real row.
///
/// The kernel's row offset is `(floor(256*|x|) - 32) * STRIDE` computed in
/// *float* arithmetic and read out of a mantissa, so what has to be a power of
/// two is the offset's mask, not the row count: any lane outside the table
/// band produces an arbitrary 12-bit offset, and everything up to `0xfff` plus
/// one row has to be readable — rounded up to sixteen, because the
/// block-loading `Simd::gather_run` override reads a whole eight-slot block
/// whether or not every slot in it is wanted. Only the first `REAL_ROWS * STRIDE` slots are
/// ever the answer to anything; the rest exist so the mask needs no clamp.
pub const SLOTS: usize = 4096 + 16;

/// Real rows — one per value of `floor(256 * |x|) - 32` for `|x|` in
/// `[0.125, 0.96875)`.
pub const REAL_ROWS: usize = 216;

/// `(n, degree)` in [`ASNCS`] for packed row `r`, the six band formulas of
/// `e_asin.c`'s `asncs_band_index` re-expressed against the unified row index.
///
/// Only the *inverse* mapping is needed here — the kernel never evaluates
/// this — so it is written as the plain table-of-bands it is.
pub const fn source_row(r: usize) -> (usize, usize) {
    if r < 32 {
        (11 * r, 5)
    } else if r < 96 {
        (11 * (r - 32) + 352, 5)
    } else if r < 160 {
        (1056 + (r - 96) * 12, 6)
    } else if r < 204 {
        (992 + (r - 96) * 13, 7)
    } else if r < 212 {
        (884 + (r - 96) * 14, 8)
    } else {
        (768 + (r - 96) * 15, 9)
    }
}

/// The packed table itself: `ROWS * STRIDE` bit patterns, row-major.
///
/// Row `r` is `[t1, c0, c1, .., c8, outer, final]`, where `c0..c<degree>` are
/// upstream's own coefficients in upstream's own order and `c<degree>..c8` are
/// `+0.0`.
pub static PACKED: Aligned = Aligned(build());

/// [`PACKED`]'s storage, aligned so that every row begins on a cache line.
#[repr(C, align(64))]
pub struct Aligned(pub [u64; SLOTS]);

impl core::ops::Deref for Aligned {
    type Target = [u64; SLOTS];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Build [`PACKED`] from [`ASNCS`], at compile time.
const fn build() -> [u64; SLOTS] {
    let mut out = [0u64; SLOTS];
    let mut r = 0;
    while r < REAL_ROWS {
        let (n, degree) = source_row(r);
        let base = r * STRIDE;
        // Slot 0 of the upstream row is `x0`, which the kernel computes.
        out[base] = ASNCS[n + 1];
        let mut j = 0;
        while j < 9 {
            out[base + 1 + j] = if j < degree { ASNCS[n + 2 + j] } else { 0 };
            j += 1;
        }
        out[base + 10] = ASNCS[n + 2 + degree];
        out[base + 11] = ASNCS[n + 3 + degree];
        r += 1;
    }
    out
}
