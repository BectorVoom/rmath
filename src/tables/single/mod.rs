//! Single-precision tables.
//!
//! Every constant in here is an `f64`. That is the algorithm, not an
//! oversight: the platform's `expf`, `logf`, `log2f` and `powf` all do their
//! arithmetic in double precision over a small table and round once at the
//! end. Reproducing that is what makes [`crate::kernels::single`] bit-exact,
//! and widening to `f64xN` first is what makes it vectorised rather than a
//! scalar loop.

pub mod exp;
pub mod log;
pub mod log2;
pub mod poly;
