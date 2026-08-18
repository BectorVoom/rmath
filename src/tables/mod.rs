//! Generated data tables, one subtree per precision.
//!
//! Produced by `tools/gen_tables.py` from ARM optimized-routines' C sources,
//! rather than transcribed. The generator evaluates the same `#if` conditions
//! the C build would, so the constants are provably the ones glibc compiles
//! in — which is what lets [`crate::policy::BitExact`] mean *bit*-exact.
//!
//! Regenerate with:
//!
//! ```text
//! python3 tools/gen_tables.py
//! ```

pub mod double;
pub mod single;
