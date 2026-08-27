//! `log1pf` and `log10f`.
//!
//! Both CORE-MATH, both **correctly rounded**, and neither built with
//! `-mfma`: glibc's `sysdeps/x86_64/fpu/multiarch/Makefile` has FMA variants
//! for `logf` and `log2f` but not for these two, and neither `__log1pf` nor
//! `__log10f` in `libm.so.6` contains a `vfmadd`. The arithmetic below is
//! unfused except at the one place `log1p` calls `fmaf` outright.
//!
//! Both take CORE-MATH's two-step shape, and both spend their accurate path
//! on the same problem: `log(1 + z)` and `log10(x)` each have a zero inside
//! the argument range, and near it the leading term cancels against the
//! table entry. The fast path detects that from the result's own low bits —
//! `r`'s mantissa ending in enough zeros means the sum lost them — and the
//! accurate path either re-evaluates at higher order or, for the handful of
//! arguments where even that lands on a halfway point, reads the answer out
//! of a short table.
//!
//! Verified by exhaustive sweep: all 2^32 `f32` bit patterns.

/// `__math_divzerof (1)`: `log` of zero, i.e. `-inf`.
const NEG_INF: f32 = f32::NEG_INFINITY;

/// `__math_invalidf (x)`: the NaN a negative argument produces.
///
/// The [`black_box`](core::hint::black_box) stops Rust folding `0/0` at
/// compile time, which would give the opposite sign from the `divss` the C
/// compiles to. See [`super::invtrig`] for the same note.
#[inline(always)]
#[allow(clippy::eq_op)]
fn invalid(x: f32) -> f32 {
    let x = core::hint::black_box(x);
    (x - x) / (x - x)
}

/// `LOG1P_X0[j] ~= 1/c_j` for the 32 subintervals `log1p`'s table covers.
static LOG1P_X0: [f64; 32] = [
    f64::from_bits(0x3fef81f820000000),
    f64::from_bits(0x3fee9131ac000000),
    f64::from_bits(0x3fedae6077000000),
    f64::from_bits(0x3fecd85689000000),
    f64::from_bits(0x3fec0e0704000000),
    f64::from_bits(0x3feb4e81b5000000),
    f64::from_bits(0x3fea98ef60000000),
    f64::from_bits(0x3fe9ec8e95000000),
    f64::from_bits(0x3fe948b0fd000000),
    f64::from_bits(0x3fe8acb90f000000),
    f64::from_bits(0x3fe8181818000000),
    f64::from_bits(0x3fe78a4c81000000),
    f64::from_bits(0x3fe702e05c000000),
    f64::from_bits(0x3fe6816817000000),
    f64::from_bits(0x3fe6058160000000),
    f64::from_bits(0x3fe58ed231000000),
    f64::from_bits(0x3fe51d07eb000000),
    f64::from_bits(0x3fe4afd6a0000000),
    f64::from_bits(0x3fe446f865000000),
    f64::from_bits(0x3fe3e22cbd000000),
    f64::from_bits(0x3fe3813814000000),
    f64::from_bits(0x3fe323e34a000000),
    f64::from_bits(0x3fe2c9fb4e000000),
    f64::from_bits(0x3fe27350b9000000),
    f64::from_bits(0x3fe21fb781000000),
    f64::from_bits(0x3fe1cf06ae000000),
    f64::from_bits(0x3fe1811812000000),
    f64::from_bits(0x3fe135c811000000),
    f64::from_bits(0x3fe0ecf56c000000),
    f64::from_bits(0x3fe0a6810a000000),
    f64::from_bits(0x3fe0624dd3000000),
    f64::from_bits(0x3fe0204081000000),
];

/// `log(c_j)`, the fast path's table term.
static LOG1P_LIXB: [f64; 32] = [
    f64::from_bits(0x3f8fc0a8909b4218),
    f64::from_bits(0x3fa77458f51aac89),
    f64::from_bits(0x3fb341d793afb997),
    f64::from_bits(0x3fba926d3a5ebd2a),
    f64::from_bits(0x3fc0d77e7a8a823d),
    f64::from_bits(0x3fc44d2b6c557102),
    f64::from_bits(0x3fc7ab89040accec),
    f64::from_bits(0x3fcaf3c94ecab3d6),
    f64::from_bits(0x3fce27076d54e6c9),
    f64::from_bits(0x3fd0a324e3888ad5),
    f64::from_bits(0x3fd22941fc0c7357),
    f64::from_bits(0x3fd3a64c56ae3fdb),
    f64::from_bits(0x3fd51aad874af21f),
    f64::from_bits(0x3fd686c81d300ea0),
    f64::from_bits(0x3fd7eaf83c7fa9b5),
    f64::from_bits(0x3fd947941aa610ec),
    f64::from_bits(0x3fda9cec9a3f023b),
    f64::from_bits(0x3fdbeb4d9ea4156e),
    f64::from_bits(0x3fdd32fe7f35e5c7),
    f64::from_bits(0x3fde7442617b817a),
    f64::from_bits(0x3fdfaf588dd5ed10),
    f64::from_bits(0x3fe0723e5c635c39),
    f64::from_bits(0x3fe109f39d53c990),
    f64::from_bits(0x3fe19ee6b38a4668),
    f64::from_bits(0x3fe23130d7f93c3b),
    f64::from_bits(0x3fe2c0e9ec9b0b85),
    f64::from_bits(0x3fe34e289cb35ecc),
    f64::from_bits(0x3fe3d9026ad3d3f3),
    f64::from_bits(0x3fe4618bc1eadbbb),
    f64::from_bits(0x3fe4e7d8127dd8a9),
    f64::from_bits(0x3fe56bf9d5967092),
    f64::from_bits(0x3fe5ee02a926936e),
];

/// The same logarithms to more bits, for the accurate path.
static LOG1P_LIX: [f64; 32] = [
    f64::from_bits(0x3f8fc0a890fc03e4),
    f64::from_bits(0x3fa77458f532dcfc),
    f64::from_bits(0x3fb341d793bbd1d1),
    f64::from_bits(0x3fba926d3a6ad563),
    f64::from_bits(0x3fc0d77e7a908e59),
    f64::from_bits(0x3fc44d2b6c5b7d1e),
    f64::from_bits(0x3fc7ab890410d909),
    f64::from_bits(0x3fcaf3c94ed0bff3),
    f64::from_bits(0x3fce27076d5af2e6),
    f64::from_bits(0x3fd0a324e38b90e3),
    f64::from_bits(0x3fd22941fc0f7966),
    f64::from_bits(0x3fd3a64c56b145ea),
    f64::from_bits(0x3fd51aad874df82d),
    f64::from_bits(0x3fd686c81d3314af),
    f64::from_bits(0x3fd7eaf83c82afc3),
    f64::from_bits(0x3fd947941aa916fb),
    f64::from_bits(0x3fda9cec9a42084a),
    f64::from_bits(0x3fdbeb4d9ea71b7c),
    f64::from_bits(0x3fdd32fe7f38ebd5),
    f64::from_bits(0x3fde7442617e8788),
    f64::from_bits(0x3fdfaf588dd8f31f),
    f64::from_bits(0x3fe0723e5c64df40),
    f64::from_bits(0x3fe109f39d554c97),
    f64::from_bits(0x3fe19ee6b38bc96f),
    f64::from_bits(0x3fe23130d7fabf43),
    f64::from_bits(0x3fe2c0e9ec9c8e8c),
    f64::from_bits(0x3fe34e289cb4e1d3),
    f64::from_bits(0x3fe3d9026ad556fb),
    f64::from_bits(0x3fe4618bc1ec5ec2),
    f64::from_bits(0x3fe4e7d8127f5bb1),
    f64::from_bits(0x3fe56bf9d597f399),
    f64::from_bits(0x3fe5ee02a9281675),
];

/// The accurate path's degree-8 `log(1 + z)` series.
static LOG1P_B: [f64; 8] = [
    f64::from_bits(0x3ff0000000000000), // 0x1p+0
    f64::from_bits(0xbfe0000000000000), // -0x1p-1
    f64::from_bits(0x3fd5555555556f6b), // 0x1.5555555556f6bp-2
    f64::from_bits(0xbfd00000000029b9), // -0x1.00000000029b9p-2
    f64::from_bits(0x3fc9999988d176e4), // 0x1.9999988d176e4p-3
    f64::from_bits(0xbfc55555418889a7), // -0x1.55555418889a7p-3
    f64::from_bits(0x3fc24adeca50e2bc), // 0x1.24adeca50e2bcp-3
    f64::from_bits(0xbfc001ba33bf57cf), // -0x1.001ba33bf57cfp-3
];

/// The fast path's degree-4 correction. `LOG1P_C[0]` is a residual, not a
/// constant term: the fit is anchored so that the table entry carries it.
static LOG1P_C: [f64; 5] = [
    f64::from_bits(0xbd43902c33434e7f), // -0x1.3902c33434e7fp-43
    f64::from_bits(0x3feffffffe1cbed5), // 0x1.ffffffe1cbed5p-1
    f64::from_bits(0xbfdffffff7d1b014), // -0x1.ffffff7d1b014p-2
    f64::from_bits(0x3fd5564e0ed3613a), // 0x1.5564e0ed3613ap-2
    f64::from_bits(0xbfd0012232a00d4a), // -0x1.0012232a00d4ap-2
];

/// `ln(2)`, single-rounded — the fast path's per-exponent scale.
const LOG1P_LN2: f64 = f64::from_bits(0x3fe62e42fefa39ef);
/// The fast path's error bound, added rather than subtracted because the
/// polynomial's error has a known sign here.
const LOG1P_EPS: f64 = f64::from_bits(0x3db83073119f21d8);
/// High part of `ln(2)` for the accurate path's exponent term.
const LOG1P_LN2H: f64 = f64::from_bits(0x3fe62e4000000000);
/// Low part.
const LOG1P_LN2L: f64 = f64::from_bits(0x3eb7f7d1cf79abca);

/// `ln(1 + x)`, correctly rounded.
pub fn log1p(x: f32) -> f32 {
    let mut z = f64::from(x);
    let ux = x.to_bits();
    let ax = ux & (!0u32 >> 1);

    if ax < 0x3c88_0000 {
        // |x| < 0x1.1p-6, where `ln(1+x)` is `x` plus a small correction and
        // the table path's cancellation would dominate.
        if ax < 0x3300_0000 {
            if ax == 0 {
                return x;
            }
            return x.mul_add(-x, x);
        }
        let z2 = z * z;
        let z4 = z2 * z2;
        let f = z2
            * ((LOG1P_B[1] + z * LOG1P_B[2])
                + z2 * (LOG1P_B[3] + z * LOG1P_B[4])
                + z4 * ((LOG1P_B[5] + z * LOG1P_B[6]) + z2 * LOG1P_B[7]));
        let mut r = z + f;
        // A result whose low 28 mantissa bits are all zero is sitting on an
        // `f32` rounding boundary; `f + (z - r)` is the part the sum lost,
        // and scaling it up decides the direction.
        if r.to_bits() & 0xfff_ffff == 0 {
            r += f64::from_bits(0x40d0000000000000) * (f + (z - r));
        }
        return r as f32;
    }

    if ux >= 0xbf80_0000 || ax >= 0x7f80_0000 {
        // `x <= -1`, an infinity, or a NaN.
        if ux == 0xbf80_0000 {
            return NEG_INF; // log1p(-1)
        }
        if ux == 0x7f80_0000 {
            return x; // +inf
        }
        if ux << 1 > 0xff00_0000 {
            return x + x; // NaN
        }
        return invalid(0.0);
    }

    let tp = (z + 1.0).to_bits();
    let e = (tp >> 52) as i64 - 0x3ff;
    let m52 = tp & (!0u64 >> 12);
    let j = ((tp >> (52 - 5)) & 31) as usize;
    let xd = f64::from_bits(m52 | (0x3ffu64 << 52));
    z = xd * LOG1P_X0[j] - 1.0;

    let z2 = z * z;
    let ef = e as f64;
    let r = (LOG1P_LN2 * ef + LOG1P_LIXB[j])
        + z * ((LOG1P_C[1] + z * LOG1P_C[2]) + z2 * (LOG1P_C[3] + z * LOG1P_C[4]));
    let ub = r as f32;
    let lb = (r + LOG1P_EPS) as f32;
    if ub == lb {
        return ub;
    }

    let z4 = z2 * z2;
    let f = z
        * ((LOG1P_B[0] + z * LOG1P_B[1])
            + z2 * (LOG1P_B[2] + z * LOG1P_B[3])
            + z4 * ((LOG1P_B[4] + z * LOG1P_B[5]) + z2 * (LOG1P_B[6] + z * LOG1P_B[7])));
    let lh = LOG1P_LN2H * ef;
    let ll = LOG1P_LN2L * ef;
    let rl = f + ll + LOG1P_LIX[j];
    let mut tr = rl + lh;
    if tr.to_bits() & 0xfff_ffff == 0 {
        // Three arguments where the accurate path lands exactly on a
        // halfway point; CORE-MATH tabulates the answer.
        if x == f32::from_bits(0xbc923d58) {
            return f32::from_bits(0xbc938f87) - f32::from_bits(0x30000000);
        }
        if x == f32::from_bits(0xbd1d20af) {
            return f32::from_bits(0xbd203889) + f32::from_bits(0x30800000);
        }
        if x == f32::from_bits(0x3efd81ad) {
            return f32::from_bits(0x3ecdeee1) + f32::from_bits(0x32000000);
        }
        tr += 64.0 * (rl + (lh - tr));
    } else if rl + (lh - tr) == 0.0 {
        // Three more where the correction vanishes exactly, which leaves the
        // sum ambiguous in the other direction.
        if x == f32::from_bits(0x3ddbfec3) {
            return f32::from_bits(0x3dd0f671) + f32::from_bits(0x31000000);
        }
        if x == f32::from_bits(0xbd1d20af) {
            return f32::from_bits(0xbd203889) + f32::from_bits(0x30800000);
        }
        if x == f32::from_bits(0x3ca1e3f1) {
            return f32::from_bits(0x3ca04fc0) + f32::from_bits(0x30000000);
        }
    }
    tr as f32
}

/// `LOG10_TR[j] ~= 1/c_j` for the 65 subintervals, `[0, 64]` inclusive so
/// that the rounded index can reach one past the top.
static LOG10_TR: [f64; 65] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fef81f820000000),
    f64::from_bits(0x3fef07c1f0000000),
    f64::from_bits(0x3fee9131ac000000),
    f64::from_bits(0x3fee1e1e1e000000),
    f64::from_bits(0x3fedae6077000000),
    f64::from_bits(0x3fed41d41d000000),
    f64::from_bits(0x3fecd85689000000),
    f64::from_bits(0x3fec71c71c000000),
    f64::from_bits(0x3fec0e0704000000),
    f64::from_bits(0x3febacf915000000),
    f64::from_bits(0x3feb4e81b5000000),
    f64::from_bits(0x3feaf286bd000000),
    f64::from_bits(0x3fea98ef60000000),
    f64::from_bits(0x3fea41a41a000000),
    f64::from_bits(0x3fe9ec8e95000000),
    f64::from_bits(0x3fe999999a000000),
    f64::from_bits(0x3fe948b0fd000000),
    f64::from_bits(0x3fe8f9c190000000),
    f64::from_bits(0x3fe8acb90f000000),
    f64::from_bits(0x3fe8618618000000),
    f64::from_bits(0x3fe8181818000000),
    f64::from_bits(0x3fe7d05f41000000),
    f64::from_bits(0x3fe78a4c81000000),
    f64::from_bits(0x3fe745d174000000),
    f64::from_bits(0x3fe702e05c000000),
    f64::from_bits(0x3fe6c16c17000000),
    f64::from_bits(0x3fe6816817000000),
    f64::from_bits(0x3fe642c859000000),
    f64::from_bits(0x3fe6058160000000),
    f64::from_bits(0x3fe5c9882c000000),
    f64::from_bits(0x3fe58ed231000000),
    f64::from_bits(0x3fe5555555000000),
    f64::from_bits(0x3fe51d07eb000000),
    f64::from_bits(0x3fe4e5e0a7000000),
    f64::from_bits(0x3fe4afd6a0000000),
    f64::from_bits(0x3fe47ae148000000),
    f64::from_bits(0x3fe446f865000000),
    f64::from_bits(0x3fe4141414000000),
    f64::from_bits(0x3fe3e22cbd000000),
    f64::from_bits(0x3fe3b13b14000000),
    f64::from_bits(0x3fe3813814000000),
    f64::from_bits(0x3fe3521cfb000000),
    f64::from_bits(0x3fe323e34a000000),
    f64::from_bits(0x3fe2f684be000000),
    f64::from_bits(0x3fe2c9fb4e000000),
    f64::from_bits(0x3fe29e412a000000),
    f64::from_bits(0x3fe27350b9000000),
    f64::from_bits(0x3fe2492492000000),
    f64::from_bits(0x3fe21fb781000000),
    f64::from_bits(0x3fe1f7047e000000),
    f64::from_bits(0x3fe1cf06ae000000),
    f64::from_bits(0x3fe1a7b961000000),
    f64::from_bits(0x3fe1811812000000),
    f64::from_bits(0x3fe15b1e5f000000),
    f64::from_bits(0x3fe135c811000000),
    f64::from_bits(0x3fe1111111000000),
    f64::from_bits(0x3fe0ecf56c000000),
    f64::from_bits(0x3fe0c97150000000),
    f64::from_bits(0x3fe0a6810a000000),
    f64::from_bits(0x3fe0842108000000),
    f64::from_bits(0x3fe0624dd3000000),
    f64::from_bits(0x3fe0410410000000),
    f64::from_bits(0x3fe0204081000000),
    f64::from_bits(0x3fe0000000000000),
];

/// `log10(c_j)`.
static LOG10_TL: [f64; 65] = [
    f64::from_bits(0xbd0d45fd6237ebe3),
    f64::from_bits(0x3f7b947689311b6e),
    f64::from_bits(0x3f8b5e909c96d7d5),
    f64::from_bits(0x3f945f4f59ed2165),
    f64::from_bits(0x3f9af5f92cbd8f1e),
    f64::from_bits(0x3fa0ba01a606de8c),
    f64::from_bits(0x3fa3ed119b9a2b7b),
    f64::from_bits(0x3fa714834298eec2),
    f64::from_bits(0x3faa30a9d98357fb),
    f64::from_bits(0x3fad41d512670813),
    f64::from_bits(0x3fb02428c0f65519),
    f64::from_bits(0x3fb1a23444eecc3e),
    f64::from_bits(0x3fb31b30543f4cb4),
    f64::from_bits(0x3fb48f3ed39bfd04),
    f64::from_bits(0x3fb5fe8049a0e423),
    f64::from_bits(0x3fb769140a6aa008),
    f64::from_bits(0x3fb8cf1836c98cb3),
    f64::from_bits(0x3fba30a9d55541a1),
    f64::from_bits(0x3fbb8de4d1ee823e),
    f64::from_bits(0x3fbce6e4202ca2e6),
    f64::from_bits(0x3fbe3bc1accace07),
    f64::from_bits(0x3fbf8c9683b5abd4),
    f64::from_bits(0x3fc06cbd68ca9a6e),
    f64::from_bits(0x3fc11142f19df730),
    f64::from_bits(0x3fc1b3e71fa7a97f),
    f64::from_bits(0x3fc254b4d37a46e3),
    f64::from_bits(0x3fc2f3b6912cbf07),
    f64::from_bits(0x3fc390f683115886),
    f64::from_bits(0x3fc42c7e7fffc5a8),
    f64::from_bits(0x3fc4c65808c78d3c),
    f64::from_bits(0x3fc55e8c50751c55),
    f64::from_bits(0x3fc5f52445dec3d8),
    f64::from_bits(0x3fc68a288c3f1200),
    f64::from_bits(0x3fc71da17bdf0d19),
    f64::from_bits(0x3fc7af973608afd9),
    f64::from_bits(0x3fc84011952a2579),
    f64::from_bits(0x3fc8cf1837a7ea60),
    f64::from_bits(0x3fc95cb2891e43d6),
    f64::from_bits(0x3fc9e8e7b0f869e0),
    f64::from_bits(0x3fca73beaa5db18d),
    f64::from_bits(0x3fcafd3e394558d3),
    f64::from_bits(0x3fcb856cf060d9f1),
    f64::from_bits(0x3fcc0c5134de1ffc),
    f64::from_bits(0x3fcc91f1371bc99f),
    f64::from_bits(0x3fcd1652ffcd3f53),
    f64::from_bits(0x3fcd997c6f635e75),
    f64::from_bits(0x3fce1b733ab90f3b),
    f64::from_bits(0x3fce9c3ceadac856),
    f64::from_bits(0x3fcf1bdeec43a305),
    f64::from_bits(0x3fcf9a5e7a5fa3fe),
    f64::from_bits(0x3fd00be05ac02f2b),
    f64::from_bits(0x3fd04a054d81a2d4),
    f64::from_bits(0x3fd087a0835957fb),
    f64::from_bits(0x3fd0c4b457099517),
    f64::from_bits(0x3fd101431aa1fe51),
    f64::from_bits(0x3fd13d4f08b98dd8),
    f64::from_bits(0x3fd178da53edb892),
    f64::from_bits(0x3fd1b3e71e9f9d58),
    f64::from_bits(0x3fd1ee777defdeed),
    f64::from_bits(0x3fd2288d7b48e23b),
    f64::from_bits(0x3fd2622b0f52e49f),
    f64::from_bits(0x3fd29b522a4c6314),
    f64::from_bits(0x3fd2d404b0e30f80),
    f64::from_bits(0x3fd30c4478f3fbe5),
    f64::from_bits(0x3fd34413509f7915),
];

/// The fast path's degree-3 correction.
static LOG10_B: [f64; 3] = [
    f64::from_bits(0x3fdbcb7b15c5a2f8), // 0x1.bcb7b15c5a2f8p-2
    f64::from_bits(0xbfcbcbb1dbb88eba), // -0x1.bcbb1dbb88ebap-3
    f64::from_bits(0x3fc2871c39d521c6), // 0x1.2871c39d521c6p-3
];

/// The accurate path's degree-7 replacement.
static LOG10_C: [f64; 7] = [
    f64::from_bits(0x3fdbcb7b1526e50e), // 0x1.bcb7b1526e50ep-2
    f64::from_bits(0xbfcbcb7b1526e53d), // -0x1.bcb7b1526e53dp-3
    f64::from_bits(0x3fc287a7636f3fa2), // 0x1.287a7636f3fa2p-3
    f64::from_bits(0xbfbbcb7b146a14b3), // -0x1.bcb7b146a14b3p-4
    f64::from_bits(0x3fb63c627d5219cb), // 0x1.63c627d5219cbp-4
    f64::from_bits(0xbfb2880736c8762d), // -0x1.2880736c8762dp-4
    f64::from_bits(0x3fafc1ecf913961a), // 0x1.fc1ecf913961ap-5
];

/// `10^k` for `k` in `0..=10`, where `log10` is exactly `k`. Zero-padded
/// to 16 so the index computed from the exponent needs no bounds test.
static LOG10_ST: [f32; 16] = [
    f32::from_bits(0x3f800000),
    f32::from_bits(0x41200000),
    f32::from_bits(0x42c80000),
    f32::from_bits(0x447a0000),
    f32::from_bits(0x461c4000),
    f32::from_bits(0x47c35000),
    f32::from_bits(0x49742400),
    f32::from_bits(0x4b189680),
    f32::from_bits(0x4cbebc20),
    f32::from_bits(0x4e6e6b28),
    f32::from_bits(0x501502f9),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
    f32::from_bits(0x00000000),
];

/// `log10(2)`, single-rounded: the fast path's per-exponent scale.
const LOG10_2: f64 = f64::from_bits(0x3fd34413509f79ff);
/// The fast path's error bound.
const LOG10_EPS: f64 = f64::from_bits(0x3ddb008000000000);
/// The accurate path's `log10(2)`, deliberately a *different* rounding —
/// truncated two bits shorter so that `LOG10_2A * e` stays exact, with
/// [`LOG10_2_CORR`] carrying what that costs.
const LOG10_2A: f64 = f64::from_bits(0x3fd34413509f7a00);
/// The correction to [`LOG10_2A`], subtracted per unit of exponent.
const LOG10_2_CORR: f64 = f64::from_bits(0x3c90cee0ed4ca7e9);

/// `log10(x)`, correctly rounded.
pub fn log10(x: f32) -> f32 {
    let mut ux = x.to_bits();

    // Subnormal, zero, infinite, NaN, or — `ux` being unsigned — negative.
    if !(1 << 23..0x7f80_0000).contains(&ux) {
        if ux == 0 || ux >= 0x7f80_0000 {
            // `as_special`: `+inf` passes through, `±0` is a pole, a NaN is
            // quietened, and anything else here is negative.
            if ux == 0x7f80_0000 {
                return x;
            }
            if ux << 1 == 0 {
                return NEG_INF;
            }
            if ux << 1 > 0xff00_0000 {
                return x + x;
            }
            return invalid(x);
        }
        // Subnormal: normalise, and pay for it in the exponent field.
        let n = ux.leading_zeros() - 8;
        ux <<= n;
        ux = ux.wrapping_sub(n << 23);
    }

    let m = ux & ((1 << 23) - 1);
    let j = ((m + (1 << (23 - 7))) >> (23 - 6)) as usize;
    let ix = LOG10_TR[j];
    let l = LOG10_TL[j];
    let e = (ux as i32 >> 23) - 127;

    // `je = floor((e + 1) * log10(2))` by a fixed-point multiply: the index
    // of the power of ten this exponent could be, if `x` is one.
    let je = ((((e + 1) as u32).wrapping_mul(0x4d1_04d4)) >> 28) as usize;
    if ux == LOG10_ST[je].to_bits() {
        return je as f32;
    }

    let tz = f64::from_bits(((i64::from(m) | (1023i64 << 23)) << (52 - 23)) as u64);
    let z = tz * ix - 1.0;
    let z2 = z * z;
    let ef = f64::from(e);
    let r = ((ef * LOG10_2 + l) + z * LOG10_B[0]) + z2 * (LOG10_B[1] + z * LOG10_B[2]);
    let ub = r as f32;
    let lb = (r + LOG10_EPS) as f32;
    if ub == lb {
        return ub;
    }

    let mut f = z
        * ((LOG10_C[0] + z * LOG10_C[1])
            + z2 * ((LOG10_C[2] + z * LOG10_C[3])
                + z2 * (LOG10_C[4] + z * LOG10_C[5] + z2 * LOG10_C[6])));
    f -= LOG10_2_CORR * ef;
    f += l - LOG10_TL[0];
    let el = ef * LOG10_2A;
    let mut r = el + f;
    if r.to_bits() & ((1 << 28) - 1) == 0 {
        // On an `f32` rounding boundary: recover what `el + f` lost and let
        // it decide the direction.
        let dr = (el - r) + f;
        r += dr * 32.0;
    }
    r as f32
}
