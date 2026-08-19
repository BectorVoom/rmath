use rmath::fast;

#[test]
fn test_fast_free_functions_f64() {
    let x = 1.0_f64;
    assert!((fast::exp(x) - x.exp()).abs() < 1e-14);
    assert!((fast::ln(x) - x.ln()).abs() < 1e-14);
    assert!((fast::sin(x) - x.sin()).abs() < 1e-14);
    assert!((fast::cos(x) - x.cos()).abs() < 1e-14);
    assert!((fast::tan(x) - x.tan()).abs() < 1e-14);
    assert!((fast::sqrt(4.0_f64) - 2.0).abs() < 1e-15);
    assert!((fast::pow(2.0_f64, 3.0_f64) - 8.0).abs() < 1e-14);
    assert!((fast::erf(x) - 0.8427007929497148).abs() < 1e-14);
    assert!((fast::erfc(x) - 0.15729920705028513).abs() < 1e-14);
    assert!((fast::cbrt(27.0_f64) - 3.0).abs() < 1e-14);

    let (s, c) = fast::sincos(x);
    assert!((s - x.sin()).abs() < 1e-14);
    assert!((c - x.cos()).abs() < 1e-14);
}

#[test]
fn test_fast_free_functions_f32() {
    let x = 1.0_f32;
    assert!((fast::exp(x) - x.exp()).abs() < 1e-5);
    assert!((fast::ln(x) - x.ln()).abs() < 1e-5);
    assert!((fast::sin(x) - x.sin()).abs() < 1e-5);
    assert!((fast::cos(x) - x.cos()).abs() < 1e-5);
    assert!((fast::sqrt(4.0_f32) - 2.0).abs() < 1e-6);
    assert!((fast::pow(2.0_f32, 3.0_f32) - 8.0).abs() < 1e-5);
}

#[test]
#[cfg(feature = "wide")]
fn test_fast_free_functions_vectors() {
    use wide::{f64x4, f32x8};

    let vx = f64x4::splat(1.0);
    let exp_vx = fast::exp(vx);
    assert!((exp_vx.to_array()[0] - 1.0_f64.exp()).abs() < 1e-14);

    let sin_vx = fast::sin(vx);
    assert!((sin_vx.to_array()[0] - 1.0_f64.sin()).abs() < 1e-14);

    let (s_v, c_v) = fast::sincos(vx);
    assert!((s_v.to_array()[0] - 1.0_f64.sin()).abs() < 1e-14);
    assert!((c_v.to_array()[0] - 1.0_f64.cos()).abs() < 1e-14);

    let vf = f32x8::splat(1.0);
    let exp_vf = fast::exp(vf);
    assert!((exp_vf.to_array()[0] - 1.0_f32.exp()).abs() < 1e-5);
}
