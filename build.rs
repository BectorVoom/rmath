fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Ok(output) = std::process::Command::new("rustc").arg("-Vv").output() {
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout);
            let single_line = s.trim().replace('\n', " | ");
            println!("cargo:rustc-env=RMATH_RUSTC_VERBOSE={}", single_line);
        }
    }
}
