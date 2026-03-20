// build.rs — Compile llama.cpp/ggml for the `ggml-backend` feature.
//
// Strategy (per issue #59):
//   -ffunction-sections + -fdata-sections  → enables Linker GC in downstream
//   -Os                                    → minimize compiled kernel footprint
//
// C and C++ sources are compiled in separate cc::Build instances to avoid
// flag conflicts (-std=c11 is invalid for C++ mode).

fn main() {
    if std::env::var("CARGO_FEATURE_GGML_BACKEND").is_err() {
        return;
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vendor = "vendor/llama.cpp";

    // ── Shared flag builder ───────────────────────────────────────────────────
    let common_flags: Vec<&str> = vec!["-ffunction-sections", "-fdata-sections", "-Os"];
    let common_defines: &[(&str, Option<&str>)] = &[
        ("GGML_USE_CPU", None),
        ("LLAMA_NO_METAL", None),
        ("LLAMA_NO_CUDA", None),
        ("LLAMA_NO_VULKAN", None),
        ("GGML_BACKEND_DL", None),
        // Version strings normally injected by CMake from git
        ("GGML_VERSION", Some("\"0.9.8\"")),
        ("GGML_COMMIT", Some("\"unknown\"")),
    ];
    let includes = vec![
        format!("{vendor}/ggml/include"),
        format!("{vendor}/ggml/src"),
        format!("{vendor}/ggml/src/ggml-cpu"),
        format!("{vendor}/include"),
        format!("{vendor}/src"),
        format!("{vendor}/src/models"),
        "ffi".to_string(),
    ];

    let arm_flags: Vec<&str> = if target_arch == "aarch64" {
        vec!["-march=armv8.2-a+fp16+dotprod"]
    } else {
        vec![]
    };
    let arm_defines: &[(&str, Option<&str>)] = if target_arch == "aarch64" {
        &[("GGML_USE_NEON", None)]
    } else {
        &[]
    };

    // Helper: apply common setup to a Build
    fn setup(
        b: &mut cc::Build,
        flags: &[&str],
        extra_flags: &[&str],
        defines: &[(&str, Option<&str>)],
        extra_defines: &[(&str, Option<&str>)],
        includes: &[String],
    ) {
        for f in flags {
            b.flag(f);
        }
        for f in extra_flags {
            b.flag(f);
        }
        for (k, v) in defines {
            b.define(k, *v);
        }
        for (k, v) in extra_defines {
            b.define(k, *v);
        }
        for inc in includes {
            b.include(inc);
        }
    }

    // ── C sources ─────────────────────────────────────────────────────────────
    let c_files_common = [
        format!("{vendor}/ggml/src/ggml.c"),
        format!("{vendor}/ggml/src/ggml-alloc.c"),
        format!("{vendor}/ggml/src/ggml-quants.c"),
        format!("{vendor}/ggml/src/ggml-cpu/ggml-cpu.c"),
        format!("{vendor}/ggml/src/ggml-cpu/quants.c"),
    ];

    let arch_dir = if target_arch == "aarch64" {
        "arm"
    } else if target_arch == "x86_64" || target_arch == "x86" {
        "x86"
    } else {
        ""
    };

    let mut c_build = cc::Build::new();
    c_build.flag("-std=c11");
    setup(
        &mut c_build,
        &common_flags,
        &arm_flags,
        common_defines,
        arm_defines,
        &includes,
    );
    for f in &c_files_common {
        c_build.file(f);
    }
    if !arch_dir.is_empty() {
        c_build.file(format!(
            "{vendor}/ggml/src/ggml-cpu/arch/{arch_dir}/quants.c"
        ));
    }
    c_build.compile("ltggml_c");

    // ── C++ sources ───────────────────────────────────────────────────────────
    let cpp_files_ggml = [
        format!("{vendor}/ggml/src/ggml.cpp"),
        format!("{vendor}/ggml/src/ggml-backend.cpp"),
        format!("{vendor}/ggml/src/ggml-opt.cpp"),
        format!("{vendor}/ggml/src/ggml-backend-reg.cpp"),
        format!("{vendor}/ggml/src/ggml-threading.cpp"),
        format!("{vendor}/ggml/src/gguf.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/ggml-cpu.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/ops.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/binary-ops.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/unary-ops.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/repack.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/traits.cpp"),
        format!("{vendor}/ggml/src/ggml-cpu/vec.cpp"),
    ];

    // Compile every .cpp in src/ and src/models/ — the files are tightly coupled
    // and CMake builds them all. Filtering here risks missing symbols.
    let llama_src = format!("{vendor}/src");
    let llama_src_path = std::path::Path::new(&llama_src);
    let llama_models_path = llama_src_path.join("models");

    let mut cpp_files_llama: Vec<String> = std::fs::read_dir(llama_src_path)
        .expect("vendor/llama.cpp/src not found")
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.extension()? == "cpp" {
                Some(p.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();

    let models_files: Vec<String> = std::fs::read_dir(&llama_models_path)
        .expect("vendor/llama.cpp/src/models not found")
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.extension()? == "cpp" {
                Some(p.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();

    cpp_files_llama.extend(models_files);

    let mut cpp_build = cc::Build::new();
    cpp_build.cpp(true).flag("-std=c++17");
    setup(
        &mut cpp_build,
        &common_flags,
        &arm_flags,
        common_defines,
        arm_defines,
        &includes,
    );
    for f in &cpp_files_ggml {
        cpp_build.file(f);
    }
    if target_os == "linux" {
        cpp_build.file(format!("{vendor}/ggml/src/ggml-cpu/hbm.cpp"));
    }
    if !arch_dir.is_empty() {
        cpp_build.file(format!(
            "{vendor}/ggml/src/ggml-cpu/arch/{arch_dir}/cpu-feats.cpp"
        ));
        cpp_build.file(format!(
            "{vendor}/ggml/src/ggml-cpu/arch/{arch_dir}/repack.cpp"
        ));
    }
    for f in &cpp_files_llama {
        cpp_build.file(f);
    }

    // Our thin C++ wrappers
    cpp_build.file("ffi/embedding.cpp");
    cpp_build.file("ffi/cross_encoder.cpp");

    cpp_build.compile("ltggml_cpp");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ffi/embedding.h");
    println!("cargo:rerun-if-changed=ffi/embedding.cpp");
    println!("cargo:rerun-if-changed=ffi/cross_encoder.h");
    println!("cargo:rerun-if-changed=ffi/cross_encoder.cpp");
}
