// Links the prebuilt Graviton2 static llama.cpp archives from the
// `static-llama-cpp-rs-builder` release. llama.cpp/GGUF is the crate's only inference
// backend, so this always runs — the crate builds only on aarch64-linux with the artifacts
// present (see docs/development.md).
//
// Mirrors the release's `consume.build.rs` (the single source of truth for the tested
// link line). Point STATIC_LLAMA_DIR at a VERIFIED, extracted release directory
// (SHA256SUMS already checked) containing `lib/*.a` and `bindings.rs`.
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=STATIC_LLAMA_DIR");

    let dir = env::var("STATIC_LLAMA_DIR").map(PathBuf::from).expect(
        "STATIC_LLAMA_DIR must point at the extracted, SHA-verified static-llama-cpp release",
    );

    let libdir = dir.join("lib");
    assert!(
        libdir.join("libllama.a").exists(),
        "libllama.a not found in {} — set STATIC_LLAMA_DIR to the extracted, SHA-verified release",
        libdir.display()
    );
    println!("cargo:rustc-link-search=native={}", libdir.display());

    // Static archives in dependency order (== build-info.json `link_line`).
    for lib in ["llama", "ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // C++ runtime + OS deps (dynamic, from the base image). No -lgomp: OpenMP disabled.
    for lib in ["stdc++", "pthread", "m", "dl"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    // Expose the generated FFI bindings path for `include!(env!("STATIC_LLAMA_BINDINGS"))`.
    let bindings = dir.join("bindings.rs");
    assert!(
        bindings.exists(),
        "bindings.rs not found in {}",
        dir.display()
    );
    println!(
        "cargo:rustc-env=STATIC_LLAMA_BINDINGS={}",
        bindings.display()
    );
}
