// src/ffi.rs — Raw unsafe bindings to the ltggml C library.
// Only compiled when the `ggml-backend` feature is active.

#[cfg(feature = "ggml-backend")]
pub mod embedding {
    use std::os::raw::{c_char, c_float, c_int};

    /// Opaque handle to a loaded GGUF embedding model + context.
    pub enum LtEmbedderOpaque {}

    extern "C" {
        pub fn lt_emb_load(model_path: *const c_char, n_threads: c_int) -> *mut LtEmbedderOpaque;

        pub fn lt_emb_compute(
            ctx: *mut LtEmbedderOpaque,
            text: *const c_char,
            out_buf: *mut c_float,
            buf_len: c_int,
        ) -> c_int;

        pub fn lt_emb_dim(ctx: *mut LtEmbedderOpaque) -> c_int;

        pub fn lt_emb_free(ctx: *mut LtEmbedderOpaque);
    }
}

#[cfg(feature = "ggml-backend")]
pub mod cross_encoder {
    use std::os::raw::{c_char, c_float, c_int};

    /// Opaque handle to a loaded GGUF cross-encoder model + context.
    pub enum LtCrossEncoderOpaque {}

    extern "C" {
        pub fn lt_ce_load(model_path: *const c_char, n_threads: c_int)
            -> *mut LtCrossEncoderOpaque;

        pub fn lt_ce_score(
            ctx: *mut LtCrossEncoderOpaque,
            query: *const c_char,
            document: *const c_char,
        ) -> c_float;

        pub fn lt_ce_free(ctx: *mut LtCrossEncoderOpaque);
    }
}
