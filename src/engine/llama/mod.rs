//! llama.cpp / GGUF embedding backend — the sole [`EmbeddingBackend`] implementation.
//!
//! Loads a GGUF model through the prebuilt static llama.cpp archives (raw FFI in [`ffi`]),
//! runs it as a non-causal embedding model with last-token pooling, and returns the raw,
//! un-normalized pooled `raw_embedding_dimension` vector per input. Prefixing, tokenization
//! and truncation/normalization are handled by the shared engine layer.

mod ffi;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::{Mutex, Once};
use std::time::Instant;

use crate::error::{InferenceError, LTEmbedError, ModelLoadError};
use crate::traits::tokenizer::TokenizerOutput;

use super::backend::{BackendRunProfile, EmbeddingBackend};

/// llama.cpp log callback: drops INFO/DEBUG chatter (e.g. per-token "control token ... is not
/// marked as EOG" vocab-load lines) and forwards WARN and above to stderr.
extern "C" fn quiet_log_callback(
    level: ffi::ggml_log_level,
    text: *const c_char,
    _user: *mut c_void,
) {
    if level >= ffi::GGML_LOG_LEVEL_WARN && !text.is_null() {
        if let Ok(s) = unsafe { CStr::from_ptr(text) }.to_str() {
            eprint!("{s}");
        }
    }
}

/// llama.cpp requires a one-time process-global backend init. It is never freed (process
/// exit reclaims it); freeing per-backend would break any concurrently-live backend.
fn ensure_backend_initialized() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        // Install the log filter before init/model load so it also catches load-time lines.
        ffi::llama_log_set(Some(quiet_log_callback), std::ptr::null_mut());
        ffi::llama_backend_init();
    });
}

/// Owns the raw llama.cpp handles. The pointers are only ever touched behind the
/// [`LlamaBackend`] mutex, so it is sound to move the struct across threads.
struct ModelContext {
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
}

// Safety: the handles are accessed exclusively through `LlamaBackend`'s `Mutex`, which
// serializes all FFI calls that read or mutate them.
unsafe impl Send for ModelContext {}

impl Drop for ModelContext {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free(self.ctx);
            ffi::llama_model_free(self.model);
        }
    }
}

pub(crate) struct LlamaBackend {
    inner: Mutex<ModelContext>,
    raw_dim: usize,
    /// Context length (also the max single-sequence token count llama can pool at once).
    n_ctx: usize,
}

impl LlamaBackend {
    /// Load a GGUF model as a CPU embedding backend.
    ///
    /// `context_length` sizes the KV context and the (single) micro-batch; a non-causal
    /// embedding model must fit each whole sequence in one micro-batch, so this is also the
    /// longest input the backend accepts.
    pub(crate) fn load(
        gguf_path: &Path,
        raw_embedding_dimension: usize,
        context_length: usize,
        n_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        ensure_backend_initialized();

        let cpath =
            CString::new(gguf_path.as_os_str().to_string_lossy().as_bytes()).map_err(|err| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "GGUF path is not a valid C string: {err}"
                )))
            })?;

        unsafe {
            let mut mparams = ffi::llama_model_default_params();
            mparams.n_gpu_layers = 0; // CPU only

            let model = ffi::llama_model_load_from_file(cpath.as_ptr(), mparams);
            if model.is_null() {
                return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "failed to load GGUF model at {}",
                    gguf_path.display()
                ))));
            }

            let n_embd = ffi::llama_model_n_embd(model) as usize;
            if n_embd != raw_embedding_dimension {
                ffi::llama_model_free(model);
                return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "GGUF embedding dimension {n_embd} != expected {raw_embedding_dimension}"
                ))));
            }

            let n = context_length as u32;
            let mut cparams = ffi::llama_context_default_params();
            cparams.embeddings = true;
            cparams.pooling_type = ffi::LLAMA_POOLING_TYPE_LAST;
            // Encoder embedding model: force bidirectional (non-causal) attention rather than
            // relying on the architecture default, so pooling sees full-context representations.
            cparams.attention_type = ffi::LLAMA_ATTENTION_TYPE_NON_CAUSAL;
            cparams.n_ctx = n;
            cparams.n_batch = n;
            cparams.n_ubatch = n;
            cparams.n_seq_max = 1;
            cparams.n_threads = n_threads as i32;
            cparams.n_threads_batch = n_threads as i32;

            let ctx = ffi::llama_init_from_model(model, cparams);
            if ctx.is_null() {
                ffi::llama_model_free(model);
                return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(
                    "failed to create llama.cpp context".into(),
                )));
            }
            ffi::llama_set_embeddings(ctx, true);

            Ok(Self {
                inner: Mutex::new(ModelContext { model, ctx }),
                raw_dim: raw_embedding_dimension,
                n_ctx: context_length,
            })
        }
    }

    /// Raw, un-normalized pooled embedding for a single tokenized input.
    ///
    /// Safety: caller holds the backend mutex (via `&ModelContext`), so no other FFI call
    /// touches `ctx` concurrently.
    unsafe fn embed_one(
        &self,
        ctx: *mut ffi::llama_context,
        tokens: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError> {
        if tokens.is_empty() {
            return Err(LTEmbedError::Inference(InferenceError::AllPadding));
        }
        if tokens.len() > self.n_ctx {
            return Err(LTEmbedError::Inference(InferenceError::SequenceTooLong {
                encoded: tokens.len(),
                model: self.n_ctx,
            }));
        }

        // Fresh KV state per independent input.
        ffi::llama_memory_clear(ffi::llama_get_memory(ctx), true);

        let n = tokens.len() as i32;
        let mut batch = ffi::llama_batch_init(n, 0, 1);
        for (i, &tok) in tokens.iter().enumerate() {
            *batch.token.add(i) = tok as ffi::llama_token;
            *batch.pos.add(i) = i as ffi::llama_pos;
            *batch.n_seq_id.add(i) = 1;
            *(*batch.seq_id.add(i)).add(0) = 0;
            *batch.logits.add(i) = 1; // request output for every token (pooled)
        }
        batch.n_tokens = n;

        // Non-causal encoder context (see `load`): call encode(), not decode(). Using decode()
        // makes recent llama.cpp warn and silently fall back to encode() per input.
        let rc = ffi::llama_encode(ctx, batch);
        if rc != 0 {
            ffi::llama_batch_free(batch);
            return Err(LTEmbedError::Inference(InferenceError::Backend(format!(
                "llama_encode failed (rc={rc})"
            ))));
        }
        ffi::llama_synchronize(ctx);

        let ptr = ffi::llama_get_embeddings_seq(ctx, 0);
        if ptr.is_null() {
            ffi::llama_batch_free(batch);
            return Err(LTEmbedError::Inference(InferenceError::Tensor(
                "llama_get_embeddings_seq returned null".into(),
            )));
        }
        let raw = std::slice::from_raw_parts(ptr, self.raw_dim).to_vec();
        ffi::llama_batch_free(batch);
        Ok(raw)
    }
}

impl EmbeddingBackend for LlamaBackend {
    fn raw_embedding_dimension(&self) -> usize {
        self.raw_dim
    }

    fn embed(
        &self,
        tokenized: &[TokenizerOutput],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<BackendRunProfile>), LTEmbedError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| LTEmbedError::Inference(InferenceError::MutexPoisoned))?;
        let ctx = guard.ctx;

        let mut embeddings = Vec::with_capacity(tokenized.len());
        let mut run_ms = 0.0;
        // Sequences are decoded independently (one decode per input). True multi-sequence
        // batching in a single decode is a future throughput optimization.
        for item in tokenized {
            let start = collect_profile.then(Instant::now);
            let raw = unsafe { self.embed_one(ctx, &item.input_ids) }?;
            if let Some(start) = start {
                run_ms += start.elapsed().as_secs_f64() * 1_000.0;
            }
            embeddings.push(raw);
        }

        let profile = collect_profile.then_some(BackendRunProfile {
            tensorize_ms: 0.0,
            run_ms,
            extract_ms: 0.0,
        });
        Ok((embeddings, profile))
    }
}
