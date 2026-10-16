//! llama.cpp / GGUF embedding backend — the sole [`EmbeddingBackend`] implementation.
//!
//! Loads a GGUF model through the prebuilt static llama.cpp archives (raw FFI in [`ffi`]),
//! runs it as a non-causal embedding model with mean pooling, and returns the raw,
//! un-normalized pooled vector per input. For EmbeddingGemma 2 that is the 768-wide output of
//! the model's 512→768 projection, not its 512-wide hidden state. It also tokenizes, with the
//! GGUF's own vocabulary: the engine has no other tokenizer. Prefixing, the input length
//! limit and truncation/normalization are handled by the shared engine layer.

mod ffi;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::{Mutex, Once};
use std::time::Instant;

use crate::error::{InferenceError, LTEmbedError, ModelLoadError};

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
    /// Length of each pooled vector. `load` checked that the GGUF's `n_embd_out` equals it
    /// (see [`check_embedding_widths`]).
    raw_dim: usize,
    /// Context length (also the max single-sequence token count llama can pool at once).
    n_ctx: usize,
}

/// Checks the GGUF's embedding widths against the bundle's `raw_embedding_dimension`.
///
/// `n_embd` is the model's hidden width (EmbeddingGemma 2: 512). `n_embd_out` is the length
/// of each pooled vector, i.e. of the buffer behind `llama_get_embeddings_seq`: llama.cpp sets
/// it to the width of the model's output projection (EmbeddingGemma 2: 768), or to `n_embd`
/// for a model without one. `embed_one` reads `raw_embedding_dimension` floats from that
/// buffer, so `n_embd_out` must equal it; `n_embd` only has to be a valid width. The widths go
/// through `usize::try_from`, not `as`, so a negative value cannot wrap into a match.
fn check_embedding_widths(
    n_embd: i32,
    n_embd_out: i32,
    raw_embedding_dimension: usize,
) -> Result<(), LTEmbedError> {
    if !usize::try_from(n_embd).is_ok_and(|width| width > 0) {
        return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "GGUF n_embd {n_embd} is not a valid hidden width"
        ))));
    }
    if usize::try_from(n_embd_out).ok() != Some(raw_embedding_dimension) {
        return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "GGUF n_embd_out {n_embd_out} != expected raw_embedding_dimension \
             {raw_embedding_dimension} (GGUF n_embd is {n_embd})"
        ))));
    }
    Ok(())
}

/// Checks the pooling type a GGUF declares under `key` (`<architecture>.pooling_type`).
///
/// The backend always runs llama.cpp's MEAN pooling. A GGUF whose converter recorded another
/// pooling, or none at all, belongs to a model that mean pooling would embed wrongly, so it is
/// rejected. `declared` is the value as llama.cpp formats it: a decimal integer.
fn check_declared_pooling(key: &str, declared: Option<&str>) -> Result<(), LTEmbedError> {
    let mean = ffi::LLAMA_POOLING_TYPE_MEAN;
    match declared {
        Some(value) if value.parse::<ffi::llama_pooling_type>() == Ok(mean) => Ok(()),
        Some(value) => Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "GGUF {key} is {value}, but this engine runs mean pooling ({mean}) only"
        )))),
        None => Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "GGUF has no {key}; this engine runs mean pooling ({mean}) only and requires a \
             GGUF that declares it"
        )))),
    }
}

/// Longest GGUF metadata value [`gguf_meta_value`] reads: enough for an architecture name or
/// an integer.
const META_VALUE_CAPACITY: usize = 128;

/// A GGUF metadata value as llama.cpp formats it (numbers in decimal), or `None` if the GGUF
/// has no `key`. A value of `META_VALUE_CAPACITY` bytes or more is an error.
///
/// Safety: `model` is a live model handle.
unsafe fn gguf_meta_value(
    model: *const ffi::llama_model,
    key: &str,
) -> Result<Option<String>, LTEmbedError> {
    let invalid = |reason: String| {
        LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "GGUF metadata {key}: {reason}"
        )))
    };
    let ckey = CString::new(key).map_err(|err| invalid(err.to_string()))?;
    let mut buf = [0u8; META_VALUE_CAPACITY];
    // Like snprintf: writes at most `buf.len() - 1` bytes plus a NUL and returns the full
    // value length, or -1 (writing an empty string) when the key is absent.
    let len =
        ffi::llama_model_meta_val_str(model, ckey.as_ptr(), buf.as_mut_ptr().cast(), buf.len());
    let Ok(len) = usize::try_from(len) else {
        return Ok(None);
    };
    if len >= buf.len() {
        return Err(invalid(format!("{len}-byte value is longer than expected")));
    }
    String::from_utf8(buf[..len].to_vec())
        .map(Some)
        .map_err(|err| invalid(err.to_string()))
}

/// Reads `<architecture>.pooling_type` from the GGUF and checks it with
/// [`check_declared_pooling`].
///
/// Safety: `model` is a live model handle.
unsafe fn check_gguf_pooling(model: *const ffi::llama_model) -> Result<(), LTEmbedError> {
    let Some(architecture) = gguf_meta_value(model, "general.architecture")? else {
        return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(
            "GGUF has no general.architecture".into(),
        )));
    };
    let key = format!("{architecture}.pooling_type");
    let declared = gguf_meta_value(model, &key)?;
    check_declared_pooling(&key, declared.as_deref())
}

/// Tokenizes the whole of `text` with `vocab`, adding the special tokens the GGUF declares
/// (BOS/EOS) and matching special-token text, as llama.cpp's own tools do.
///
/// Safety: `vocab` belongs to a live model.
unsafe fn tokenize_with_vocab(
    vocab: *const ffi::llama_vocab,
    text: &str,
) -> Result<Vec<u32>, String> {
    let too_long = || format!("a text of {} bytes is too long to tokenize", text.len());
    let text_len = i32::try_from(text.len()).map_err(|_| too_long())?;
    // Byte fallback yields at most one token per byte; BOS and EOS add two more.
    let capacity = text.len() + 2;
    let n_tokens_max = i32::try_from(capacity).map_err(|_| too_long())?;
    let mut tokens: Vec<ffi::llama_token> = vec![0; capacity];
    // `tokens` holds `n_tokens_max` elements, the most llama.cpp writes.
    let n = ffi::llama_tokenize(
        vocab,
        text.as_ptr().cast::<c_char>(),
        text_len,
        tokens.as_mut_ptr(),
        n_tokens_max,
        true,
        true,
    );
    let n = usize::try_from(n).map_err(|_| {
        format!(
            "llama_tokenize failed (rc={n}) for a text of {} bytes",
            text.len()
        )
    })?;
    tokens.truncate(n);
    tokens
        .into_iter()
        .map(|token| {
            u32::try_from(token).map_err(|_| format!("llama_tokenize returned token id {token}"))
        })
        .collect()
}

/// Text the load-time check tokenizes with the GGUF's vocabulary (see
/// [`check_special_tokens`]).
const SPECIAL_TOKENS_CHECK_TEXT: &str = "LTEmbed load check";

/// Checks that the GGUF's tokenization of [`SPECIAL_TOKENS_CHECK_TEXT`], `ids`, starts with the
/// vocabulary's BOS token and ends with its EOS token.
///
/// The model's SentenceTransformers pipeline wraps every input in BOS and EOS, and mean
/// pooling averages over them. A GGUF whose tokenizer metadata does not add both
/// (`tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token`) would embed every input
/// differently, so it is rejected. `bos` and `eos` are the vocabulary's ids, negative
/// (`LLAMA_TOKEN_NULL`) when it has none.
fn check_special_tokens(
    ids: &[u32],
    bos: ffi::llama_token,
    eos: ffi::llama_token,
) -> Result<(), LTEmbedError> {
    let runtime = |msg: String| LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg));
    let token = |name: &str, id: ffi::llama_token| {
        u32::try_from(id).map_err(|_| runtime(format!("the GGUF's vocabulary has no {name} token")))
    };
    let (bos, eos) = (token("BOS", bos)?, token("EOS", eos)?);
    let adds_bos = ids.first() == Some(&bos);
    let adds_eos = ids.len() >= 2 && ids.last() == Some(&eos);
    let missing = match (adds_bos, adds_eos) {
        (true, true) => return Ok(()),
        (false, true) => "BOS",
        (true, false) => "EOS",
        (false, false) => "BOS or EOS",
    };
    let ends = match (ids.first(), ids.last()) {
        (Some(first), Some(last)) => format!("{} tokens, from {first} to {last}", ids.len()),
        _ => "no tokens".to_string(),
    };
    Err(runtime(format!(
        "the GGUF's tokenization adds no {missing}: {SPECIAL_TOKENS_CHECK_TEXT:?} gives {ends}, \
         but BOS is {bos} and EOS {eos}. The model wraps every input in BOS and EOS, so this \
         GGUF would embed differently"
    )))
}

/// Tokenizes [`SPECIAL_TOKENS_CHECK_TEXT`] and checks the result with
/// [`check_special_tokens`].
///
/// Safety: `vocab` belongs to a live model.
unsafe fn check_gguf_special_tokens(vocab: *const ffi::llama_vocab) -> Result<(), LTEmbedError> {
    let ids = tokenize_with_vocab(vocab, SPECIAL_TOKENS_CHECK_TEXT)
        .map_err(|msg| LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg)))?;
    check_special_tokens(
        &ids,
        ffi::llama_vocab_bos(vocab),
        ffi::llama_vocab_eos(vocab),
    )
}

/// Checks that a sequence of `n_tokens` tokens fits one `llama_encode` call on a context of
/// `n_ctx` tokens.
///
/// The engine never passes an empty sequence, because the GGUF's tokenization adds BOS and
/// EOS (checked at load), nor one longer than the bundle's `max_length`, which sizes the
/// context: it returns `InputTooLong` first. A sequence outside `1..=n_ctx` is therefore an
/// internal error. It must not reach llama.cpp, which rejects an empty batch and aborts the
/// process on a sequence longer than its micro-batch.
fn check_sequence_length(n_tokens: usize, n_ctx: usize) -> Result<(), LTEmbedError> {
    if n_tokens == 0 || n_tokens > n_ctx {
        return Err(LTEmbedError::Inference(InferenceError::Internal(format!(
            "the backend got a sequence of {n_tokens} tokens, outside 1..={n_ctx}"
        ))));
    }
    Ok(())
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
        // Validate before narrowing into llama.cpp's u32/i32 fields: an oversized context or
        // thread count would otherwise wrap silently and mis-size the context.
        let n_ctx = u32::try_from(context_length)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "context_length {context_length} out of range (must be 1..={})",
                    u32::MAX
                )))
            })?;
        let n_threads = i32::try_from(n_threads)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "n_threads {n_threads} out of range (must be 1..={})",
                    i32::MAX
                )))
            })?;

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

            if let Err(err) = check_embedding_widths(
                ffi::llama_model_n_embd(model),
                ffi::llama_model_n_embd_out(model),
                raw_embedding_dimension,
            )
            .and_then(|()| check_gguf_pooling(model))
            .and_then(|()| check_gguf_special_tokens(ffi::llama_model_get_vocab(model)))
            {
                ffi::llama_model_free(model);
                return Err(err);
            }

            let n = n_ctx;
            let mut cparams = ffi::llama_context_default_params();
            cparams.embeddings = true;
            // Mean over every token of the sequence, prompt and BOS/EOS included, as the
            // model's SentenceTransformers pooling does (`include_prompt: true`). The GGUF's own
            // declaration was checked above; this states it rather than relying on it.
            cparams.pooling_type = ffi::LLAMA_POOLING_TYPE_MEAN;
            // No effect with the pinned llama.cpp (v0.7.0): `llama_encode` always runs non-causal
            // (bidirectional) attention, whatever this says, and with `n_batch = n_ctx` the
            // setting changes no other parameter; embeddings are bit-identical with
            // LLAMA_ATTENTION_TYPE_CAUSAL. It states what the model needs, as its GGUF declares
            // (`gemma-embedding2.attention.causal = false`), in case a later llama.cpp reads
            // it in `llama_encode`.
            cparams.attention_type = ffi::LLAMA_ATTENTION_TYPE_NON_CAUSAL;
            // Stated rather than left to AUTO, which also resolves to enabled on CPU. Flash
            // attention stores K and V in F16 (both are RMS-normed in this architecture); the
            // residual stream, which F16 would overflow, stays F32. On aarch64, ggml
            // accumulates sequences under 64 tokens in F16. Run on the BF16 GGUF, both
            // architectures stay within cosine 1e-5 of the F32 SentenceTransformers reference
            // with flash attention on or off, and turning it on halves the compute buffer and
            // latency at 8k tokens (docs/architecture.md § "Long inputs and memory").
            cparams.flash_attn_type = ffi::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cparams.n_ctx = n;
            cparams.n_batch = n;
            cparams.n_ubatch = n;
            cparams.n_seq_max = 1;
            cparams.n_threads = n_threads;
            cparams.n_threads_batch = n_threads;

            let ctx = ffi::llama_init_from_model(model, cparams);
            if ctx.is_null() {
                ffi::llama_model_free(model);
                return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(
                    "failed to create llama.cpp context".into(),
                )));
            }
            // The context reports the pooling it will actually run.
            let pooling = ffi::llama_pooling_type(ctx);
            if pooling != ffi::LLAMA_POOLING_TYPE_MEAN {
                ffi::llama_free(ctx);
                ffi::llama_model_free(model);
                return Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "llama.cpp context pools with type {pooling}, not mean ({})",
                    ffi::LLAMA_POOLING_TYPE_MEAN
                ))));
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
        check_sequence_length(tokens.len(), self.n_ctx)?;

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
        // The buffer holds `n_embd_out` floats (the projected width, not the hidden width),
        // which `load` checked equals `raw_dim`. It stays valid until the next encode on `ctx`.
        let raw = std::slice::from_raw_parts(ptr, self.raw_dim).to_vec();
        ffi::llama_batch_free(batch);
        Ok(raw)
    }
}

impl EmbeddingBackend for LlamaBackend {
    fn raw_embedding_dimension(&self) -> usize {
        self.raw_dim
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, LTEmbedError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| LTEmbedError::Inference(InferenceError::MutexPoisoned))?;
        // Safety: the guard keeps the model alive and serializes FFI calls on it.
        unsafe { tokenize_with_vocab(ffi::llama_model_get_vocab(guard.model), text) }
            .map_err(LTEmbedError::Tokenization)
    }

    fn embed(
        &self,
        tokenized: &[Vec<u32>],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<BackendRunProfile>), LTEmbedError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| LTEmbedError::Inference(InferenceError::MutexPoisoned))?;
        let ctx = guard.ctx;

        let mut embeddings = Vec::with_capacity(tokenized.len());
        let mut run_ms = 0.0;
        // Sequences run independently (one `llama_encode` per input). True multi-sequence
        // batching in a single `llama_encode` is a future throughput optimization.
        for tokens in tokenized {
            let start = collect_profile.then(Instant::now);
            let raw = unsafe { self.embed_one(ctx, tokens) }?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RAW_EMBEDDING_DIMENSION;

    const RAW: i32 = RAW_EMBEDDING_DIMENSION as i32;
    /// EmbeddingGemma 2's hidden width, apart from its 768-wide projected output.
    const HIDDEN: i32 = 512;

    fn runtime_error(result: Result<(), LTEmbedError>) -> String {
        match result {
            Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => msg,
            other => panic!("expected ModelLoad(Runtime), got {other:?}"),
        }
    }

    /// EmbeddingGemma 2: hidden width 512, pooled output projected to 768.
    #[test]
    fn test_check_embedding_widths_accepts_projected_output() {
        check_embedding_widths(HIDDEN, RAW, RAW_EMBEDDING_DIMENSION).unwrap();
    }

    /// A model without an output projection pools its hidden width.
    #[test]
    fn test_check_embedding_widths_accepts_unprojected_output() {
        check_embedding_widths(RAW, RAW, RAW_EMBEDDING_DIMENSION).unwrap();
    }

    /// A bundle that names the hidden width as its raw dimension: `embed_one` would read 512
    /// of the 768 pooled floats, i.e. a truncated vector, not the projected embedding.
    #[test]
    fn test_check_embedding_widths_rejects_hidden_width_as_raw_dimension() {
        let msg = runtime_error(check_embedding_widths(HIDDEN, RAW, 512));
        assert!(msg.contains("n_embd_out 768"), "{msg}");
        assert!(msg.contains("raw_embedding_dimension 512"), "{msg}");
    }

    /// A pooled output narrower than the bundle's raw dimension: reading
    /// `raw_embedding_dimension` floats from the pooled buffer would overrun it.
    #[test]
    fn test_check_embedding_widths_rejects_narrow_output() {
        let msg = runtime_error(check_embedding_widths(RAW, HIDDEN, RAW_EMBEDDING_DIMENSION));
        assert!(msg.contains("n_embd_out 512"), "{msg}");
    }

    #[test]
    fn test_check_embedding_widths_rejects_invalid_widths() {
        runtime_error(check_embedding_widths(HIDDEN, -1, RAW_EMBEDDING_DIMENSION));
        let msg = runtime_error(check_embedding_widths(-1, RAW, RAW_EMBEDDING_DIMENSION));
        assert!(msg.contains("n_embd -1"), "{msg}");
        let msg = runtime_error(check_embedding_widths(0, RAW, RAW_EMBEDDING_DIMENSION));
        assert!(msg.contains("n_embd 0"), "{msg}");
        // `-1_i32 as usize` is `usize::MAX`; a cast before the comparison would accept this.
        runtime_error(check_embedding_widths(HIDDEN, -1, usize::MAX));
    }

    #[test]
    fn test_check_declared_pooling_accepts_mean() {
        check_declared_pooling("gemma-embedding2.pooling_type", Some("1")).unwrap();
    }

    #[test]
    fn test_check_declared_pooling_rejects_other_pooling() {
        // NONE, CLS, LAST, RANK, and values that are not a pooling type at all.
        for value in ["0", "2", "3", "4", "-1", "mean", "", " 1"] {
            let msg = runtime_error(check_declared_pooling("arch.pooling_type", Some(value)));
            assert!(
                msg.contains(&format!("arch.pooling_type is {value}")),
                "{msg}"
            );
        }
    }

    #[test]
    fn test_check_declared_pooling_rejects_missing_declaration() {
        let msg = runtime_error(check_declared_pooling("arch.pooling_type", None));
        assert!(msg.contains("GGUF has no arch.pooling_type"), "{msg}");
    }

    /// EmbeddingGemma 2's vocabulary: `<bos>` is 2 and `<eos>` 1.
    const BOS: ffi::llama_token = 2;
    const EOS: ffi::llama_token = 1;

    #[test]
    fn test_check_special_tokens_accepts_bos_and_eos() {
        check_special_tokens(&[2, 43, 2335, 1], BOS, EOS).unwrap();
    }

    /// A GGUF that adds no EOS (`tokenizer.ggml.add_eos_token = false`), no BOS, or neither.
    #[test]
    fn test_check_special_tokens_rejects_missing_bos_or_eos() {
        for (ids, missing) in [
            (&[2, 43, 2335][..], "adds no EOS"),
            (&[43, 2335, 1][..], "adds no BOS"),
            (&[43, 2335][..], "adds no BOS or EOS"),
            (&[][..], "adds no BOS or EOS"),
        ] {
            let msg = runtime_error(check_special_tokens(ids, BOS, EOS));
            assert!(msg.contains(missing), "{ids:?}: {msg}");
            assert!(msg.contains("BOS is 2 and EOS 1"), "{msg}");
        }
    }

    /// A lone token that is both BOS and EOS is not a text wrapped in both.
    #[test]
    fn test_check_special_tokens_needs_two_tokens_when_bos_is_eos() {
        let msg = runtime_error(check_special_tokens(&[7], 7, 7));
        assert!(msg.contains("adds no EOS"), "{msg}");
        check_special_tokens(&[7, 7], 7, 7).unwrap();
    }

    #[test]
    fn test_check_sequence_length_accepts_one_to_the_context_length() {
        for n_tokens in [1, 2, 8191, 8192] {
            check_sequence_length(n_tokens, 8192).unwrap();
        }
    }

    /// Never reached through the engine, which returns `InputTooLong` first; llama.cpp would
    /// abort on the long sequence.
    #[test]
    fn test_check_sequence_length_rejects_empty_and_overlong_sequences() {
        for n_tokens in [0, 8193] {
            match check_sequence_length(n_tokens, 8192) {
                Err(LTEmbedError::Inference(InferenceError::Internal(msg))) => {
                    assert!(msg.contains(&format!("{n_tokens} tokens")), "{msg}");
                    assert!(msg.contains("1..=8192"), "{msg}");
                }
                other => panic!("{n_tokens} tokens: expected Inference(Internal), got {other:?}"),
            }
        }
    }

    #[test]
    fn test_check_special_tokens_rejects_vocabulary_without_them() {
        let msg = runtime_error(check_special_tokens(&[2, 43, 1], -1, EOS));
        assert!(msg.contains("has no BOS token"), "{msg}");
        let msg = runtime_error(check_special_tokens(&[2, 43, 1], BOS, -1));
        assert!(msg.contains("has no EOS token"), "{msg}");
    }
}
