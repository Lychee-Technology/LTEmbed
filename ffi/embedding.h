/* ffi/embedding.h — C ABI for llama.cpp-backed embedding inference.
 *
 * Only three types cross the boundary. No llama.cpp types are exposed.
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct LtEmbedder LtEmbedder;

/* Load a GGUF embedding model from disk.
 * Returns NULL on failure (logs error to stderr). */
LtEmbedder *lt_emb_load(const char *model_path, int n_threads);

/* Compute the L2-normalized embedding for `text`.
 * `out_buf` must be caller-allocated with at least `buf_len` floats.
 * Returns 0 on success, -1 on error. */
int lt_emb_compute(LtEmbedder *ctx, const char *text, float *out_buf, int buf_len);

/* Returns the embedding dimension of the loaded model. */
int lt_emb_dim(LtEmbedder *ctx);

void lt_emb_free(LtEmbedder *ctx);

#ifdef __cplusplus
}
#endif
