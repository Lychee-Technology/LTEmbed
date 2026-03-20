/* ffi/cross_encoder.h — C ABI for llama.cpp-backed cross-encoder reranking.
 *
 * No llama.cpp types are exposed through this header.
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct LtCrossEncoder LtCrossEncoder;

/* Load a GGUF cross-encoder model. Returns NULL on failure. */
LtCrossEncoder *lt_ce_load(const char *model_path, int n_threads);

/* Score a (query, document) pair.
 * Returns the raw relevance logit. Apply sigmoid for a [0,1] probability.
 * Returns NAN on error. */
float lt_ce_score(LtCrossEncoder *ctx, const char *query, const char *document);

void lt_ce_free(LtCrossEncoder *ctx);

#ifdef __cplusplus
}
#endif
