// ffi/cross_encoder.cpp — llama.cpp cross-encoder reranking implementation.
#include "cross_encoder.h"
#include "llama.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cstdio>

struct LtCrossEncoder {
    llama_model   *model;
    llama_context *ctx;
};

LtCrossEncoder *lt_ce_load(const char *model_path, int n_threads) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[lt_ce] failed to load model: %s\n", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.n_ctx           = 512;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[lt_ce] failed to create context\n");
        llama_model_free(model);
        return nullptr;
    }

    return new LtCrossEncoder{model, ctx};
}

float lt_ce_score(LtCrossEncoder *ce, const char *query, const char *document) {
    if (!ce || !query || !document) return NAN;

    std::string combined = std::string(query) + " [SEP] " + std::string(document);

    const llama_vocab *vocab = llama_model_get_vocab(ce->model);

    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(
        vocab, combined.c_str(), (int)combined.size(),
        tokens.data(), (int)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false
    );
    if (n_tokens <= 0) {
        fprintf(stderr, "[lt_ce] tokenization failed\n");
        return NAN;
    }
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ce->ctx), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ce->ctx, batch) != 0) {
        fprintf(stderr, "[lt_ce] decode failed\n");
        return NAN;
    }

    float *logits = llama_get_logits(ce->ctx);
    if (!logits) {
        fprintf(stderr, "[lt_ce] no logits available\n");
        return NAN;
    }

    return logits[0];
}

void lt_ce_free(LtCrossEncoder *ce) {
    if (!ce) return;
    llama_free(ce->ctx);
    llama_model_free(ce->model);
    delete ce;
}
