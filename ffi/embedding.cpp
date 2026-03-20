// ffi/embedding.cpp — llama.cpp embedding backend implementation.
#include "embedding.h"
#include "llama.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>

struct LtEmbedder {
    llama_model   *model;
    llama_context *ctx;
    int            dim;
};

LtEmbedder *lt_emb_load(const char *model_path, int n_threads) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only

    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[lt_emb] failed to load model: %s\n", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings      = true;
    cparams.n_ctx           = 512;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[lt_emb] failed to create context\n");
        llama_model_free(model);
        return nullptr;
    }

    int dim = llama_model_n_embd(model);
    return new LtEmbedder{model, ctx, dim};
}

int lt_emb_compute(LtEmbedder *emb, const char *text, float *out_buf, int buf_len) {
    if (!emb || !text || !out_buf || buf_len < emb->dim) return -1;

    const llama_vocab *vocab = llama_model_get_vocab(emb->model);

    // Tokenize
    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(
        vocab, text, (int)strlen(text),
        tokens.data(), (int)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false
    );
    if (n_tokens <= 0) {
        fprintf(stderr, "[lt_emb] tokenization failed or empty input\n");
        return -1;
    }
    tokens.resize(n_tokens);

    // Clear KV cache before each inference
    llama_memory_clear(llama_get_memory(emb->ctx), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(emb->ctx, batch) != 0) {
        fprintf(stderr, "[lt_emb] decode failed\n");
        return -1;
    }

    // Retrieve sequence-level embedding
    float *embd = llama_get_embeddings_seq(emb->ctx, 0);
    if (!embd) {
        embd = llama_get_embeddings_ith(emb->ctx, -1);
    }
    if (!embd) {
        fprintf(stderr, "[lt_emb] no embeddings available\n");
        return -1;
    }

    // Copy and L2-normalize
    memcpy(out_buf, embd, emb->dim * sizeof(float));
    float norm = 0.0f;
    for (int i = 0; i < emb->dim; ++i) norm += out_buf[i] * out_buf[i];
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (int i = 0; i < emb->dim; ++i) out_buf[i] /= norm;
    }
    return 0;
}

int lt_emb_dim(LtEmbedder *emb) {
    return emb ? emb->dim : -1;
}

void lt_emb_free(LtEmbedder *emb) {
    if (!emb) return;
    llama_free(emb->ctx);
    llama_model_free(emb->model);
    delete emb;
}
