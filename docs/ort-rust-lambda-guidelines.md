# Building ONNX Runtime for Rust AWS Lambda (ARM64)

**Guidelines for production-grade embedding inference on AWS Lambda Graviton**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Decision](#2-architecture-decision)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Prepare the ONNX Model](#4-step-1--prepare-the-onnx-model)
5. [Step 2 — Build OnnxRuntime (Minimal, ARM64)](#5-step-2--build-onnxruntime-minimal-arm64)
6. [Step 3 — Rust Project Setup](#6-step-3--rust-project-setup)
7. [Step 4 — Docker Build Pipeline](#7-step-4--docker-build-pipeline)
8. [Step 5 — Lambda Deployment](#8-step-5--lambda-deployment)
9. [Embedding Inference Implementation](#9-embedding-inference-implementation)
10. [Size Budget Reference](#10-size-budget-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Known Limitations](#12-known-limitations)

---

## 1. Overview

This document describes how to build and deploy a Rust AWS Lambda function that performs text embedding inference using ONNX Runtime (ORT) on ARM64 (AWS Graviton). It covers the full pipeline from ORT compilation to Lambda ZIP deployment.

### Target Stack

| Component | Choice | Rationale |
|---|---|---|
| Runtime | AWS Lambda Custom OS (AL2023) | Smallest cold-start footprint |
| Architecture | `arm64` (Graviton3) | ~20% cheaper than x86, better performance-per-watt |
| ML Runtime | ONNX Runtime (via `ort` crate) | Best Rust integration, supports INT8 |
| Model | Any BERT-family ONNX model | e.g. `jina-embeddings-v5-text-nano-retrieval` |
| Linking | Dynamic (`.so`) | Smaller than static `.a` in practice |

---

## 2. Architecture Decision

### Why Not fastembed-rs / candle?

| Option | Status | Issue |
|---|---|---|
| `fastembed-rs` | ✅ Good for BERT family | Does not yet support EuroBERT / Qwen3-based models (2026-03) |
| `candle` (CPU FP32) | ⚠️ Limited | No optimized BLAS on Linux ARM64; poor performance without Accelerate |
| `ort` crate + ONNX | ✅ **Recommended** | Universal model support, INT8, ARM XNNPACK backend |

### Why Not Static Linking ORT?

Static linking (`libonnxruntime.a`) produces a **larger** artifact than dynamic linking (`.so`) because:

- The static archive includes all symbols with no lazy loading
- C++ LTO/LTCG settings inflate `.a` size significantly on some platforms
- Rust's own LTO **cannot cross the C FFI boundary** to eliminate dead C++ code

**Use dynamic linking + Minimal Build for the best size/performance tradeoff.**

---

## 3. Prerequisites

### Host Machine

- Docker (with `buildx` for cross-compilation support)
- Python 3.9+ (for ORT build scripts and model conversion utilities)

### Docker Images Used

| Stage | Image | Purpose |
|---|---|---|
| ORT build | `public.ecr.aws/amazonlinux/amazonlinux:2023` | Matches Lambda runtime |
| Rust build | Same AL2023 image | glibc compatibility |

### Lambda Configuration

- **Runtime**: `Amazon Linux 2023` (Custom OS / Bring Your Own Bootstrap)
- **Architecture**: `arm64`
- **Memory**: 512 MB (minimum recommended; 256 MB is possible for small models)
- **Timeout**: 30 seconds (covers cold-start model load + first inference)

---

## 4. Step 1 — Prepare the ONNX Model

### 4.1 Export or Download the ONNX Model

Many models on HuggingFace already provide ONNX weights in an `onnx/` subfolder.
For models that offer task-specific variants (e.g. jina-embeddings-v5), use the
pre-merged task variant directly — LoRA adapters are already fused into the weights.

```bash
# Example: download using huggingface-cli
pip install huggingface-hub
huggingface-cli download jinaai/jina-embeddings-v5-text-nano-retrieval \
  --include "onnx/model.onnx" \
  --local-dir ./model
```

For models without pre-exported ONNX, use `optimum`:

```bash
pip install optimum[exporters]
optimum-cli export onnx \
  --model jinaai/jina-embeddings-v5-text-nano-retrieval \
  --task feature-extraction \
  ./model/onnx/
```

### 4.2 Generate the Reduced Operator Config

This is the single most important size-reduction step. It tells ORT to compile
only the operators actually used by your model.

```bash
# Clone OnnxRuntime source (needed for the script)
git clone --depth 1 --branch v1.20.1 https://github.com/microsoft/onnxruntime.git

# Generate config from your ONNX model
python onnxruntime/tools/python/create_reduced_build_config.py \
  --format ONNX \
  ./model/onnx/model.onnx \
  ./model/

# This produces: ./model/model.required_operators.config
```

### 4.3 (Optional) Quantize to INT8

INT8 quantization reduces the model file size by ~4x with minimal accuracy loss
for embedding models. Use `onnxruntime.quantization`:

```bash
pip install onnxruntime
python - <<'EOF'
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="./model/onnx/model.onnx",
    model_output="./model/onnx/model_int8.onnx",
    weight_type=QuantType.QInt8,
)
EOF
```

> **Verification**: After quantization, confirm cosine similarity between FP32 and
> INT8 embeddings is > 0.99 on a representative sample of inputs. If it falls below
> this threshold, the model may not be suitable for dynamic quantization and a
> calibration dataset should be used instead.

### 4.4 Know Your Model's Pooling Strategy

Different embedding model families use different pooling approaches. Using the wrong
pooling type produces incorrect embeddings with no runtime error.

| Architecture | Pooling | Prefix Required |
|---|---|---|
| BERT / RoBERTa | `[CLS]` token (index 0) | None |
| e5-small-v2 | Mean pooling | `query:` / `passage:` |
| jina-embeddings-v5-nano | Last token (`attention_mask` guided) | `Query:` / `Document:` |
| Qwen3-Embedding | Last token | `Query:` (queries only) |

Verify the pooling strategy by reading the model card before implementation.

---

## 5. Step 2 — Build OnnxRuntime (Minimal, ARM64)

### 5.1 Why a Custom Build?

The pre-built `libonnxruntime.so` from Microsoft is ~80MB and includes all
operators, all execution providers, and full RTTI. For Lambda deployment
(250MB uncompressed limit), a custom Minimal Build is essential.

### 5.2 Build Flags Reference

| Flag | Effect | Required? |
|---|---|---|
| `--config MinSizeRel` | Optimize for size over speed | ✅ Yes |
| `--minimal_build` | Enable operator subsetting | ✅ Yes |
| `--include_ops_by_config` | Compile only ops used by your model | ✅ Yes |
| `--disable_ml_ops` | Exclude ONNX ML-domain operators | ✅ Yes (embedding models don't use them) |
| `--disable_exceptions` | Replace exceptions with `abort()` | ✅ Yes (pairs with `--disable_rtti`) |
| `--disable_rtti` | Remove C++ type info tables | ✅ Yes |
| `--enable_reduced_operator_type_support` | Further trim per-type kernel variants | ✅ Yes |
| `--skip_tests` | Skip test compilation | ✅ Yes |
| `--build_shared_lib` | Build `.so` instead of `.a` | ✅ Yes (`.a` is larger) |

> **Note on `--disable_rtti`**: Disabling RTTI is safe for CPU-only embedding
> inference. It removes C++ `typeinfo` structures and `dynamic_cast` support.
> These are only used in ORT for multi-EP selection and diagnostics — paths
> that are never exercised in a single-model CPU inference scenario.
> It **must** be paired with `--disable_exceptions`.

### 5.3 Additional Compiler Flags for Dead Code Elimination

While these flags cannot penetrate C++ virtual dispatch tables or static
registration patterns (limiting their overall effectiveness to ~5–10% additional
reduction on top of Minimal Build), they are still worth including:

```bash
-DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections"
-DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections"
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,--gc-sections,--strip-all"
```

> **Why limited?** OnnxRuntime's operator registration uses global constructor
> objects (`__attribute__((constructor))`). The linker treats these as having
> side effects and will **not** garbage-collect them even with `--gc-sections`.
> Source-level operator exclusion via `--include_ops_by_config` is the effective
> mechanism; linker-level GC is supplementary.

### 5.4 Dockerfile for ORT Minimal Build

```dockerfile
# Dockerfile.ort-build
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install build dependencies
RUN dnf install -y \
    gcc gcc-c++ cmake git python3 python3-pip \
    ninja-build patch tar gzip wget

# Clone ORT at a pinned release tag
ARG ORT_VERSION=v1.20.1
RUN git clone --depth 1 --branch ${ORT_VERSION} \
    https://github.com/microsoft/onnxruntime.git /ort

WORKDIR /ort

# Copy the reduced operator config generated in Step 4.2
COPY model/model.required_operators.config /ops.config

# Build OnnxRuntime Minimal for ARM64
RUN ./build.sh \
    --config MinSizeRel \
    --build_shared_lib \
    --minimal_build \
    --disable_ml_ops \
    --disable_exceptions \
    --disable_rtti \
    --include_ops_by_config /ops.config \
    --enable_reduced_operator_type_support \
    --skip_tests \
    --parallel \
    --cmake_extra_defines \
        "CMAKE_CXX_FLAGS=-ffunction-sections -fdata-sections" \
        "CMAKE_C_FLAGS=-ffunction-sections -fdata-sections" \
        "CMAKE_SHARED_LINKER_FLAGS=-Wl,--gc-sections,--strip-all"

# The built library is at:
# /ort/build/Linux/MinSizeRel/libonnxruntime.so
```

Build and extract:

```bash
docker build -t ort-builder -f Dockerfile.ort-build .
docker create --name ort-tmp ort-builder
docker cp ort-tmp:/ort/build/Linux/MinSizeRel/libonnxruntime.so ./lib/
docker rm ort-tmp
```

### 5.5 Expected Output Size

| Build Type | Approximate Size |
|---|---|
| Full build (pre-built) | ~80 MB |
| Minimal Build (all flags above) | **~25–45 MB** |

Exact size depends on the operator set required by your model. BERT-family models
use a relatively small number of operators, typically yielding sizes toward the
lower end of this range.

---

## 6. Step 3 — Rust Project Setup

### 6.1 Cargo.toml

```toml
[package]
name = "embedding-lambda"
version = "0.1.0"
edition = "2021"

[dependencies]
# Lambda runtime
lambda_runtime = "0.13"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# ONNX Runtime — load-dynamic avoids compile-time link dependency
ort = { version = "2", features = ["load-dynamic"] }

# Tokenizer (HuggingFace tokenizers format)
tokenizers = { version = "0.20", default-features = false, features = ["http"] }

# Numerical operations for pooling
ndarray = "0.16"

[profile.release]
lto = true            # Enables Rust-side LTO (intra-Rust dead code elimination)
opt-level = "z"       # Optimize for binary size
codegen-units = 1     # Required for full LTO
strip = true          # Strip debug symbols and symbol table
panic = "abort"       # Removes panic unwinding machinery (~50KB savings)
```

> **Note on `lto = true`**: Rust's LTO applies to Rust code only. It cannot
> eliminate dead code inside `libonnxruntime.so`. Its benefit here is reducing
> the Rust binary itself, not the ORT library.

### 6.2 Build Configuration (`.cargo/config.toml`)

```toml
[target.aarch64-unknown-linux-gnu]
# Tell the linker where libonnxruntime.so lives at link time
rustflags = [
    "-Clink-args=-Wl,-rpath,$ORIGIN/lib",
    "-Clink-args=-L./lib",
]

[env]
# Point ort crate to the pre-built library at runtime
ORT_DYLIB_PATH = { value = "./lib/libonnxruntime.so", relative = true }
```

### 6.3 Recommended Project Layout

```
embedding-lambda/
├── Cargo.toml
├── .cargo/
│   └── config.toml
├── src/
│   └── main.rs
├── model/
│   ├── onnx/
│   │   └── model_int8.onnx         # Quantized ONNX model
│   ├── tokenizer.json               # HuggingFace tokenizer
│   └── model.required_operators.config
├── lib/
│   └── libonnxruntime.so            # Built in Step 2 (not committed to git)
└── Dockerfile.build                 # Full build pipeline Dockerfile
```

---

## 7. Step 4 — Docker Build Pipeline

This Dockerfile builds the final Lambda deployment ZIP in a single container,
starting from the pre-built `libonnxruntime.so` extracted in Step 2.

```dockerfile
# Dockerfile.build
FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS builder

# Install Rust and build tools
RUN dnf install -y gcc gcc-c++ openssl-devel tar gzip zip findutils
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy project files
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY src ./src
COPY model ./model
COPY lib ./lib          # Pre-built libonnxruntime.so from Step 2

# Set ORT to find the pre-built library (load-dynamic mode)
ENV ORT_DYLIB_PATH=/build/lib/libonnxruntime.so

# Build in release mode targeting the Lambda runtime
RUN cargo build --release

# ── Package Stage ──────────────────────────────────────────────────────────────

WORKDIR /package

# Lambda requires the entry point to be named exactly 'bootstrap'
RUN cp /build/target/release/embedding-lambda ./bootstrap

# Bundle model and tokenizer
RUN cp -r /build/model ./model

# Bundle the ORT dynamic library
RUN mkdir -p lib && cp /build/lib/libonnxruntime.so ./lib/

# Create the deployment ZIP
# Directory structure inside ZIP:
#   bootstrap          <- Lambda entry point
#   model/onnx/        <- ONNX model files
#   model/tokenizer.*  <- Tokenizer files
#   lib/               <- Dynamic libraries
RUN zip -r /deploy.zip bootstrap model/ lib/

# Report final sizes for verification
RUN echo "=== Artifact sizes ===" && \
    ls -lh bootstrap lib/libonnxruntime.so && \
    echo "Total compressed:" && ls -lh /deploy.zip && \
    echo "Total uncompressed:" && unzip -l /deploy.zip | tail -1
```

### Build and Extract

```bash
# Build (on Apple Silicon Mac or Linux ARM64 host — no cross-compilation needed)
docker build -t lambda-builder -f Dockerfile.build .

# Extract the ZIP
docker create --name lambda-tmp lambda-builder
docker cp lambda-tmp:/deploy.zip ./deploy.zip
docker rm lambda-tmp

# Verify sizes
unzip -l deploy.zip
```

> **Cross-compilation note**: If building on an x86 host, add
> `--platform linux/arm64` to the `docker build` command. QEMU emulation will be
> used, significantly increasing build time (30+ minutes). Consider using
> GitHub Actions with an `ubuntu-24.04-arm` runner instead for faster builds.

---

## 8. Step 5 — Lambda Deployment

### 8.1 Lambda Function Configuration

| Setting | Value |
|---|---|
| Runtime | `Amazon Linux 2023` (Custom OS) |
| Architecture | `arm64` |
| Handler | `bootstrap` (irrelevant for custom runtime, but must be set) |
| Memory | 512 MB (adjust down after profiling) |
| Timeout | 30 seconds (covers cold-start) |
| Ephemeral storage | 512 MB (default) |

### 8.2 Required Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `LD_LIBRARY_PATH` | `/var/task/lib` | Allows the OS to find `libonnxruntime.so` at runtime |
| `ORT_DYLIB_PATH` | `/var/task/lib/libonnxruntime.so` | Tells the `ort` crate exactly where to load ORT from |

> **Why both?** `LD_LIBRARY_PATH` is used by the OS dynamic linker for any
> transitive `.so` dependencies ORT may load. `ORT_DYLIB_PATH` is read by the
> `ort` crate's `load-dynamic` feature to call `dlopen()` explicitly.

### 8.3 Upload and Verify

```bash
# Upload via AWS CLI
aws lambda update-function-code \
  --function-name my-embedding-fn \
  --zip-file fileb://deploy.zip \
  --architectures arm64

# Test cold-start behavior
aws lambda invoke \
  --function-name my-embedding-fn \
  --payload '{"texts": ["Query: hello world"]}' \
  --cli-binary-format raw-in-base64-out \
  response.json && cat response.json
```

---

## 9. Embedding Inference Implementation

### 9.1 Critical Implementation Details

**Singleton model initialization**: The model must be loaded exactly once per
Lambda instance, not once per invocation. Use `OnceLock` for lazy, thread-safe
initialization.

**Prefix handling**: Models using asymmetric retrieval (query vs. document)
require specific text prefixes. These must be prepended **before** tokenization,
not after. Failure to do so produces incorrect embeddings silently.

**Pooling strategy**: Must match the model's training configuration exactly.
jina-embeddings-v5 and Qwen3-based models use last-token pooling. BERT-family
models use `[CLS]` token or mean pooling. See Section 4.4.

### 9.2 Complete Implementation Example

```rust
// src/main.rs
use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use ndarray::Array2;
use ort::{inputs, Session};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

// ── Model singleton ────────────────────────────────────────────────────────────

struct EmbeddingModel {
    session: Session,
    tokenizer: Tokenizer,
}

static MODEL: OnceLock<EmbeddingModel> = OnceLock::new();

fn get_model() -> &'static EmbeddingModel {
    MODEL.get_or_init(|| {
        // Initialize ORT from the path set in ORT_DYLIB_PATH env var.
        // This must be called before any other ort usage.
        ort::init()
            .with_name("embedding")
            .commit()
            .expect("Failed to initialize ORT");

        let session = Session::builder()
            .expect("Failed to create session builder")
            .with_intra_threads(2)   // Tune for Lambda vCPU count
            .commit_from_file("/var/task/model/onnx/model_int8.onnx")
            .expect("Failed to load ONNX model");

        let tokenizer = Tokenizer::from_file("/var/task/model/tokenizer.json")
            .expect("Failed to load tokenizer");

        EmbeddingModel { session, tokenizer }
    })
}

// ── Pooling strategies ─────────────────────────────────────────────────────────

/// Last-token pooling: used by jina-v5, Qwen3-Embedding, and other
/// decoder-based embedding models. Takes the last non-padding token's
/// hidden state as the sequence representation.
fn last_token_pool(
    last_hidden_state: &ort::Value,
    attention_mask: &[i64],
) -> Vec<f32> {
    let tensor = last_hidden_state
        .try_extract_tensor::<f32>()
        .expect("Failed to extract hidden states");

    // Find index of the last non-padding token (last 1 in attention_mask)
    let last_token_idx = attention_mask
        .iter()
        .rposition(|&m| m == 1)
        .expect("Empty attention mask");

    // Extract the embedding vector at that position
    let shape = tensor.shape();
    let hidden_size = shape[2];
    let embedding: Vec<f32> = tensor
        .as_slice()
        .expect("Non-contiguous tensor")
        [last_token_idx * hidden_size..(last_token_idx + 1) * hidden_size]
        .to_vec();

    l2_normalize(embedding)
}

/// CLS token pooling: used by standard BERT and RoBERTa models.
/// Takes the first token ([CLS]) hidden state.
#[allow(dead_code)]
fn cls_token_pool(last_hidden_state: &ort::Value) -> Vec<f32> {
    let tensor = last_hidden_state
        .try_extract_tensor::<f32>()
        .expect("Failed to extract hidden states");
    let hidden_size = tensor.shape()[2];
    let embedding = tensor.as_slice().unwrap()[..hidden_size].to_vec();
    l2_normalize(embedding)
}

/// Mean pooling: used by e5-small-v2 and all-MiniLM family.
#[allow(dead_code)]
fn mean_pool(last_hidden_state: &ort::Value, attention_mask: &[i64]) -> Vec<f32> {
    let tensor = last_hidden_state
        .try_extract_tensor::<f32>()
        .expect("Failed to extract hidden states");
    let shape = tensor.shape(); // [batch=1, seq_len, hidden_size]
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let data = tensor.as_slice().unwrap();

    let mut sum = vec![0.0f32; hidden_size];
    let mut count = 0usize;

    for i in 0..seq_len {
        if attention_mask[i] == 1 {
            for j in 0..hidden_size {
                sum[j] += data[i * hidden_size + j];
            }
            count += 1;
        }
    }

    let mean: Vec<f32> = sum.iter().map(|&s| s / count as f32).collect();
    l2_normalize(mean)
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

// ── Core inference ─────────────────────────────────────────────────────────────

fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
    let model = get_model();

    // Tokenize batch
    // truncation=true and padding=true handle variable-length inputs.
    // Max length should match the model's training context window.
    let encodings = model
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| Error::from(format!("Tokenization failed: {e}")))?;

    let batch_size = encodings.len();
    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

    // Build input tensors (batch_size × seq_len)
    let mut input_ids = vec![0i64; batch_size * max_len];
    let mut attention_mask = vec![0i64; batch_size * max_len];
    let mut token_type_ids = vec![0i64; batch_size * max_len]; // zeros for BERT

    for (i, enc) in encodings.iter().enumerate() {
        for (j, (&id, &mask)) in enc
            .get_ids()
            .iter()
            .zip(enc.get_attention_mask().iter())
            .enumerate()
        {
            input_ids[i * max_len + j] = id as i64;
            attention_mask[i * max_len + j] = mask as i64;
        }
    }

    let shape = [batch_size, max_len];
    let input_ids_t = Array2::from_shape_vec(shape, input_ids)?;
    let attention_mask_t = Array2::from_shape_vec(shape, attention_mask.clone())?;
    let token_type_ids_t = Array2::from_shape_vec(shape, token_type_ids)?;

    // Run inference
    // Input names must match the ONNX model's input node names.
    // Inspect with: python -c "import onnx; m=onnx.load('model.onnx');
    //   print([n.name for n in m.graph.input])"
    let outputs = model.session.run(inputs![
        "input_ids"      => input_ids_t.view(),
        "attention_mask" => attention_mask_t.view(),
        "token_type_ids" => token_type_ids_t.view(),   // omit if not in model inputs
    ]?)?;

    let last_hidden_state = &outputs["last_hidden_state"];

    // Apply pooling per-sample
    // ⚠️ Change pooling function here based on your model (see Section 4.4)
    let embeddings: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| {
            let mask_slice = &attention_mask[i * max_len..(i + 1) * max_len];
            last_token_pool(last_hidden_state, mask_slice)
        })
        .collect();

    Ok(embeddings)
}

// ── Lambda handler ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Request {
    texts: Vec<String>,
    #[serde(default)]
    prefix: Option<String>,   // e.g. "Query: " or "Document: "
}

#[derive(Serialize)]
struct Response {
    embeddings: Vec<Vec<f32>>,
    model: &'static str,
    dimension: usize,
}

async fn handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let req = event.payload;

    // Apply prefix if provided (required for asymmetric retrieval models)
    let texts: Vec<String> = match &req.prefix {
        Some(p) => req.texts.iter().map(|t| format!("{p}{t}")).collect(),
        None => req.texts,
    };

    let embeddings = embed_texts(&texts)?;
    let dimension = embeddings.first().map(|e| e.len()).unwrap_or(0);

    Ok(Response {
        embeddings,
        model: "jina-embeddings-v5-text-nano-retrieval",
        dimension,
    })
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Trigger model loading on cold start (before any invocation)
    // This moves the load time out of the first request's latency.
    let _ = get_model();

    run(service_fn(handler)).await
}
```

---

## 10. Size Budget Reference

### Lambda Deployment Limits

| Limit | Value |
|---|---|
| ZIP file (upload) | 50 MB |
| Uncompressed deployment package | **250 MB** |
| Container image | 10 GB (use if ZIP limit is exceeded) |

### Typical Component Sizes

| Component | FP32 | INT8 | Notes |
|---|---|---|---|
| Model (jina-v5-nano) | ~430 MB | **~115 MB** | INT8 required for ZIP deployment |
| `libonnxruntime.so` (full) | ~80 MB | — | Pre-built Microsoft binary |
| `libonnxruntime.so` (minimal) | **~25–45 MB** | — | Custom build, Step 2 |
| Rust binary (`bootstrap`) | ~5–15 MB | — | After `strip = true` |
| Tokenizer files | ~2 MB | — | |
| **Total (INT8 + Minimal ORT)** | | **~150–175 MB** | ✅ Within 250 MB limit |
| **Total (FP32 + Minimal ORT)** | | **~465+ MB** | ❌ Exceeds limit → use container |

### If ZIP Limit is Exceeded

Switch to container image deployment:

```dockerfile
FROM public.ecr.aws/amazonlinux/amazonlinux:2023
COPY bootstrap /var/task/bootstrap
COPY model /var/task/model
COPY lib /var/task/lib
ENV LD_LIBRARY_PATH=/var/task/lib
ENV ORT_DYLIB_PATH=/var/task/lib/libonnxruntime.so
ENTRYPOINT ["/var/task/bootstrap"]
```

---

## 11. Troubleshooting

### `libonnxruntime.so: cannot open shared object file`

The Lambda runtime cannot find the library. Verify:
1. `LD_LIBRARY_PATH=/var/task/lib` is set in Lambda environment variables
2. `lib/libonnxruntime.so` is present in the ZIP at the path `lib/libonnxruntime.so`
3. The `.so` file is a real file, not a symlink (use `cp -L` when copying)

```bash
# Verify symlink resolution before packaging
ls -la lib/
# Should show: -rwxr-xr-x  libonnxruntime.so  (not a -> symlink)
```

### `OrtStatus: Failed to load model`

1. Verify the ONNX model path inside Lambda is `/var/task/model/onnx/model_int8.onnx`
2. Lambda's working directory is not `/var/task`; use absolute paths
3. Confirm the model was not corrupted during ZIP packaging:
   ```bash
   python -c "import onnx; onnx.checker.check_model('model_int8.onnx'); print('OK')"
   ```

### `Tokenization failed: No such file or directory`

Same root cause as above — use absolute paths for all model artifacts:
```rust
Tokenizer::from_file("/var/task/model/tokenizer.json")
```

### Cold Start Exceeds Timeout

1. Increase Lambda timeout to 60 seconds temporarily to measure actual cold-start time
2. Enable **Provisioned Concurrency** to keep instances warm (adds fixed cost)
3. Profile which operation is slowest:
   - `ORT init` (~100ms)
   - Model load from disk into RAM (~500ms–2s depending on model size)
   - First inference JIT compilation (~200ms for ORT graph optimization)

### Embeddings Are Incorrect (Wrong Similarity Scores)

Most likely cause: wrong pooling strategy. Verify by comparing output against the
Python reference implementation:

```python
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v5-text-nano-retrieval")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v5-text-nano-retrieval")
inputs = tokenizer(["Query: hello"], return_tensors="pt")
with torch.no_grad():
    output = model(**inputs)
# For last-token pooling:
seq_len = inputs["attention_mask"].sum(dim=1) - 1
ref_embedding = output.last_hidden_state[0, seq_len[0], :]
ref_embedding = F.normalize(ref_embedding, dim=-1)
print(ref_embedding[:5])  # Compare first 5 values with your Rust output
```

Cosine similarity between Rust and Python output should be > 0.9999 for correct
implementation.

---

## 12. Known Limitations

### ORT Minimal Build

- `--disable_exceptions` means ORT errors call `abort()` instead of returning an
  error code. A malformed input that would normally return an error will crash
  the Lambda instance. Validate inputs before passing to ORT.

- `--disable_rtti` disables `dynamic_cast` and `typeid`. This is safe for CPU
  inference but means detailed type-level error messages will show mangled C++
  symbol names rather than human-readable class names.

- The `--minimal_build` flag edits ORT source files during compilation. If you
  want to rebuild for a different model, regenerate the operator config and
  rebuild from scratch:
  ```bash
  cd onnxruntime && git checkout HEAD -- ./onnxruntime/core/providers
  ```

### Lambda Constraints

- `/tmp` is the only writable directory (512 MB by default). If you need to
  download the model at runtime rather than bundling it, write to `/tmp`.
  Note that `/tmp` contents **persist across warm invocations** but are lost
  on cold starts.

- ARM64 Lambda does not support AVX2/AVX-512 SIMD. ORT will use NEON intrinsics
  on ARM64, which are well-optimized for matrix operations.

### License

`jina-embeddings-v5-text-nano` is licensed under **CC BY-NC 4.0**. For
commercial use, contact Jina AI. Verify the license of any model you deploy
before production use.
