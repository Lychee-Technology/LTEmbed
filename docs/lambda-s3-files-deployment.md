# Lambda Deployment with S3 Files

Deploy LTEmbed as a Lambda ZIP with `libonnxruntime.so` in-package
and model weights served from an S3 Files mount.

## Architecture

| Component | Location | Rationale |
|-----------|----------|-----------|
| `bootstrap` | Lambda ZIP (`/var/task/`) | Lambda entry point |
| `libonnxruntime.so` | Lambda ZIP (`/var/task/lib/`) | Avoid network-loading a dynamic library |
| `model.ort` | S3 Files mount | Model weights are large; avoid ZIP limit |
| `tokenizer.json` | S3 Files mount | Co-locate with model for version consistency |
| `build-info.json` | S3 Files mount | Co-locate with model for version consistency |

## S3 Bucket Layout

```
s3://<bucket>/ltembed/<version>/ort_bundle/
  model.ort
  tokenizer.json
  build-info.json
  SHA256SUMS                       # optional integrity manifest
```

Use immutable version directories so bundle changes are explicit
configuration changes, not in-place overwrites.

## S3 Files Mount

Configure a Lambda file system mount pointed at the bundle prefix:

- **Local mount path**: `/mnt/s3files/ltembed/<version>/ort_bundle`
- **S3 ARN / prefix**: `arn:aws:s3:::<bucket>` with prefix scoped to
  `<version>/ort_bundle`

The Lambda execution role needs:
- `elasticfilesystem:ClientMount` (for the EFS-based mount)
- `s3:GetObject` on the bucket prefix (for S3 Files data plane)

## ZIP Package

```
bootstrap                    # compiled Rust binary
lib/
  libonnxruntime.so           # custom minimal-build .so
```

Supported by `OnnxEngine::from_bundle_dir_with_dylib(...)`:

```rust
use ltembed::engine::{EmbeddingInput, OnnxEngine, OnnxEngineConfig};
use std::path::PathBuf;
use std::sync::OnceLock;

struct AppState {
    engine: OnnxEngine,
}

static STATE: OnceLock<AppState> = OnceLock::new();

fn get_engine() -> &'static AppState {
    STATE.get_or_init(|| {
        let bundle_dir = std::env::var("LTEMBED_BUNDLE_DIR")
            .unwrap_or_else(|_| "/mnt/s3files/ltembed/bundle/ort_bundle".into());
        let dylib_path = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "/var/task/lib/libonnxruntime.so".into());
        let engine = OnnxEngine::from_bundle_dir_with_dylib(
            &bundle_dir,
            &dylib_path,
            OnnxEngineConfig {
                output_dimension: 512,
                l2_normalize: true,
            },
        )
        .expect("Failed to initialize OnnxEngine");
        AppState { engine }
    })
}
```

## Lambda Configuration

| Setting | Value |
|---------|-------|
| Runtime | `provided.al2023` (custom runtime) |
| Architecture | `arm64` |
| Memory | 512 MB (tune after profiling) |
| Timeout | 30 s (covers first cold start) |
| Ephemeral storage | 512 MB (default) |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `LD_LIBRARY_PATH` | `/var/task/lib` | OS dynamic linker |
| `ORT_DYLIB_PATH` | `/var/task/lib/libonnxruntime.so` | `ort` crate `dlopen` target |
| `LTEMBED_BUNDLE_DIR` | `/mnt/s3files/ltembed/<version>/ort_bundle` | model + tokenizer + metadata |

## Cold Start

Model data is read lazily by S3 Files on first access. The first
`OnnxEngine::from_bundle_dir_with_dylib(...)` call within a new Lambda
execution environment will:

1. Read `build-info.json` (small, fast)
2. Read `tokenizer.json` (small, fast)
3. Read `model.ort` (large, triggers S3 Files lazy load)

Subsequent invocations reuse the `OnceLock` singleton and cached
file data. Benchmark cold start against a fully local bundle to
establish the latency budget.

## Verification Checklist

- [ ] S3 Files mount is visible at `/mnt/s3files/...`
- [ ] `ls /mnt/s3files/.../ort_bundle/` shows `model.ort`, `tokenizer.json`, `build-info.json`
- [ ] Lambda execution role has mount and S3 read permissions
- [ ] `cargo test` passes locally with `LTEMBED_TEST_BUNDLE_DIR=/mnt/s3files/.../ort_bundle`
- [ ] Lambda test invocation returns embeddings with expected dimension
- [ ] Cold start latency measured and acceptable for the workload
