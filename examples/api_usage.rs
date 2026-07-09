use std::error::Error;
use std::path::{Path, PathBuf};

use ltembed::engine::{EmbeddingEngine, EmbeddingInput, EngineConfig};

const BUNDLE_DIR: &str = "gguf_bundle";
const MODEL_FILE: &str = "model.gguf";
const TOKENIZER_FILE: &str = "tokenizer.json";
const BUILD_INFO_FILE: &str = "build-info.json";

fn find_bundle_dir(start_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    for candidate_root in start_dir.ancestors() {
        let bundle_dir = candidate_root.join(BUNDLE_DIR);
        if bundle_dir.join(MODEL_FILE).exists()
            && bundle_dir.join(TOKENIZER_FILE).exists()
            && bundle_dir.join(BUILD_INFO_FILE).exists()
        {
            return Ok(bundle_dir);
        }
    }

    Err(format!(
        "required {BUNDLE_DIR} (model.gguf + tokenizer.json + build-info.json) not found under \
         '{}' or any ancestor",
        start_dir.display()
    )
    .into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let bundle_dir = find_bundle_dir(&std::env::current_dir()?)?;

    let engine = EmbeddingEngine::from_gguf_bundle_dir(
        &bundle_dir,
        EngineConfig {
            output_dimension: 512,
            l2_normalize: true,
        },
    )?;

    let inputs = [
        EmbeddingInput::query("Hello, world!"),
        EmbeddingInput::query("LTEmbed Rust API example"),
    ];
    let embeddings = engine.embed_batch(&inputs)?;
    let first = embeddings
        .first()
        .ok_or("expected at least one embedding from embed_batch")?;
    let preview: Vec<String> = first
        .iter()
        .take(5)
        .map(|value| format!("{value:.6}"))
        .collect();

    println!("inputs: {}", embeddings.len());
    println!("embedding_dim: {}", first.len());
    println!("first_embedding_head: [{}]", preview.join(", "));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::find_bundle_dir;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ltembed-api-usage-{nanos}"))
    }

    #[test]
    fn test_find_bundle_dir_falls_back_to_repo_root_for_worktree_layout() {
        let temp_root = unique_temp_dir();
        let repo_root = temp_root.join("repo");
        let repo_bundle = repo_root.join("gguf_bundle");
        let worktree_dir = repo_root.join(".worktrees").join("branch");

        fs::create_dir_all(&repo_bundle).unwrap();
        fs::create_dir_all(&worktree_dir).unwrap();
        fs::write(repo_bundle.join("tokenizer.json"), "{}").unwrap();
        fs::write(repo_bundle.join("model.gguf"), "stub").unwrap();
        fs::write(repo_bundle.join("build-info.json"), "{}").unwrap();

        let resolved = find_bundle_dir(&worktree_dir).unwrap();
        assert_eq!(resolved, repo_bundle);

        fs::remove_dir_all(temp_root).unwrap();
    }
}
