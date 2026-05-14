use std::error::Error;
use std::path::{Path, PathBuf};

use ltembed::engine::{EmbeddingInput, OnnxEngine, OnnxEngineConfig};

const BUNDLE_DIR: &str = "ort_bundle";
const TOKENIZER_FILE: &str = "tokenizer.json";
const BUILD_INFO_FILE: &str = "build-info.json";

fn require_file(path: &Path) -> Result<(), Box<dyn Error>> {
    if Path::new(path).exists() {
        return Ok(());
    }

    Err(format!("required asset missing: {}", path.display()).into())
}

fn find_bundle_dir(start_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    for candidate_root in start_dir.ancestors() {
        let bundle_dir = candidate_root.join(BUNDLE_DIR);
        let tokenizer_path = bundle_dir.join(TOKENIZER_FILE);
        let model_path = bundle_dir.join("model.ort");
        let build_info_path = bundle_dir.join(BUILD_INFO_FILE);
        if tokenizer_path.exists() && model_path.exists() && build_info_path.exists() {
            return Ok(bundle_dir);
        }
    }

    Err(format!(
        "required ort_bundle not found under '{}' or any ancestor",
        start_dir.display()
    )
    .into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let bundle_dir = find_bundle_dir(&std::env::current_dir()?)?;
    let model_path = bundle_dir.join("model.ort");

    require_file(&model_path)?;
    require_file(&bundle_dir.join(TOKENIZER_FILE))?;
    require_file(&bundle_dir.join(BUILD_INFO_FILE))?;

    let engine = OnnxEngine::from_bundle_dir(
        &bundle_dir,
        &model_path,
        OnnxEngineConfig {
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
        let repo_bundle = repo_root.join("ort_bundle");
        let worktree_dir = repo_root.join(".worktrees").join("branch");

        fs::create_dir_all(&repo_bundle).unwrap();
        fs::create_dir_all(&worktree_dir).unwrap();
        fs::write(repo_bundle.join("tokenizer.json"), "{}").unwrap();
        fs::write(repo_bundle.join("model.ort"), "stub").unwrap();
        fs::write(repo_bundle.join("build-info.json"), "{}").unwrap();

        let resolved = find_bundle_dir(&worktree_dir).unwrap();
        assert_eq!(resolved, repo_bundle);

        fs::remove_dir_all(temp_root).unwrap();
    }
}
