use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use ltembed::engine::ZeroVecEngine;
use ltembed::traits::pooling::MeanPooling;

const ASSETS_DIR: &str = "assets";
const CONFIG_FILE: &str = "config.json";
const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";

fn require_file(path: &Path) -> Result<(), Box<dyn Error>> {
    if Path::new(path).exists() {
        return Ok(());
    }

    Err(format!("required asset missing: {}", path.display()).into())
}

fn find_assets_dir(start_dir: &Path) -> Result<PathBuf, Box<dyn Error>> {
    for candidate_root in start_dir.ancestors() {
        let assets_dir = candidate_root.join(ASSETS_DIR);
        let config_path = assets_dir.join(CONFIG_FILE);
        let model_path = assets_dir.join(MODEL_FILE);
        let tokenizer_path = assets_dir.join(TOKENIZER_FILE);
        if config_path.exists() && model_path.exists() && tokenizer_path.exists() {
            return Ok(assets_dir);
        }
    }

    Err(format!(
        "required assets not found under '{}' or any ancestor",
        start_dir.display()
    )
    .into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let assets_dir = find_assets_dir(&std::env::current_dir()?)?;
    let config_path = assets_dir.join(CONFIG_FILE);
    let model_path = assets_dir.join(MODEL_FILE);
    let tokenizer_path = assets_dir.join(TOKENIZER_FILE);

    require_file(&config_path)?;
    require_file(&model_path)?;
    require_file(&tokenizer_path)?;

    let config_json = fs::read_to_string(&config_path)?;
    let engine = ZeroVecEngine::new(
        &model_path.to_string_lossy(),
        &config_json,
        &tokenizer_path.to_string_lossy(),
        Box::new(MeanPooling),
    )?;

    let inputs = ["query: Hello, world!", "query: LTEmbed Rust API example"];
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
    use super::find_assets_dir;
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
    fn test_find_assets_dir_falls_back_to_repo_root_for_worktree_layout() {
        let temp_root = unique_temp_dir();
        let repo_root = temp_root.join("repo");
        let repo_assets = repo_root.join("assets");
        let worktree_dir = repo_root.join(".worktrees").join("branch");

        fs::create_dir_all(&repo_assets).unwrap();
        fs::create_dir_all(&worktree_dir).unwrap();
        fs::write(repo_assets.join("config.json"), "{}").unwrap();
        fs::write(repo_assets.join("tokenizer.json"), "{}").unwrap();
        fs::write(repo_assets.join("model.safetensors"), "stub").unwrap();

        let resolved = find_assets_dir(&worktree_dir).unwrap();
        assert_eq!(resolved, repo_assets);

        fs::remove_dir_all(temp_root).unwrap();
    }
}
