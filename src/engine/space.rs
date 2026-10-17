use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::{LTEmbedError, ModelLoadError};

use super::bundle::ModelSpec;
use super::config::EngineConfig;

/// Version of [`EmbeddingSpace`]'s string form. Bump it when the same model file and settings
/// would give different vectors, for example if the pooling, truncation or normalization
/// changes, or a llama.cpp release changes how the model tokenizes or pools, or when the form
/// itself changes, so that indexes tagged with the old form no longer match.
const STRING_FORM_VERSION: u32 = 1;

/// Bytes [`ModelFile::sha256`] reads at a time.
const HASH_CHUNK: usize = 1 << 16;

/// The vector space an [`EmbeddingEngine`](super::EmbeddingEngine) embeds into, from
/// [`EmbeddingEngine::embedding_space`](super::EmbeddingEngine::embedding_space).
///
/// Two engines whose spaces are equal return comparable vectors. Vectors from engines whose
/// spaces differ do not belong in one index: a different model file, prompt, width or
/// normalization gives vectors that are not comparable, even when they have the same length.
///
/// Every field is something the engine uses, not something its bundle claims. The model is
/// identified by the sha256 of the `model.gguf` it loaded, not by `build-info.json`'s
/// `target_id` or `quant` or by `provenance.json`, so a bundle whose metadata was edited
/// cannot report another model's space. Settings that change vectors only by floating-point
/// rounding, such as the thread count, the bundle's `max_length` or the llama.cpp release,
/// are not part of it (`docs/architecture.md` § "Embedding space").
///
/// Its string form ([`fmt::Display`]) is meant to be stored beside an index:
///
/// ```text
/// ltembed-space/1;model-sha256=<hex>;query-prefix=<prefix>;document-prefix=<prefix>;output-dimension=<n>;l2-normalize=<true|false>
/// ```
///
/// In the values, `%`, `;` and every byte outside printable ASCII are written as `%XX` (none
/// occurs in the hex hash), so the string is printable ASCII and two different spaces never
/// share one. The form is stable: LTEmbed changes it only together with the version after
/// `ltembed-space/`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EmbeddingSpace {
    /// The sha256 of the `model.gguf` the engine loaded, in lowercase hex: the value in the
    /// bundle's `SHA256SUMS`, and in `docs/bundle-format.md` for the pinned quants.
    pub model_sha256: String,
    /// The prefix the engine adds to every query: the bundle's `query_prefix`.
    pub query_prefix: String,
    /// The prefix the engine adds to every document: the bundle's `document_prefix`.
    pub document_prefix: String,
    /// The engine's [`EngineConfig::output_dimension`].
    pub output_dimension: usize,
    /// The engine's [`EngineConfig::l2_normalize`].
    pub l2_normalize: bool,
}

impl EmbeddingSpace {
    pub(crate) fn new(model_sha256: String, spec: &ModelSpec, config: EngineConfig) -> Self {
        Self {
            model_sha256,
            query_prefix: spec.query_prefix.clone(),
            document_prefix: spec.document_prefix.clone(),
            output_dimension: config.output_dimension,
            l2_normalize: config.l2_normalize,
        }
    }
}

impl fmt::Display for EmbeddingSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ltembed-space/{STRING_FORM_VERSION};model-sha256=")?;
        write_escaped(f, &self.model_sha256)?;
        f.write_str(";query-prefix=")?;
        write_escaped(f, &self.query_prefix)?;
        f.write_str(";document-prefix=")?;
        write_escaped(f, &self.document_prefix)?;
        write!(
            f,
            ";output-dimension={};l2-normalize={}",
            self.output_dimension, self.l2_normalize
        )
    }
}

/// Writes `value` with `%`, `;` and every byte outside printable ASCII as `%XX` (uppercase
/// hex): the string form's field separator never appears in a value, and an escaped value
/// reads back to exactly one original.
fn write_escaped(f: &mut fmt::Formatter<'_>, value: &str) -> fmt::Result {
    for byte in value.bytes() {
        if (b' '..=b'~').contains(&byte) && byte != b'%' && byte != b';' {
            f.write_char(char::from(byte))?;
        } else {
            write!(f, "%{byte:02X}")?;
        }
    }
    Ok(())
}

/// The bundle's `model.gguf`, opened before the backend loads it and kept open, so that its
/// hash is that of the file the backend loaded even if the path is replaced afterwards. The
/// backend opens the path itself, so [`ModelFile::check_still_at_path`] checks, once it has,
/// that the path did not change in between.
pub(crate) struct ModelFile {
    path: PathBuf,
    file: File,
}

impl ModelFile {
    pub(crate) fn open(path: &Path) -> Result<Self, LTEmbedError> {
        let file = File::open(path).map_err(|err| {
            LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                "failed to open GGUF model at {}: {err}",
                path.display()
            )))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Fails unless the path still names the file this handle opened (the same device and
    /// inode). Called once the backend has loaded the path: if another file was renamed over
    /// it, or it was removed, the backend may have read a file other than this one.
    pub(crate) fn check_still_at_path(&self) -> Result<(), LTEmbedError> {
        let runtime = |msg: String| LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg));
        let same_file = match (self.file.metadata(), fs::metadata(&self.path)) {
            (Ok(opened), Ok(now)) => (opened.dev(), opened.ino()) == (now.dev(), now.ino()),
            (Err(err), _) | (_, Err(err)) => {
                return Err(runtime(format!(
                    "failed to check that GGUF model at {} did not change while it loaded: {err}",
                    self.path.display()
                )));
            }
        };
        if !same_file {
            return Err(runtime(format!(
                "GGUF model at {} was replaced while it loaded; load it again",
                self.path.display()
            )));
        }
        Ok(())
    }

    /// The file's sha256 in lowercase hex. It reads at explicit offsets rather than through the
    /// file's cursor, so concurrent calls do not disturb each other.
    pub(crate) fn sha256(&self) -> Result<String, LTEmbedError> {
        let mut hasher = Sha256::new();
        let mut buf = vec![0; HASH_CHUNK];
        let mut offset = 0;
        loop {
            let n = match self.file.read_at(&mut buf, offset) {
                Ok(0) => break,
                Ok(n) => n,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(LTEmbedError::Io(io::Error::new(
                        err.kind(),
                        format!("failed to read {}: {err}", self.path.display()),
                    )));
                }
            };
            hasher.update(&buf[..n]);
            offset += n as u64;
        }
        Ok(hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    /// The sha256 of the release GGUF, `embeddinggemma-2-UD-Q5_K_XL.gguf`.
    const RELEASE_GGUF_SHA256: &str =
        "a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41";

    fn unique_temp_path(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ltembed-space-unit-tests-{nanos}-{counter}-{name}"))
    }

    /// The space of the release bundle at `EngineConfig::default()`.
    fn release_space() -> EmbeddingSpace {
        EmbeddingSpace::new(
            RELEASE_GGUF_SHA256.to_string(),
            &ModelSpec::defaults(),
            EngineConfig::default(),
        )
    }

    /// The string an index built with the release bundle stores. It must not change unless the
    /// vectors do, and then only with a new version.
    #[test]
    fn test_string_form_of_the_release_bundle() {
        assert_eq!(
            release_space().to_string(),
            "ltembed-space/1;\
             model-sha256=a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41;\
             query-prefix=task: search result | query: ;\
             document-prefix=title: none | text: ;\
             output-dimension=512;l2-normalize=true"
        );
    }

    /// Every field appears in the string form.
    #[test]
    fn test_string_form_changes_with_every_field() {
        let release = release_space();
        let mut changed = Vec::new();
        let mut space = release.clone();
        space.model_sha256 =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into();
        changed.push(space);
        let mut space = release.clone();
        space.query_prefix = "Query: ".into();
        changed.push(space);
        let mut space = release.clone();
        space.document_prefix = "Document: ".into();
        changed.push(space);
        let mut space = release.clone();
        space.output_dimension = 256;
        changed.push(space);
        let mut space = release.clone();
        space.l2_normalize = false;
        changed.push(space);
        for space in changed {
            assert_ne!(space, release);
            assert_ne!(space.to_string(), release.to_string(), "{space:?}");
        }
    }

    #[test]
    fn test_string_form_escapes_separators_and_non_ascii() {
        let mut space = release_space();
        space.query_prefix = "a;b%c=d\n".into();
        space.document_prefix = "标题: é ".into();
        let string = space.to_string();
        assert!(
            string.contains(";query-prefix=a%3Bb%25c=d%0A;document-prefix="),
            "{string}"
        );
        assert!(
            string.contains(";document-prefix=%E6%A0%87%E9%A2%98: %C3%A9 ;"),
            "{string}"
        );
        assert!(string.bytes().all(|byte| (b' '..=b'~').contains(&byte)));
        assert_eq!(string.split(';').count(), 6, "{string}");
    }

    /// Without escaping, a `;` in a prefix could make two different spaces print the same.
    #[test]
    fn test_string_form_keeps_different_prefixes_apart() {
        let mut joined = release_space();
        joined.query_prefix = "q;document-prefix=d".into();
        joined.document_prefix = String::new();
        let mut split = release_space();
        split.query_prefix = "q".into();
        split.document_prefix = "d".into();
        assert_ne!(joined.to_string(), split.to_string());
    }

    fn sha256_of_contents(contents: &[u8]) -> String {
        let path = unique_temp_path("model.gguf");
        fs::write(&path, contents).unwrap();
        let sha256 = ModelFile::open(&path).unwrap().sha256().unwrap();
        fs::remove_file(path).unwrap();
        sha256
    }

    /// FIPS 180-2 test vectors; the million `a`s span several reads.
    #[test]
    fn test_model_file_sha256_matches_known_digests() {
        assert_eq!(
            sha256_of_contents(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_of_contents(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let million = vec![b'a'; 1_000_000];
        assert!(million.len() > 3 * HASH_CHUNK);
        assert_eq!(
            sha256_of_contents(&million),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// The hash is of the file that was opened: replacing or removing the path afterwards does
    /// not change it.
    #[test]
    fn test_model_file_hashes_the_file_it_opened() {
        let path = unique_temp_path("model.gguf");
        fs::write(&path, b"abc").unwrap();
        let model_file = ModelFile::open(&path).unwrap();

        let replacement = unique_temp_path("replacement.gguf");
        fs::write(&replacement, b"another model").unwrap();
        fs::rename(&replacement, &path).unwrap();
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(model_file.sha256().unwrap(), abc);
        assert_ne!(ModelFile::open(&path).unwrap().sha256().unwrap(), abc);

        fs::remove_file(&path).unwrap();
        assert_eq!(model_file.sha256().unwrap(), abc);
    }

    /// The path must still name the opened file, through a symlink too: a symlink renamed over
    /// it, as a deployment swaps a model, or its removal fails the check.
    #[test]
    fn test_model_file_check_fails_once_the_path_names_another_file() {
        let target = unique_temp_path("model.gguf");
        fs::write(&target, b"abc").unwrap();
        let path = unique_temp_path("link.gguf");
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let model_file = ModelFile::open(&path).unwrap();
        model_file.check_still_at_path().unwrap();

        // The same contents in another file are still another file.
        let copy = unique_temp_path("copy.gguf");
        fs::write(&copy, b"abc").unwrap();
        let link = unique_temp_path("new-link.gguf");
        std::os::unix::fs::symlink(&copy, &link).unwrap();
        fs::rename(&link, &path).unwrap();
        match model_file.check_still_at_path() {
            Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
                assert!(msg.contains(&path.display().to_string()), "{msg}");
                assert!(msg.contains("replaced"), "{msg}");
            }
            other => panic!("expected ModelLoad(Runtime), got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
        match model_file.check_still_at_path() {
            Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
                assert!(msg.contains(&path.display().to_string()), "{msg}");
            }
            other => panic!("expected ModelLoad(Runtime), got {other:?}"),
        }
        fs::remove_file(target).unwrap();
        fs::remove_file(copy).unwrap();
    }

    #[test]
    fn test_model_file_errors_name_the_path() {
        let missing = unique_temp_path("missing.gguf");
        match ModelFile::open(&missing) {
            Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
                assert!(msg.contains(&missing.display().to_string()), "{msg}");
            }
            Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
            Ok(_) => panic!("opened a missing file"),
        }

        // A directory opens, but cannot be read.
        let dir = unique_temp_path("dir");
        fs::create_dir(&dir).unwrap();
        let result = ModelFile::open(&dir).unwrap().sha256();
        fs::remove_dir(&dir).unwrap();
        match result {
            Err(LTEmbedError::Io(err)) => {
                assert!(
                    err.to_string().contains(&dir.display().to_string()),
                    "{err}"
                );
            }
            other => panic!("expected Io, got {other:?}"),
        }
    }
}
