// Reloading a tokenizer must not leave memory behind. This is a separate test binary because
// it counts live heap bytes through a global allocator: keep it the only test in this file, or
// concurrently running tests will skew the count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ltembed::engine::MAX_LENGTH;
use ltembed::traits::tokenizer::HFTokenizer;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::pre_tokenizers::whitespace::Whitespace;

struct CountingAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            LIVE_BYTES.fetch_add(
                new_size as isize - layout.size() as isize,
                Ordering::Relaxed,
            );
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const LETTERS: &[u8; 16] = b"abcdefghijklmnop";

/// A minimal BPE tokenizer: single letters, one merge, whitespace pre-tokenization.
fn write_bpe_tokenizer(path: &Path) {
    let mut vocab: Vocab = LETTERS
        .iter()
        .zip(0..)
        .map(|(&c, id)| ((c as char).to_string(), id))
        .collect();
    vocab.insert("ab".to_string(), LETTERS.len() as u32);
    let merges = vec![("a".to_string(), "b".to_string())];
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(path, false).unwrap();
}

/// `n` distinct four-letter words, i.e. `n` distinct BPE cache keys.
fn distinct_words(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            (0..4)
                .map(|digit| LETTERS[(i >> (4 * digit)) & 0xf] as char)
                .collect()
        })
        .collect()
}

#[test]
fn test_reloading_tokenizer_does_not_accumulate_memory() {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("ltembed-bpe-tokenizer-{nanos}.json"));
    write_bpe_tokenizer(&path);
    // Below the 10k-entry per-thread cache capacity, so a cache would keep every word.
    let texts: Vec<String> = distinct_words(8_000)
        .chunks(100)
        .map(|words| words.join(" "))
        .collect();

    // `encode_batch` is the path `EmbeddingEngine` uses; it fans out over rayon's pool.
    let load_and_encode = || {
        let tokenizer = HFTokenizer::from_file(path.to_str().unwrap()).unwrap();
        let outputs = tokenizer.encode_batch(&texts, MAX_LENGTH).unwrap();
        assert_eq!(outputs.len(), texts.len());
    };

    // Warm-up absorbs one-off allocations: rayon's pool and per-thread state.
    for _ in 0..2 {
        load_and_encode();
    }
    let baseline = LIVE_BYTES.load(Ordering::Relaxed);
    const RELOADS: isize = 8;
    for _ in 0..RELOADS {
        load_and_encode();
    }
    let growth_per_reload = (LIVE_BYTES.load(Ordering::Relaxed) - baseline) / RELOADS;
    fs::remove_file(&path).unwrap();

    // A cache that outlives its tokenizer keeps all 8,000 words per reload (over 1 MB). What
    // is allowed is the ~100 B per thread that an empty per-thread cache map costs.
    assert!(
        growth_per_reload < 128 * 1024,
        "live heap grew by {growth_per_reload} B per tokenizer reload"
    );
}
