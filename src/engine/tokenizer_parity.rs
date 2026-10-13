//! Token-id parity between the bundle's two tokenizers: `tokenizer.json`, through the
//! `tokenizers` crate (what the engine tokenizes inference inputs with), and the GGUF's own
//! vocabulary, through `llama_tokenize` (what upstream llama.cpp tokenizes with, and what the
//! load-time check compares against on one text).
//!
//! The corpus is fixed and deterministic: every sentence of `tests/CN_EN_Data.csv`, every
//! input of the golden, token-id, benchmark and retrieval-eval files, the hand-picked cases
//! below, and seeded random strings over many scripts, whitespace characters and
//! special-token fragments. Each text is tokenized as is and with the query and document
//! prefixes, as inference sees it. Both tokenizers must give exactly the same ids for every
//! one; there is no tolerance.
//!
//! A pass shows agreement on this corpus for the pinned `tokenizer.json`, GGUF, `tokenizers`
//! version and static llama.cpp release, not on every possible input
//! (`docs/testing.md` § "Tokenizer parity").
//!
//! The test needs the GGUF, so it is `#[ignore]`d to keep `cargo test --lib` model-free. Run
//! it with a bundle:
//!
//! ```text
//! LTEMBED_REQUIRE_TEST_BUNDLE=1 LTEMBED_TEST_BUNDLE_DIR="$PWD/gguf_bundle" \
//!   cargo test --lib tokenizer_parity -- --ignored
//! ```
//!
//! Texts that start with `<bos>` or end with `<eos>` make llama.cpp print a
//! `check_double_bos_eos` warning; both tokenizers then add a second one, and the warning is
//! expected.

use std::env::VarError;
use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use super::bundle::ModelSpec;
use super::inference::prefixed_text;
use super::llama::LlamaBackend;
use super::{EmbeddingInput, RAW_EMBEDDING_DIMENSION};
use crate::traits::tokenizer::{HFTokenizer, Tokenizer};

/// Same variables and semantics as the bundle-gated tests in `tests/integration_tests.rs`.
const TEST_BUNDLE_ENV: &str = "LTEMBED_TEST_BUNDLE_DIR";
const REQUIRE_TEST_BUNDLE_ENV: &str = "LTEMBED_REQUIRE_TEST_BUNDLE";

const CN_EN_SENTENCES: &str = "tests/CN_EN_Data.csv";
const GOLDEN_FIXTURE: &str = "tests/fixtures/test_fixtures.json";
const TOKEN_IDS_FIXTURE: &str = "tests/fixtures/token_ids.json";
const BENCHMARK_CORPUS: &str = "scripts/benchmark_corpus.json";
const RETRIEVAL_EVAL_CASES: &str = "scripts/retrieval_eval_cases.json";

const RANDOM_TEXTS: usize = 20_000;
const RANDOM_SEED: u64 = 0x5EED_1234_ABCD_0001;

/// Inputs where two tokenizer implementations are most likely to part ways: whitespace and
/// invisible characters, Unicode normalization, scripts without spaces, emoji sequences,
/// digits, code, and text that looks like special tokens.
const EDGE_CASES: &[&str] = &[
    "",
    " ",
    "   leading spaces",
    "trailing spaces   ",
    "\t\ttabs\tinside\t",
    "\n",
    "\r\n\r\nCRLF lines\r\n",
    "line\u{85}next line",
    "line\u{2028}separator\u{2029}paragraph",
    "\u{a0}no-break\u{a0}space",
    "\u{3000}全角空格\u{3000}",
    "zero\u{200b}width\u{200c}non\u{200d}joiner",
    "\u{feff}byte order mark",
    "left\u{200e}to right\u{200f}marks",
    "nul\0inside",
    "\u{7}\u{1b}[31mcontrol characters\u{1b}[0m",
    "caf\u{e9} (composed)",
    "cafe\u{301} (decomposed)",
    "\u{212b}ngstr\u{f6}m and \u{c5}ngstr\u{f6}m",
    "\u{fb01}nancial ligature",
    "①②③ ⅫⅣ ㎏㎞",
    "ｆｕｌｌｗｉｄｔｈ　ＡＢＣ　１２３",
    "한국어 and \u{1112}\u{1161}\u{11ab}\u{1100}\u{116e}\u{11a8}\u{110b}\u{1165}",
    "ภาษาไทยไม่มีช่องว่างระหว่างคำ",
    "हिन्दी में लिखा गया वाक्य।",
    "اللُّغَةُ العَرَبِيَّةُ جميلة",
    "עברית עם ניקוד: שָׁלוֹם",
    "Ελληνικά: ἀρχή",
    "Tiếng Việt có dấu",
    "ქართული ენა",
    "አማርኛ ቋንቋ",
    "ភាសាខ្មែរ",
    "မြန်မာဘာသာ",
    "カタカナとひらがなと漢字が混ざった文",
    "𠀀𠀁𪚥 rare CJK",
    "👨‍👩‍👧‍👦 family, 🏳️‍🌈 flag, 🇨🇳🇺🇸 flags, 1️⃣ keycap, 👍🏽 skin tone",
    "1234567890123456789012345678901234567890",
    "3.14159265358979 1,000,000.00 -0.0e+10 0x7FFF_FFFF",
    "١٢٣٤٥ ०१२३४ 一二三四五六七八九十",
    "fn main() {\n    println!(\"hi {}\", 1 + 2);\n}\n",
    "def f(x):\n\treturn x ** 2  # comment\n",
    "<div class=\"a\">&amp; &lt;tag&gt;</div>",
    "SELECT * FROM t WHERE a = 'x' AND b <> 2;",
    "{\"key\": [1, 2.5, {\"n\": null, \"t\": true}]}",
    "https://example.com/a/b?q=1&r=%20#frag user@example.com C:\\Windows\\System32",
    "$ ls -la | grep '^d' > out.txt 2>&1",
    "# Title\n\n- item **bold** _it_ `code`\n\n> quote",
    "<bos>",
    "<eos>",
    "<pad><unk><mask>",
    "</s> and <s>",
    "<|tool_response>",
    "<|turn>user\nhello<turn|>",
    "<<bos>> <bos <eos >eos> < bos >",
    "<BOS> <Eos> upper and mixed case",
    "text<eos>more text<bos>",
    "<|image|><|image|>",
    "<start_of_turn>user\nhi<end_of_turn>",
    "task: search result | query: already prefixed",
    "title: none | text: a document prefix inside the text",
];

/// The test bundle directory if it holds `model.gguf` and `tokenizer.json`. Otherwise the
/// test returns early, unless `LTEMBED_REQUIRE_TEST_BUNDLE=1`, which makes this panic. Mirrors
/// `test_bundle` in `tests/integration_tests.rs`.
fn test_bundle() -> Option<PathBuf> {
    let required = match std::env::var(REQUIRE_TEST_BUNDLE_ENV).as_deref() {
        Err(VarError::NotPresent) | Ok("" | "0") => false,
        Ok("1") => true,
        Ok(value) => panic!("{REQUIRE_TEST_BUNDLE_ENV} must be 0 or 1, got {value:?}"),
        Err(e) => panic!("{REQUIRE_TEST_BUNDLE_ENV}: {e}"),
    };
    let reason = match std::env::var_os(TEST_BUNDLE_ENV).map(PathBuf::from) {
        None => format!("{TEST_BUNDLE_ENV} is not set"),
        Some(dir) => {
            let missing: Vec<&str> = ["model.gguf", "tokenizer.json"]
                .into_iter()
                .filter(|file| !dir.join(file).exists())
                .collect();
            if missing.is_empty() {
                return Some(dir);
            }
            format!("{} has no {}", dir.display(), missing.join(" or "))
        }
    };
    if required {
        panic!(
            "{REQUIRE_TEST_BUNDLE_ENV}=1 requires the test bundle, but {reason} \
             (see docs/testing.md)"
        );
    }
    eprintln!("Skipping: {reason}");
    None
}

/// xorshift64*: enough to spread the random texts deterministically, without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Code point ranges the random texts draw from. ASCII and CJK are listed more than once to
/// weight them like real text.
const RANDOM_RANGES: &[(u32, u32)] = &[
    (0x20, 0x7E),     // ASCII
    (0x20, 0x7E),     //
    (0x20, 0x7E),     //
    (0x00, 0x1F),     // C0 controls
    (0xA0, 0xFF),     // Latin-1
    (0x100, 0x24F),   // Latin Extended
    (0x300, 0x36F),   // combining marks
    (0x370, 0x3FF),   // Greek
    (0x400, 0x4FF),   // Cyrillic
    (0x590, 0x5FF),   // Hebrew
    (0x600, 0x6FF),   // Arabic
    (0x900, 0x97F),   // Devanagari
    (0xE00, 0xE7F),   // Thai
    (0x1100, 0x11FF), // Hangul Jamo
    (0x2000, 0x206F), // general punctuation, invisible spaces and marks
    (0x2200, 0x22FF), // math operators
    (0x2500, 0x257F), // box drawing
    (0x3000, 0x303F), // CJK punctuation
    (0x3040, 0x30FF), // hiragana, katakana
    (0x4E00, 0x9FFF), // CJK ideographs
    (0x4E00, 0x9FFF), //
    (0xAC00, 0xD7A3), // Hangul syllables
    (0xE000, 0xF8FF), // private use
    (0xFE00, 0xFE0F), // variation selectors
    (0xFF01, 0xFF5E), // fullwidth ASCII
    (0x10000, 0x1007F),
    (0x13000, 0x1342F), // Egyptian hieroglyphs
    (0x1F1E6, 0x1F1FF), // regional indicators
    (0x1F300, 0x1FAFF), // emoji
    (0x20000, 0x2A6DF), // CJK Extension B
];

/// Fragments spliced into random texts whole: special-token text and its pieces, and
/// whitespace that the code point ranges rarely produce.
const RANDOM_FRAGMENTS: &[&str] = &[
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<mask>",
    "</s>",
    "<s>",
    "<|tool_response>",
    "<|turn>",
    "<turn|>",
    "<|image|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<",
    ">",
    "|",
    "<|",
    "|>",
    "\u{200d}",
    "\u{fe0f}",
    "\u{feff}",
    "\u{a0}",
    "\u{3000}",
    "\n\n",
    "\r\n",
    "\t",
    "    ",
    " ",
];

/// One random text: 1 to 12 segments, each either a fragment or 1 to 10 code points from one
/// range.
fn random_text(rng: &mut Rng) -> String {
    let mut text = String::new();
    for _ in 0..1 + rng.below(12) {
        if rng.below(5) == 0 {
            text.push_str(RANDOM_FRAGMENTS[rng.below(RANDOM_FRAGMENTS.len())]);
            continue;
        }
        let (lo, hi) = RANDOM_RANGES[rng.below(RANDOM_RANGES.len())];
        for _ in 0..1 + rng.below(10) {
            let code_point = lo + rng.below((hi - lo + 1) as usize) as u32;
            text.push(char::from_u32(code_point).expect("the ranges hold no surrogates"));
        }
    }
    text
}

fn random_texts(count: usize, seed: u64) -> Vec<String> {
    let mut rng = Rng(seed);
    (0..count).map(|_| random_text(&mut rng)).collect()
}

/// Splits one CSV line into fields, honouring double quotes and `""` escapes.
fn csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    fields.push(field);
    fields
}

fn read_json(path: &str) -> Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Every string stored under a `text` key, at any depth.
fn text_fields(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, item) in map {
                match (key.as_str(), item) {
                    ("text", Value::String(text)) => out.push(text.clone()),
                    _ => text_fields(item, out),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| text_fields(item, out)),
        _ => {}
    }
}

/// The corpus as `(source, texts)` pairs, each source non-empty.
fn corpus() -> Vec<(&'static str, Vec<String>)> {
    let csv = fs::read_to_string(CN_EN_SENTENCES).unwrap();
    let mut lines = csv.lines();
    assert_eq!(
        lines.next(),
        Some("Chinese,English"),
        "{CN_EN_SENTENCES} header"
    );
    let sentences: Vec<String> = lines
        .flat_map(|line| {
            let fields = csv_fields(line);
            assert_eq!(fields.len(), 2, "{CN_EN_SENTENCES}: {line:?}");
            fields
        })
        .collect();

    let mut golden = Vec::new();
    text_fields(&read_json(GOLDEN_FIXTURE)["fixtures"], &mut golden);

    let token_id_cases: Vec<String> = read_json(TOKEN_IDS_FIXTURE)["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let input = &case["input"];
            if let Some(text) = input["text"].as_str() {
                return text.to_string();
            }
            let repeat = &input["repeat"];
            let field = |key: &str| repeat[key].as_str().unwrap();
            let count = repeat["count"].as_u64().unwrap() as usize;
            format!(
                "{}{}{}",
                field("prefix"),
                field("unit").repeat(count),
                field("suffix")
            )
        })
        .collect();

    let mut benchmark = Vec::new();
    text_fields(&read_json(BENCHMARK_CORPUS), &mut benchmark);
    text_fields(&read_json(RETRIEVAL_EVAL_CASES), &mut benchmark);

    let mut edge_cases: Vec<String> = EDGE_CASES.iter().map(|text| text.to_string()).collect();
    // Long runs without a word boundary.
    for (unit, count) in [
        ("x", 5000),
        ("ab", 3000),
        ("中", 4000),
        ("😀", 1000),
        ("!", 2000),
    ] {
        edge_cases.push(unit.repeat(count));
    }

    let sources = vec![
        (CN_EN_SENTENCES, sentences),
        (GOLDEN_FIXTURE, golden),
        (TOKEN_IDS_FIXTURE, token_id_cases),
        ("benchmark and retrieval-eval texts", benchmark),
        ("edge cases", edge_cases),
        ("random texts", random_texts(RANDOM_TEXTS, RANDOM_SEED)),
    ];
    for (source, texts) in &sources {
        assert!(!texts.is_empty(), "no texts from {source}");
    }
    sources
}

#[test]
#[ignore = "needs a GGUF bundle; run with `cargo test --lib tokenizer_parity -- --ignored`"]
fn test_tokenizer_json_matches_gguf_vocabulary() {
    let Some(bundle_dir) = test_bundle() else {
        return;
    };
    let tokenizer_json = bundle_dir.join("tokenizer.json");
    let hf = HFTokenizer::from_file(&tokenizer_json.to_string_lossy()).unwrap();
    // Tokenizing reads only the vocabulary; the small context keeps the load cheap.
    let gguf = LlamaBackend::load(
        &bundle_dir.join("model.gguf"),
        RAW_EMBEDDING_DIMENSION,
        512,
        1,
    )
    .unwrap();
    let spec = ModelSpec::defaults();

    let sources = corpus();
    let mut encodings = 0usize;
    let mut tokens = 0usize;
    let mut mismatches = Vec::new();
    for (source, texts) in &sources {
        for (index, text) in texts.iter().enumerate() {
            let forms = [
                ("as is", text.clone()),
                ("query", prefixed_text(EmbeddingInput::query(text), &spec)),
                (
                    "document",
                    prefixed_text(EmbeddingInput::document(text), &spec),
                ),
            ];
            for (form, input) in forms {
                let expected = hf.encode(&input, usize::MAX).unwrap().input_ids;
                let actual = gguf.tokenize(&input).unwrap();
                encodings += 1;
                tokens += expected.len();
                if expected != actual {
                    mismatches.push(describe_mismatch(
                        source, index, form, &input, &expected, &actual,
                    ));
                }
            }
        }
    }

    let texts: usize = sources.iter().map(|(_, texts)| texts.len()).sum();
    let breakdown: Vec<String> = sources
        .iter()
        .map(|(source, texts)| format!("{} from {source}", texts.len()))
        .collect();
    eprintln!(
        "tokenizer parity: {texts} texts ({}), {encodings} encodings, {tokens} tokens, \
         {} mismatches",
        breakdown.join(", "),
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "tokenizer.json and the GGUF vocabulary disagree on {} of {encodings} encodings:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn describe_mismatch(
    source: &str,
    index: usize,
    form: &str,
    input: &str,
    expected: &[u32],
    actual: &[u32],
) -> String {
    let at = expected
        .iter()
        .zip(actual)
        .position(|(e, a)| e != a)
        .unwrap_or(expected.len().min(actual.len()));
    let window = |ids: &[u32]| ids[at.saturating_sub(2)..(at + 4).min(ids.len())].to_vec();
    let excerpt: String = input.chars().take(120).collect();
    format!(
        "{source} #{index} ({form}): first difference at token {at}: tokenizer.json {:?}, \
         GGUF {:?} ({} vs {} tokens); text {excerpt:?}",
        window(expected),
        window(actual),
        expected.len(),
        actual.len()
    )
}

#[test]
fn test_csv_fields_honours_quotes() {
    assert_eq!(
        csv_fields("他感冒了,He caught a cold."),
        ["他感冒了", "He caught a cold."]
    );
    assert_eq!(
        csv_fields(r#""很好,谢谢","Fine, thanks.""#),
        ["很好,谢谢", "Fine, thanks."]
    );
    assert_eq!(
        csv_fields(r#""He said ""hi""",x"#),
        [r#"He said "hi""#, "x"]
    );
    assert_eq!(csv_fields(""), [""]);
}

#[test]
fn test_random_texts_are_deterministic_and_varied() {
    let texts = random_texts(500, RANDOM_SEED);
    assert_eq!(texts, random_texts(500, RANDOM_SEED));
    assert_ne!(texts, random_texts(500, RANDOM_SEED + 1));
    assert!(texts.iter().all(|text| !text.is_empty()));
    // The generator reaches beyond ASCII and the Basic Multilingual Plane, and splices
    // special-token text.
    assert!(texts.iter().any(|text| !text.is_ascii()));
    assert!(texts
        .iter()
        .any(|text| text.chars().any(|c| c > '\u{FFFF}')));
    assert!(texts.iter().any(|text| text.contains("<eos>")));
}

#[test]
fn test_text_fields_collects_nested_text_values() {
    let value: Value = serde_json::from_str(
        r#"{"a": {"text": "one", "kind": "query"}, "b": [{"text": "two"}, {"id": "x"}],
            "text": "three"}"#,
    )
    .unwrap();
    let mut out = Vec::new();
    text_fields(&value, &mut out);
    out.sort();
    assert_eq!(out, ["one", "three", "two"]);
}

#[test]
fn test_corpus_reads_every_source() {
    let sources = corpus();
    let count = |name: &str| {
        sources
            .iter()
            .find(|(source, _)| *source == name)
            .map(|(_, texts)| texts.len())
            .unwrap()
    };
    // Two sentences per row.
    assert_eq!(count(CN_EN_SENTENCES) % 2, 0);
    assert!(count(CN_EN_SENTENCES) >= 5000);
    let entries = |path: &str, key: &str| read_json(path)[key].as_array().unwrap().len();
    assert_eq!(count(GOLDEN_FIXTURE), entries(GOLDEN_FIXTURE, "fixtures"));
    assert_eq!(
        count(TOKEN_IDS_FIXTURE),
        entries(TOKEN_IDS_FIXTURE, "cases")
    );
    assert_eq!(count("random texts"), RANDOM_TEXTS);
    // The repeated token-id cases are expanded, not stored as their spec.
    let token_id_texts = &sources
        .iter()
        .find(|(s, _)| *s == TOKEN_IDS_FIXTURE)
        .unwrap()
        .1;
    assert!(token_id_texts.iter().any(|text| text.len() > 100_000));
}
