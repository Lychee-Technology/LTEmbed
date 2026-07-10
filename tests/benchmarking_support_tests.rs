use approx::assert_relative_eq;
use ltembed::benchmarking::{
    benchmark_scenarios, scenario_by_name, scenario_inputs, scenario_token_lengths,
    selected_scenarios, LatencyStats,
};
use ltembed::engine::EmbeddingInputKind;
use ltembed::error::LTEmbedError;
use ltembed::traits::tokenizer::{Tokenizer, TokenizerOutput};

#[derive(Debug)]
struct CountingTokenizer;

impl Tokenizer for CountingTokenizer {
    fn encode(&self, text: &str, max_length: usize) -> Result<TokenizerOutput, LTEmbedError> {
        let tokens = text.split_whitespace().count() + 2;
        if tokens > max_length {
            return Err(LTEmbedError::InputTooLong {
                tokens,
                max: max_length,
            });
        }

        Ok(TokenizerOutput {
            input_ids: vec![1; tokens],
            attention_mask: vec![1; tokens],
            token_type_ids: vec![0; tokens],
        })
    }
}

#[test]
fn test_benchmark_scenarios_are_single_zh_and_en() {
    let names: Vec<_> = benchmark_scenarios().iter().map(|s| s.name).collect();
    assert_eq!(names, vec!["single/zh", "single/en"]);
    assert_eq!(scenario_by_name("single/zh").unwrap().batch_size, 1);
    assert_eq!(scenario_by_name("single/en").unwrap().text_profile, "en");
    assert!(scenario_by_name("missing/scenario").is_none());
}

#[test]
fn test_latency_stats_uses_expected_percentiles() {
    let stats = LatencyStats::from_samples_ms(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    assert_eq!(stats.mean_ms, 25.0);
    assert_eq!(stats.median_ms, 25.0);
    assert_eq!(stats.p95_ms, 38.5);
    assert_relative_eq!(stats.p99_ms, 39.7, epsilon = 1e-9);
    assert_eq!(stats.min_ms, 10.0);
    assert_eq!(stats.max_ms, 40.0);
}

#[test]
fn test_latency_stats_rejects_empty_samples() {
    let err = LatencyStats::from_samples_ms(&[]).unwrap_err();
    assert!(err.contains("empty"));
}

#[test]
fn test_selected_scenarios_returns_requested_scenario() {
    let selected = selected_scenarios(Some("single/zh")).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].name, "single/zh");
}

#[test]
fn test_selected_scenarios_rejects_unknown_name() {
    let err = selected_scenarios(Some("missing/scenario")).unwrap_err();
    assert!(err.contains("unknown scenario"));
}

#[test]
fn test_single_scenarios_carry_query_kind() {
    let zh = scenario_inputs(scenario_by_name("single/zh").expect("exists"));
    assert_eq!(zh.len(), 1);
    assert_eq!(zh[0].kind, EmbeddingInputKind::Query);
    let en = scenario_inputs(scenario_by_name("single/en").expect("exists"));
    assert_eq!(en.len(), 1);
    assert_eq!(en[0].kind, EmbeddingInputKind::Query);
}

#[test]
fn test_scenario_token_lengths_follow_tokenizer_outputs() {
    let tokenizer = CountingTokenizer;
    let scenario = scenario_by_name("single/zh").expect("scenario should exist");

    let lengths = scenario_token_lengths(&tokenizer, scenario, 512).unwrap();

    assert_eq!(lengths.len(), 1);
    assert!(lengths[0] > 0);
}
