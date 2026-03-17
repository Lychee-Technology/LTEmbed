use approx::assert_relative_eq;
use ltembed::benchmarking::{
    benchmark_scenarios, gemm_microbenchmark_scenarios, padded_seq_len, scenario_by_name,
    scenario_texts, scenario_token_lengths, selected_scenarios, LatencyStats,
};
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
fn test_benchmark_scenarios_match_issue_38_plan() {
    let scenario_names: Vec<_> = benchmark_scenarios()
        .iter()
        .map(|scenario| scenario.name)
        .collect();
    assert_eq!(
        scenario_names,
        vec![
            "single/short",
            "single/medium",
            "single/long",
            "batch/medium/1",
            "batch/medium/4",
            "batch/medium/8",
            "batch/mixed/8",
            "batch/medium/16",
        ]
    );

    assert_eq!(scenario_by_name("batch/medium/16").unwrap().batch_size, 16);
    assert_eq!(
        scenario_by_name("single/long").unwrap().text_profile,
        "long"
    );
    assert!(scenario_by_name("missing/scenario").is_none());
}

#[test]
fn test_gemm_microbenchmark_scenarios_target_expected_workloads() {
    let scenario_names: Vec<_> = gemm_microbenchmark_scenarios()
        .into_iter()
        .map(|scenario| scenario.name)
        .collect();

    assert_eq!(
        scenario_names,
        vec!["single/long", "batch/medium/8", "batch/medium/16"]
    );
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
    let selected = selected_scenarios(Some("batch/medium/8")).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].name, "batch/medium/8");
}

#[test]
fn test_selected_scenarios_rejects_unknown_name() {
    let err = selected_scenarios(Some("missing/scenario")).unwrap_err();
    assert!(err.contains("unknown scenario"));
}

#[test]
fn test_batch_mixed_scenario_uses_variable_length_texts() {
    let scenario = scenario_by_name("batch/mixed/8").expect("scenario should exist");
    let texts = scenario_texts(scenario);
    let lengths: Vec<_> = texts.iter().map(|text| text.len()).collect();

    assert_eq!(texts.len(), 8);
    assert!(lengths.iter().any(|&len| len == lengths[0]));
    assert!(lengths.iter().any(|&len| len != lengths[0]));
}

#[test]
fn test_scenario_token_lengths_follow_tokenizer_outputs() {
    let tokenizer = CountingTokenizer;
    let scenario = scenario_by_name("batch/medium/8").expect("scenario should exist");

    let lengths = scenario_token_lengths(&tokenizer, scenario, 512).unwrap();

    assert_eq!(lengths.len(), 8);
    assert!(lengths.iter().all(|&length| length == lengths[0]));
    assert_eq!(padded_seq_len(&lengths), lengths[0]);
}

#[test]
fn test_scenario_token_lengths_preserve_mixed_padding_shape() {
    let tokenizer = CountingTokenizer;
    let scenario = scenario_by_name("batch/mixed/8").expect("scenario should exist");

    let lengths = scenario_token_lengths(&tokenizer, scenario, 512).unwrap();

    assert_eq!(lengths.len(), 8);
    assert!(lengths.iter().any(|&length| length != lengths[0]));
    assert_eq!(padded_seq_len(&lengths), *lengths.iter().max().unwrap());
}
