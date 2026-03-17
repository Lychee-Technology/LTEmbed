use approx::assert_relative_eq;
use ltembed::benchmarking::{
    benchmark_scenarios, scenario_by_name, scenario_texts, selected_scenarios, LatencyStats,
};

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
