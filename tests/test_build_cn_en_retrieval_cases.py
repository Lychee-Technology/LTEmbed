import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_cn_en_retrieval_cases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_cn_en", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_fixture_covers_all_benchmark_scenarios():
    gen = load_module()
    pairs = [("短", "short"), ("中等长度的句子", "a medium length sentence"), ("最长的一句话在这里", "the longest sentence here")]
    fixture = gen.build_fixture(pairs)
    scenarios = fixture["scenarios"]
    assert set(scenarios.keys()) == {
        "single/zh",
        "single/en",
        "single/medium",
        "single/long",
        "batch/medium/8",
    }
    assert len(scenarios["single/zh"]) == 1
    assert scenarios["single/zh"][0]["kind"] == "query"
    assert len(scenarios["single/en"]) == 1
    # zh and en come from the SAME representative pair
    zh = scenarios["single/zh"][0]["text"]
    en = scenarios["single/en"][0]["text"]
    assert (zh, en) == gen.pick_representative(pairs)
    # medium/long/batch come byte-identical from the checked-in corpus
    corpus = json.loads(gen.DEFAULT_CORPUS.read_text(encoding="utf-8"))
    assert scenarios["single/medium"][0]["text"] == corpus["medium"]["text"]
    assert scenarios["single/long"][0]["text"] == corpus["long"]["text"]
    assert len(scenarios["batch/medium/8"]) == 8
    assert len(scenarios["single/long"][0]["text"]) > 4 * len(scenarios["single/medium"][0]["text"])


def test_pick_representative_is_median_english_length_and_deterministic():
    gen = load_module()
    pairs = [("a", "x"), ("b", "yyy"), ("c", "zz")]
    # sorted by len(en): "x"(1), "zz"(2), "yyy"(3) -> median index 1 -> ("c","zz")
    assert gen.pick_representative(pairs) == ("c", "zz")
    assert gen.pick_representative(pairs) == gen.pick_representative(pairs)
