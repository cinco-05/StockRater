import json

import pandas as pd

import stock_analyzer as analyzer


def test_self_test_passes():
    assert analyzer.self_test() == 0


def test_reverse_dcf_reports_exact_and_bounded_solutions():
    data = analyzer._synthetic_company()
    assumptions = analyzer.build_assumptions(data, 0.04, "test")
    starting_fcff, _, _, _ = analyzer.normalized_owner_earnings(data, assumptions)
    assert starting_fcff is not None

    growth, bound = analyzer.reverse_dcf(data, starting_fcff, assumptions)
    assert growth is not None
    assert bound == "exact"

    analyzer.put_metric(data, "current_price", 1_000_000.0, "test", "current", None, 1.0)
    growth, bound = analyzer.reverse_dcf(data, starting_fcff, assumptions)
    assert growth == 0.60
    assert bound == "lower_bound"


def test_corrupt_cache_entry_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzer, "CACHE_DIR", tmp_path)
    path = analyzer._cache_path("raw", "TEST")
    path.write_text("not json", encoding="utf-8")
    assert analyzer.cache_get("raw", "TEST") is None
    assert not path.exists()


def test_cache_schema_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzer, "CACHE_DIR", tmp_path)
    value = {"frame": pd.DataFrame([[1.0]], index=["row"], columns=["column"])}
    analyzer.cache_put("raw", "TEST", value)
    envelope = json.loads(analyzer._cache_path("raw", "TEST").read_text(encoding="utf-8"))
    assert envelope["schema"] == analyzer.CACHE_SCHEMA_VERSION
    restored = analyzer.cache_get("raw", "TEST")
    assert restored["frame"].iloc[0, 0] == 1.0


def test_backtest_drops_rows_without_benchmark(tmp_path, capsys):
    rows = []
    for index in range(25):
        rows.append({
            "date": f"2026-01-{index % 25 + 1:02d}",
            "ticker": f"T{index}",
            "score": index,
            "future_return": index / 100,
            "benchmark_return": "" if index < 5 else 0.01,
        })
    path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    assert analyzer.run_backtest(str(path)) == 0
    output = capsys.readouterr().out
    assert "Excluded 5 row(s)" in output
    assert "benchmark-relative excess return" in output


def test_invalid_configuration_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"model": {"unknown": 1}}', encoding="utf-8")
    try:
        analyzer.load_model_config(str(path))
    except analyzer.ConfigurationError as exc:
        assert "Unknown model setting" in str(exc)
    else:
        raise AssertionError("invalid configuration was accepted")
