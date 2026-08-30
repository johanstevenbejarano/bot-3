from typing import NamedTuple

import pandas as pd

from src.walk_forward import make_folds, run_walk_forward


def test_make_folds_covers_expected_range_without_overlap_leak():
    index = pd.date_range("2020-01-01", "2023-01-01", freq="1h", tz="UTC")
    folds = make_folds(index, train_days=365, test_days=90, step_days=90)

    assert len(folds) > 1
    for fold in folds:
        assert fold.train_start < fold.train_end == fold.test_start < fold.test_end
        assert fold.test_end <= index[-1]

    # los folds avanzan en el tiempo, no se repiten
    starts = [f.train_start for f in folds]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_make_folds_empty_when_history_too_short():
    index = pd.date_range("2023-01-01", "2023-02-01", freq="1h", tz="UTC")
    folds = make_folds(index, train_days=365, test_days=90)
    assert folds == []


class _FakeResult(NamedTuple):
    num_trades: int
    expectancy_pct: float
    return_pct: float


def test_run_walk_forward_calls_grid_search_only_with_train_slice():
    index = pd.date_range("2020-01-01", periods=24 * 400, freq="1h", tz="UTC")
    raw = pd.DataFrame({"close": range(len(index))}, index=index)

    seen_train_ranges = []

    def fake_grid_search(train_raw: pd.DataFrame):
        seen_train_ranges.append((train_raw.index[0], train_raw.index[-1]))
        return "some_cfg"

    def fake_run(df_slice: pd.DataFrame, cfg):
        assert cfg == "some_cfg"
        return _FakeResult(num_trades=10, expectancy_pct=1.0, return_pct=5.0)

    results = run_walk_forward(
        raw, grid_search_fn=fake_grid_search, run_fn=fake_run,
        train_days=365, test_days=30, step_days=30,
    )

    assert len(results) >= 1
    assert len(seen_train_ranges) == len(results)
    for fr in results:
        assert fr.test_num_trades == 10
        assert fr.test_expectancy_pct == 1.0


def test_on_fold_complete_callback_fires_once_per_fold_in_order():
    index = pd.date_range("2020-01-01", periods=24 * 400, freq="1h", tz="UTC")
    raw = pd.DataFrame({"close": range(len(index))}, index=index)

    seen = []

    def fake_grid_search(train_raw: pd.DataFrame):
        return "cfg"

    def fake_run(df_slice: pd.DataFrame, cfg):
        return _FakeResult(num_trades=1, expectancy_pct=0.0, return_pct=0.0)

    results = run_walk_forward(
        raw, grid_search_fn=fake_grid_search, run_fn=fake_run,
        train_days=365, test_days=30, step_days=30,
        on_fold_complete=lambda idx, total, fr: seen.append((idx, total)),
    )

    assert len(seen) == len(results)
    assert [idx for idx, _ in seen] == list(range(len(results)))
    assert all(total == len(results) for _, total in seen)
