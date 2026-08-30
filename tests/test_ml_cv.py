import numpy as np

from src.ml_cv import purged_time_series_splits


def test_embargo_gap_is_respected_in_every_split():
    splits = purged_time_series_splits(n=1000, n_splits=4, embargo=20)

    assert len(splits) == 4
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min()
        gap = val_idx.min() - train_idx.max()
        assert gap >= 20  # al menos el embargo (arange exclusivo agrega 1 más)


def test_splits_are_sequential_and_train_set_grows():
    splits = purged_time_series_splits(n=1000, n_splits=4, embargo=10)

    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)  # el train crece en cada pliegue sucesivo
    assert train_sizes[0] < train_sizes[-1]


def test_validation_windows_do_not_overlap_across_splits():
    splits = purged_time_series_splits(n=1000, n_splits=4, embargo=10)

    val_ranges = [(val_idx.min(), val_idx.max()) for _, val_idx in splits]
    for (start_a, end_a), (start_b, _) in zip(val_ranges, val_ranges[1:]):
        assert end_a < start_b


def test_too_few_samples_for_embargo_produces_fewer_or_no_splits():
    splits = purged_time_series_splits(n=10, n_splits=4, embargo=5)
    # con tan pocos datos y un embargo grande, no debería fallar -- solo devolver menos pliegues
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min()
