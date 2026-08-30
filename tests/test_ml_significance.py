import numpy as np

from src.ml_significance import bootstrap_mean_ci


def test_clearly_positive_signal_excludes_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=2.0, scale=1.0, size=500)  # media claramente > 0
    result = bootstrap_mean_ci(values, n_boot=1000)

    assert result["excludes_zero"]
    assert result["ci_low"] > 0


def test_noise_around_zero_does_not_exclude_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=5.0, size=50)  # media ~0, mucho ruido, poca muestra
    result = bootstrap_mean_ci(values, n_boot=1000)

    assert not result["excludes_zero"]


def test_empty_input_handled_gracefully():
    result = bootstrap_mean_ci([])
    assert result["n"] == 0
    assert not result["excludes_zero"]
