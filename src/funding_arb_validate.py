"""Busca el umbral de entrada del arbitraje de funding SOLO en train y valida en test — misma
disciplina que el resto de las líneas: exige retorno medio por episodio positivo en AMBOS pares a
la vez en train antes de mirar test. Si sostiene, escala a bootstrap de significancia sobre los
episodios de test (misma verificación que reveló que la línea de DXY era ruido).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import BACKTEST, PAIRS, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.funding_arb_data import fetch_perp_close
from src.funding_arb_strategy import backtest_funding_arb
from src.funding_data import fetch_funding_rate
from src.ml_significance import bootstrap_mean_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

YEARS_OF_HISTORY = 5  # funding rate history de Binance -- ver funding_data.py
TRAIN_FRACTION = 0.7
ENTRY_THRESHOLDS = (0.0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005)
EXIT_THRESHOLD = 0.0
MIN_EPISODES_PER_SYMBOL = 15


def _split_by_time(series: pd.Series, train_fraction: float) -> tuple[pd.Series, pd.Series]:
    cutoff_idx = int(len(series) * train_fraction)
    return series.iloc[:cutoff_idx], series.iloc[cutoff_idx:]


def main() -> None:
    spot_by_symbol = {}
    perp_by_symbol = {}
    funding_by_symbol = {}
    for symbol in PAIRS:
        spot_by_symbol[symbol] = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)["close"]
        perp_by_symbol[symbol] = fetch_perp_close(symbol, years=YEARS_OF_HISTORY)
        funding_by_symbol[symbol] = fetch_funding_rate(symbol, years=YEARS_OF_HISTORY)

    train_funding = {}
    test_funding = {}
    for symbol in PAIRS:
        train_f, test_f = _split_by_time(funding_by_symbol[symbol], TRAIN_FRACTION)
        train_funding[symbol] = train_f
        test_funding[symbol] = test_f
        logger.info(
            "%s: train funding %s -> %s (%d eventos), test %s -> %s (%d eventos)",
            symbol, train_f.index[0], train_f.index[-1], len(train_f),
            test_f.index[0], test_f.index[-1], len(test_f),
        )

    logger.info("Buscando umbral de entrada en TRAIN unicamente (test no se toca todavia)...")
    best_threshold = None
    best_score = float("-inf")
    best_train_detail = {}

    for entry_th in ENTRY_THRESHOLDS:
        per_symbol_returns = {
            symbol: backtest_funding_arb(
                spot_by_symbol[symbol], perp_by_symbol[symbol], train_funding[symbol],
                entry_threshold=entry_th, exit_threshold=EXIT_THRESHOLD, commission=BACKTEST.commission,
            )
            for symbol in PAIRS
        }

        n_episodes = {s: len(r) for s, r in per_symbol_returns.items()}
        means = {s: (sum(r) / len(r) if r else float("nan")) for s, r in per_symbol_returns.items()}
        logger.info(
            "entry_threshold=%s -> %s",
            entry_th, {s: {"n_episodes": n_episodes[s], "mean_return_pct": means[s]} for s in PAIRS},
        )

        if any(n < MIN_EPISODES_PER_SYMBOL for n in n_episodes.values()):
            continue
        if min(means.values()) <= 0:
            continue

        score = min(means.values())
        if score > best_score:
            best_score = score
            best_threshold = entry_th
            best_train_detail = {
                s: {"n_episodes": len(per_symbol_returns[s]), "mean_return_pct": means[s]} for s in PAIRS
            }

    if best_threshold is None:
        report = {
            "status": "no_viable_threshold",
            "message": "Ningun umbral probado da retorno medio positivo por episodio en ambos pares en train.",
        }
        logger.warning(report["message"])
    else:
        logger.info("Mejor umbral en TRAIN: entry_threshold=%s", best_threshold)
        logger.info("TRAIN detail: %s", json.dumps(best_train_detail, indent=2))

        test_returns_by_symbol = {
            symbol: backtest_funding_arb(
                spot_by_symbol[symbol], perp_by_symbol[symbol], test_funding[symbol],
                entry_threshold=best_threshold, exit_threshold=EXIT_THRESHOLD, commission=BACKTEST.commission,
            )
            for symbol in PAIRS
        }
        test_detail = {
            s: {"n_episodes": len(r), "mean_return_pct": (sum(r) / len(r)) if r else float("nan")}
            for s, r in test_returns_by_symbol.items()
        }
        logger.info("TEST detail: %s", json.dumps(test_detail, indent=2))

        holds_up = all(
            len(r) >= MIN_EPISODES_PER_SYMBOL and (sum(r) / len(r)) > 0
            for r in test_returns_by_symbol.values()
        )

        report = {
            "entry_threshold": best_threshold,
            "exit_threshold": EXIT_THRESHOLD,
            "train_metrics": best_train_detail,
            "test_metrics": test_detail,
            "holds_up_out_of_sample": holds_up,
        }

        if holds_up:
            logger.info("SOSTIENE fuera de muestra -- corriendo bootstrap de significancia sobre TEST...")
            bootstrap_by_symbol = {}
            for symbol, returns in test_returns_by_symbol.items():
                ci = bootstrap_mean_ci(returns, n_boot=5000)
                bootstrap_by_symbol[symbol] = ci
                logger.info(
                    "%s (TEST, %d episodios): media %.4f%%, IC95%% [%.4f%%, %.4f%%] -> %s",
                    symbol, ci["n"], ci["mean"], ci["ci_low"], ci["ci_high"],
                    "EXCLUYE cero" if ci["excludes_zero"] else "incluye cero",
                )
            report["test_bootstrap"] = bootstrap_by_symbol
            both_significant_positive = all(
                bootstrap_by_symbol[s]["excludes_zero"] and bootstrap_by_symbol[s]["mean"] > 0 for s in PAIRS
            )
            report["status"] = "validated_significant" if both_significant_positive else "validated_but_not_significant"
            logger.info(
                "%s",
                "AMBOS SIMBOLOS excluyen cero -- evidencia estadistica real"
                if both_significant_positive
                else "Sostiene en agregado pero NO excluye cero -- no se puede descartar ruido",
            )
        else:
            report["status"] = "failed_out_of_sample"
            logger.info("NO sostiene fuera de muestra")

    out_path = RESULTS_DIR / "funding_arb_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
