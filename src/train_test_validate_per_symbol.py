"""Busca la mejor config POR SÍMBOLO en train (sin exigir que sirva para ambos pares a la vez)
y valida cada una en su propio tramo de test. Igual que train_test_validate.py, no se retoca
nada según lo que salga en test: se reporta tal cual.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

from src.config import BACKTEST, PAIRS, RESULTS_DIR, STRATEGY
from src.data_fetch import fetch_ohlcv
from src.indicators import add_indicators
from src.validation import grid_search_best_config_per_symbol, run_config, train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FRACTION = 0.7


def main() -> None:
    train_by_symbol = {}
    test_by_symbol = {}

    for symbol in PAIRS:
        raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=BACKTEST.years_of_history)
        with_indicators = add_indicators(raw, STRATEGY)
        train_df, test_df = train_test_split_by_time(with_indicators, TRAIN_FRACTION)
        train_by_symbol[symbol] = train_df
        test_by_symbol[symbol] = test_df

    logger.info("Buscando la mejor config POR SIMBOLO en TRAIN (test no se toca todavia)...")
    best_by_symbol = grid_search_best_config_per_symbol(train_by_symbol, min_trades=25)

    report = {}
    for symbol in PAIRS:
        best_cfg, train_result = best_by_symbol[symbol]

        if best_cfg is None:
            report[symbol] = {
                "status": "no_viable_config",
                "message": "Ninguna combinacion probada da expectancy positiva en train para este simbolo.",
            }
            logger.warning("%s: %s", symbol, report[symbol]["message"])
            continue

        test_result = run_config(test_by_symbol[symbol], best_cfg, sl_anchor="entry")
        holds_up = test_result.expectancy_pct > 0

        report[symbol] = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_metrics": train_result._asdict(),
            "test_metrics": test_result._asdict(),
            "holds_up_out_of_sample": holds_up,
        }
        logger.info(
            "%s: TRAIN exp=%.3f%% (n=%d) -> TEST exp=%.3f%% (n=%d) -> %s",
            symbol, train_result.expectancy_pct, train_result.num_trades,
            test_result.expectancy_pct, test_result.num_trades,
            "SOSTIENE fuera de muestra" if holds_up else "NO sostiene fuera de muestra",
        )

    out_path = RESULTS_DIR / "train_test_validation_per_symbol.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
