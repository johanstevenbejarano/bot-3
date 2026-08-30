"""Busca parámetros SOLO en el tramo de entrenamiento y valida la config elegida en test.

Corre una vez, sin retocar nada según lo que salga en test: si el resultado en test es malo,
se reporta tal cual (eso es justamente el punto de esta separación).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

from src.config import BACKTEST, PAIRS, RESULTS_DIR, STRATEGY
from src.data_fetch import fetch_ohlcv
from src.indicators import add_indicators
from src.validation import grid_search_best_config, run_config, train_test_split_by_time

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
        logger.info(
            "%s: train %s -> %s (%d velas), test %s -> %s (%d velas)",
            symbol, train_df.index[0], train_df.index[-1], len(train_df),
            test_df.index[0], test_df.index[-1], len(test_df),
        )

    logger.info("Buscando config en TRAIN unicamente (test no se toca todavia)...")
    best_cfg, train_detail = grid_search_best_config(train_by_symbol)

    if best_cfg is None:
        report = {
            "status": "no_viable_config",
            "message": (
                "Ninguna combinacion de parametros probada da expectancy positiva de forma "
                "consistente en todos los pares durante el periodo de entrenamiento. No hay "
                "config que pase a validar en test."
            ),
        }
        logger.warning(report["message"])
    else:
        logger.info("Mejor config en TRAIN: %s", asdict(best_cfg))
        logger.info("Detalle TRAIN: %s", json.dumps(train_detail, indent=2))

        test_detail = {
            symbol: run_config(test_by_symbol[symbol], best_cfg, sl_anchor="entry")._asdict()
            for symbol in PAIRS
        }
        logger.info("Detalle TEST (fuera de muestra): %s", json.dumps(test_detail, indent=2))

        holds_up = all(v["expectancy_pct"] > 0 for v in test_detail.values())
        report = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_fraction": TRAIN_FRACTION,
            "train_metrics": train_detail,
            "test_metrics": test_detail,
            "holds_up_out_of_sample": holds_up,
        }
        if not holds_up:
            logger.warning(
                "La config que ganaba en TRAIN pierde en TEST -> confirma sobreajuste, "
                "no usar esta config en produccion."
            )

    out_path = RESULTS_DIR / "train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
