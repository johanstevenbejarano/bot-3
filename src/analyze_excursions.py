"""Corre el análisis de MAE/MFE sobre el tramo de TRAIN (nunca sobre test, para no filtrar info)."""
from __future__ import annotations

import json
import logging

from src.config import BACKTEST, PAIRS, RESULTS_DIR, STRATEGY
from src.data_fetch import fetch_ohlcv
from src.excursion_analysis import compute_mae_mfe, summarize_excursions
from src.indicators import add_indicators
from src.strategy import compute_layers
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HOLDING_BARS = 72  # 3 días en velas de 1h


def main() -> None:
    report = {}
    for symbol in PAIRS:
        raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=BACKTEST.years_of_history)
        with_indicators = add_indicators(raw, STRATEGY)
        train_df, _ = train_test_split_by_time(with_indicators, 0.7)

        with_signal = compute_layers(train_df, STRATEGY)
        mae_mfe = compute_mae_mfe(with_signal, holding_bars=HOLDING_BARS)
        summary = summarize_excursions(mae_mfe)

        logger.info("%s: %s", symbol, json.dumps(summary, indent=2))
        report[symbol] = summary

    out_path = RESULTS_DIR / "excursion_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
