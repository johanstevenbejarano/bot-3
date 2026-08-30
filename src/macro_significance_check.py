"""Verificación de significancia estadística sobre el resultado positivo de `macro_validate.py`
(DXY) -- misma disciplina que se aplicó a la línea de clasificadores cuando el walk-forward daba
positivo (+0.44%, luego +0.98%) y el bootstrap reveló que el intervalo de confianza incluía cero.
Que "sostenga fuera de muestra" en expectancy agregada NO alcanza por sí solo: hay que ver si el
intervalo de confianza de los retornos trade-por-trade excluye cero, en TEST (la evidencia
genuinamente fuera de muestra), no en train.
"""
from __future__ import annotations

import json
import logging

from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, PAIRS, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.macro_indicators import add_macro_indicators
from src.macro_strategy import MacroContrarianStrategy, compute_layers
from src.ml_significance import bootstrap_mean_ci
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

YEARS_OF_HISTORY = 8
TRAIN_FRACTION = 0.7

# Config validada en macro_validate.py (results/macro_train_test_validation.json)
VALIDATED_CONFIG = {
    "lookback_periods": 270,
    "extreme_percentile": 0.85,
    "sl_atr_mult": 4.0,
    "tp_atr_mult": 8.0,
}


def _trade_returns_pct(df_with_indicators, cfg) -> list[float]:
    with_signal = compute_layers(df_with_indicators, cfg)
    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit

    bt = FractionalBacktest(
        bt_data, MacroContrarianStrategy,
        cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
        exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run(sl_atr_mult=cfg.sl_atr_mult, tp_atr_mult=cfg.tp_atr_mult)
    return (stats["_trades"]["ReturnPct"] * 100).tolist()


def main() -> None:
    from dataclasses import replace

    from src.config import MACRO_STRATEGY

    cfg = replace(MACRO_STRATEGY, **VALIDATED_CONFIG)

    report = {}
    for symbol in PAIRS:
        raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
        _, test_raw = train_test_split_by_time(raw, TRAIN_FRACTION)
        with_indicators = add_macro_indicators(test_raw, cfg)

        trade_returns = _trade_returns_pct(with_indicators, cfg)
        ci_result = bootstrap_mean_ci(trade_returns, n_boot=5000)

        logger.info(
            "%s (TEST, %d trades): media %.4f%%, IC95%% [%.4f%%, %.4f%%] -> %s",
            symbol, ci_result["n"], ci_result["mean"], ci_result["ci_low"], ci_result["ci_high"],
            "EXCLUYE cero" if ci_result["excludes_zero"] else "incluye cero",
        )
        report[symbol] = ci_result

    both_exclude_zero_positive = all(
        report[s]["excludes_zero"] and report[s]["mean"] > 0 for s in PAIRS
    )

    out = {
        "config": VALIDATED_CONFIG,
        "test_bootstrap_by_symbol": report,
        "both_symbols_exclude_zero_positive": both_exclude_zero_positive,
    }
    out_path = RESULTS_DIR / "macro_significance_check.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)
    logger.info(
        "%s",
        "AMBOS SIMBOLOS excluyen cero (positivo) -- evidencia estadistica real"
        if both_exclude_zero_positive
        else "Al menos un simbolo NO excluye cero -- no se puede descartar que sea ruido",
    )


if __name__ == "__main__":
    main()
