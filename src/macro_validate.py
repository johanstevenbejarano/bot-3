"""Busca parámetros de la estrategia macro (DXY) SOLO en train y valida en test — misma
disciplina que las líneas anteriores (ver FINDINGS.md): exige expectancy positiva en AMBOS pares
a la vez en train antes de mirar test.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from itertools import product
from typing import NamedTuple

import pandas as pd
from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, MACRO_STRATEGY, PAIRS, RESULTS_DIR, MacroConfig
from src.data_fetch import fetch_ohlcv
from src.macro_indicators import add_macro_indicators
from src.macro_strategy import MacroContrarianStrategy, compute_layers
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

YEARS_OF_HISTORY = 8
TRAIN_FRACTION = 0.7


class RunResult(NamedTuple):
    num_trades: int
    win_rate_pct: float
    expectancy_pct: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float


def run_config(df_with_indicators: pd.DataFrame, cfg: MacroConfig) -> RunResult:
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
    trades = stats["_trades"]
    n = len(trades)
    return RunResult(
        num_trades=n,
        win_rate_pct=float(stats["Win Rate [%]"]) if n else 0.0,
        expectancy_pct=float(trades["ReturnPct"].mean() * 100) if n else 0.0,
        return_pct=float(stats["Return [%]"]),
        max_drawdown_pct=float(stats["Max. Drawdown [%]"]),
        sharpe_ratio=float(stats["Sharpe Ratio"]) if n else float("nan"),
    )


def _train_test_slices(raw: pd.DataFrame, cfg: MacroConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_indicators = add_macro_indicators(raw, cfg)
    return train_test_split_by_time(with_indicators, TRAIN_FRACTION)


def grid_search_best_config(
    train_by_symbol: dict[str, pd.DataFrame],
    lookback_values: tuple[int, ...] = (90, 180, 270),
    extreme_percentiles: tuple[float, ...] = (0.85, 0.90, 0.95),
    sl_atr_mults: tuple[float, ...] = (2.0, 3.0, 4.0),
    tp_atr_mults: tuple[float, ...] = (4.0, 6.0, 8.0),
    min_trades_per_symbol: int = 20,
) -> tuple[MacroConfig | None, dict]:
    """Busca, SOLO en train, la config con mejor expectancy mínima consistente entre AMBOS
    símbolos -- misma disciplina que `breakout_validate.grid_search_best_config`. El lookback se
    recalcula por combinación (a diferencia de funding/sentiment) porque el DXY es mucho más
    liviano de recalcular (una sola serie diaria, no por símbolo).
    """
    best_cfg = None
    best_score = float("-inf")
    best_detail: dict = {}

    for lookback, extreme_pct, sl_mult, tp_mult in product(
        lookback_values, extreme_percentiles, sl_atr_mults, tp_atr_mults
    ):
        cfg = replace(
            MACRO_STRATEGY,
            lookback_periods=lookback, extreme_percentile=extreme_pct,
            sl_atr_mult=sl_mult, tp_atr_mult=tp_mult,
        )

        per_symbol = {}
        for symbol, raw_train in train_by_symbol.items():
            with_indicators = add_macro_indicators(raw_train, cfg)
            per_symbol[symbol] = run_config(with_indicators, cfg)

        if any(r.num_trades < min_trades_per_symbol for r in per_symbol.values()):
            continue
        expectancies = [r.expectancy_pct for r in per_symbol.values()]
        if min(expectancies) <= 0:
            continue

        score = min(expectancies)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_detail = {s: r._asdict() for s, r in per_symbol.items()}

    return best_cfg, best_detail


def main() -> None:
    raw_by_symbol = {
        symbol: fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
        for symbol in PAIRS
    }

    # split train/test se hace sobre el OHLCV crudo, antes de calcular indicadores -- el lookback
    # varía por combinación en la búsqueda, así que no puede pre-calcularse una sola vez.
    train_raw_by_symbol = {}
    test_raw_by_symbol = {}
    for symbol, raw in raw_by_symbol.items():
        train_df, test_df = train_test_split_by_time(raw, TRAIN_FRACTION)
        train_raw_by_symbol[symbol] = train_df
        test_raw_by_symbol[symbol] = test_df
        logger.info(
            "%s: train %s -> %s (%d velas), test %s -> %s (%d velas)",
            symbol, train_df.index[0], train_df.index[-1], len(train_df),
            test_df.index[0], test_df.index[-1], len(test_df),
        )

    logger.info("Buscando config en TRAIN unicamente (test no se toca todavia)...")
    best_cfg, train_detail = grid_search_best_config(train_raw_by_symbol)

    if best_cfg is None:
        report = {
            "status": "no_viable_config",
            "message": "Ninguna combinacion probada da expectancy positiva en ambos pares en train.",
        }
        logger.warning(report["message"])
    else:
        logger.info("Mejor config en TRAIN: %s", asdict(best_cfg))
        logger.info("TRAIN detail: %s", json.dumps(train_detail, indent=2))

        test_detail = {}
        for symbol, raw_test in test_raw_by_symbol.items():
            with_indicators = add_macro_indicators(raw_test, best_cfg)
            test_detail[symbol] = run_config(with_indicators, best_cfg)._asdict()
        logger.info("TEST detail: %s", json.dumps(test_detail, indent=2))

        holds_up = all(v["expectancy_pct"] > 0 for v in test_detail.values())
        report = {
            "status": "validated" if holds_up else "failed_out_of_sample",
            "config": asdict(best_cfg),
            "train_metrics": train_detail,
            "test_metrics": test_detail,
            "holds_up_out_of_sample": holds_up,
        }
        logger.info("%s", "SOSTIENE fuera de muestra" if holds_up else "NO sostiene fuera de muestra")

    out_path = RESULTS_DIR / "macro_train_test_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)


if __name__ == "__main__":
    main()
