"""Orquesta descarga de datos, indicadores, backtest y Monte Carlo para cada par."""
from __future__ import annotations

import json
import logging

from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, PAIRS, RESULTS_DIR, STRATEGY
from src.data_fetch import fetch_ohlcv
from src.indicators import add_indicators
from src.monte_carlo import monte_carlo_drawdown
from src.strategy import ConfluenceStrategy, compute_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _prepare_data(symbol: str):
    raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=BACKTEST.years_of_history)
    with_indicators = add_indicators(raw, STRATEGY)
    with_signal = compute_layers(with_indicators, STRATEGY)

    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    # FractionalBacktest reescala Open/High/Low/Close/Volume automáticamente pero no toca
    # columnas propias: atr, swing_low y swing_high deben reescalarse a mano para seguir siendo
    # comparables con Close dentro de ConfluenceStrategy.next().
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
    bt_data["swing_low"] = bt_data["swing_low"] * BACKTEST.fractional_unit
    bt_data["swing_high"] = bt_data["swing_high"] * BACKTEST.fractional_unit
    return bt_data


def run_backtest_for_pair(symbol: str) -> dict:
    data = _prepare_data(symbol)

    bt = FractionalBacktest(
        data,
        ConfluenceStrategy,
        cash=BACKTEST.initial_cash,
        commission=BACKTEST.commission,
        exclusive_orders=True,
        fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run()

    trades = stats["_trades"]
    num_trades = len(trades)

    if num_trades < BACKTEST.min_trades_for_significance:
        logger.warning(
            "%s: solo %d trades (< %d) -> muestra insuficiente para conclusiones robustas",
            symbol, num_trades, BACKTEST.min_trades_for_significance,
        )

    mc_result = None
    if num_trades > 0:
        mc_result = monte_carlo_drawdown(
            trades["ReturnPct"].to_numpy() * 100,
            n_iterations=BACKTEST.monte_carlo_iterations,
        )

    win_rate = float(stats["Win Rate [%]"]) if num_trades else 0.0
    expectancy = float(trades["ReturnPct"].mean() * 100) if num_trades else 0.0

    report = {
        "symbol": symbol,
        "timeframe": BACKTEST.timeframe,
        "period_start": str(data.index[0]),
        "period_end": str(data.index[-1]),
        "num_trades": num_trades,
        "win_rate_pct": win_rate,
        "expectancy_pct": expectancy,
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "sharpe_ratio": float(stats["Sharpe Ratio"]),
        "return_pct": float(stats["Return [%]"]),
        "monte_carlo_drawdown_pct": mc_result,
        "significant_sample": num_trades >= BACKTEST.min_trades_for_significance,
    }

    out_path = RESULTS_DIR / f"{symbol.replace('/', '-')}_backtest.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)

    return report


def main() -> None:
    for symbol in PAIRS:
        logger.info("=== Backtest %s ===", symbol)
        try:
            report = run_backtest_for_pair(symbol)
            logger.info("%s", json.dumps(report, indent=2, default=str))
        except Exception:
            logger.exception("Fallo el backtest de %s", symbol)


if __name__ == "__main__":
    main()
