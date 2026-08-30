"""Backtest de la estrategia contraria de funding rate sobre BTC/USDT (período completo)."""
from __future__ import annotations

import json
import logging

from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, FUNDING_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.funding_indicators import add_funding_indicators
from src.funding_strategy import FundingContrarianStrategy, compute_layers
from src.monte_carlo import monte_carlo_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
YEARS_OF_HISTORY = 5


def _prepare_data(symbol: str):
    raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
    with_indicators = add_funding_indicators(raw, symbol, FUNDING_STRATEGY, years=YEARS_OF_HISTORY)
    with_signal = compute_layers(with_indicators, FUNDING_STRATEGY)

    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
    return bt_data


def main() -> None:
    data = _prepare_data(SYMBOL)

    bt = FractionalBacktest(
        data, FundingContrarianStrategy,
        cash=BACKTEST.initial_cash, commission=BACKTEST.commission,
        exclusive_orders=True, fractional_unit=BACKTEST.fractional_unit,
    )
    stats = bt.run()
    trades = stats["_trades"]
    num_trades = len(trades)

    if num_trades < BACKTEST.min_trades_for_significance:
        logger.warning(
            "%s: solo %d trades (< %d) -> muestra insuficiente",
            SYMBOL, num_trades, BACKTEST.min_trades_for_significance,
        )

    mc_result = None
    if num_trades > 0:
        mc_result = monte_carlo_drawdown(
            trades["ReturnPct"].to_numpy() * 100, n_iterations=BACKTEST.monte_carlo_iterations
        )

    report = {
        "symbol": SYMBOL,
        "period_start": str(data.index[0]),
        "period_end": str(data.index[-1]),
        "num_trades": num_trades,
        "win_rate_pct": float(stats["Win Rate [%]"]) if num_trades else 0.0,
        "expectancy_pct": float(trades["ReturnPct"].mean() * 100) if num_trades else 0.0,
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "sharpe_ratio": float(stats["Sharpe Ratio"]) if num_trades else float("nan"),
        "return_pct": float(stats["Return [%]"]),
        "monte_carlo_drawdown_pct": mc_result,
        "significant_sample": num_trades >= BACKTEST.min_trades_for_significance,
    }

    out_path = RESULTS_DIR / "funding_BTC-USDT_backtest.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)
    logger.info("%s", json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
