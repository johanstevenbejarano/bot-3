"""Backtest de reversión a la media sobre BTC/USDT y ETH/USDT (período completo, sin split)."""
from __future__ import annotations

import json
import logging

from backtesting.lib import FractionalBacktest

from src.config import BACKTEST, MEANREV_STRATEGY, PAIRS, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.meanrev_indicators import add_meanrev_indicators
from src.meanrev_strategy import MeanReversionStrategy, compute_layers
from src.monte_carlo import monte_carlo_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _prepare_data(symbol: str):
    raw = fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=BACKTEST.years_of_history)
    with_indicators = add_meanrev_indicators(raw, MEANREV_STRATEGY)
    with_signal = compute_layers(with_indicators, MEANREV_STRATEGY)

    bt_data = with_signal.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    # atr y bb_basis son columnas de precio propias: FractionalBacktest no las reescala solo, hay
    # que hacerlo a mano para que sigan siendo comparables con Close dentro de la estrategia.
    bt_data["atr"] = bt_data["atr"] * BACKTEST.fractional_unit
    bt_data["bb_basis"] = bt_data["bb_basis"] * BACKTEST.fractional_unit
    return bt_data


def run_backtest_for_pair(symbol: str) -> dict:
    data = _prepare_data(symbol)

    bt = FractionalBacktest(
        data,
        MeanReversionStrategy,
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
            trades["ReturnPct"].to_numpy() * 100, n_iterations=BACKTEST.monte_carlo_iterations
        )

    report = {
        "symbol": symbol,
        "timeframe": BACKTEST.timeframe,
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

    out_path = RESULTS_DIR / f"meanrev_{symbol.replace('/', '-')}_backtest.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)
    return report


def main() -> None:
    for symbol in PAIRS:
        logger.info("=== Mean reversion backtest %s ===", symbol)
        try:
            report = run_backtest_for_pair(symbol)
            logger.info("%s", json.dumps(report, indent=2, default=str))
        except Exception:
            logger.exception("Fallo el backtest de %s", symbol)


if __name__ == "__main__":
    main()
