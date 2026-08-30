"""Backtest de pairs trading BTC/ETH sobre el período completo (sin split train/test)."""
from __future__ import annotations

import json
import logging

from src.config import BACKTEST, PAIRS_STRATEGY, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.monte_carlo import monte_carlo_drawdown
from src.pairs_strategy import PairsConfig, backtest_pairs, compute_spread_zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pairs_backtest(cfg: PairsConfig = PAIRS_STRATEGY) -> dict:
    df_a = fetch_ohlcv(cfg.symbol_a, timeframe=BACKTEST.timeframe, years=cfg.years_of_history)
    df_b = fetch_ohlcv(cfg.symbol_b, timeframe=BACKTEST.timeframe, years=cfg.years_of_history)

    df = compute_spread_zscore(df_a, df_b, cfg)
    trades_df, stats = backtest_pairs(df, cfg)

    mc_result = None
    if stats.num_trades > 0:
        mc_result = monte_carlo_drawdown(
            trades_df["net_pnl_pct"].to_numpy(), n_iterations=BACKTEST.monte_carlo_iterations
        )

    report = {
        "pair": f"{cfg.symbol_a} / {cfg.symbol_b} spread",
        "timeframe": BACKTEST.timeframe,
        "period_start": str(df.index[0]),
        "period_end": str(df.index[-1]),
        "num_trades": stats.num_trades,
        "win_rate_pct": stats.win_rate_pct,
        "expectancy_pct": stats.expectancy_pct,
        "return_pct": stats.return_pct,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "sharpe_ratio_per_trade": stats.sharpe_ratio,
        "monte_carlo_drawdown_pct": mc_result,
        "significant_sample": stats.num_trades >= BACKTEST.min_trades_for_significance,
    }

    if stats.num_trades < BACKTEST.min_trades_for_significance:
        logger.warning(
            "Solo %d trades (< %d) -> muestra insuficiente para conclusiones robustas",
            stats.num_trades, BACKTEST.min_trades_for_significance,
        )

    out_path = RESULTS_DIR / "pairs_btc_eth_backtest.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)
    logger.info("%s", json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    run_pairs_backtest()
