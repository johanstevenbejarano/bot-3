"""Prueba estadística directa de estacionalidad (hora del día / día de la semana) sobre BTC y ETH.

Con 24 horas + 7 días = 31 grupos probados, un intervalo de confianza al 95% sin corregir daría
~1.55 "positivos" esperados por puro azar (31 * 0.05). Por eso el criterio de éxito no es "algún
grupo excluye cero en un símbolo" -- es que el MISMO grupo excluya cero, con el MISMO signo, en
BTC y ETH a la vez (misma disciplina de "ambos pares" que el resto de las líneas). Que dos
símbolos coincidan por puro azar en el mismo grupo y signo es mucho menos probable (~0.05*0.05
por grupo, sin contar que BTC/ETH ni siquiera son independientes entre sí por su alta
correlación, lo que hace este criterio conservador, no exacto).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src.config import BACKTEST, PAIRS, RESULTS_DIR
from src.data_fetch import fetch_ohlcv
from src.seasonality_analysis import analyze_by_day_of_week, analyze_by_group, analyze_by_hour, compute_forward_returns
from src.validation import train_test_split_by_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

YEARS_OF_HISTORY = 8
MIN_SAMPLES = 50
TRAIN_FRACTION = 0.7
# Comisión taker ida y vuelta (entrar + salir) en Binance spot -- ver BACKTEST.commission.
ROUND_TRIP_COST_PCT = BACKTEST.commission * 2 * 100

DOW_NAMES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def _find_cross_symbol_candidates(results_by_symbol: dict[str, dict[int, dict]]) -> list[dict]:
    symbols = list(results_by_symbol.keys())
    if len(symbols) < 2:
        return []

    common_groups = set(results_by_symbol[symbols[0]].keys())
    for s in symbols[1:]:
        common_groups &= set(results_by_symbol[s].keys())

    candidates = []
    for group in sorted(common_groups):
        per_symbol = {s: results_by_symbol[s][group] for s in symbols}
        if not all(r["excludes_zero"] for r in per_symbol.values()):
            continue
        signs = {s: (r["mean"] > 0) for s, r in per_symbol.items()}
        if len(set(signs.values())) != 1:
            continue  # excluyen cero pero en direcciones distintas -> no cuenta
        candidates.append({"group": group, "per_symbol": {s: r for s, r in per_symbol.items()}})

    return candidates


def _test_set_stats(test_by_symbol: dict, group: int, kind: str) -> dict:
    """Recalcula el bootstrap CI de un grupo puntual (una hora o un día) pero SOLO sobre el
    tramo de test -- nunca se miró al buscar los candidatos en train."""
    per_symbol = {}
    for symbol, raw in test_by_symbol.items():
        forward_returns = compute_forward_returns(raw["close"])
        group_key = (
            pd.Series(raw.index.hour, index=raw.index)
            if kind == "hour"
            else pd.Series(raw.index.dayofweek, index=raw.index)
        )
        result = analyze_by_group(forward_returns, group_key, min_samples=MIN_SAMPLES)
        per_symbol[symbol] = result.get(group, {"n": 0, "mean": float("nan"), "excludes_zero": False})
    return per_symbol


def main() -> None:
    raw_by_symbol = {
        symbol: fetch_ohlcv(symbol, timeframe=BACKTEST.timeframe, years=YEARS_OF_HISTORY)
        for symbol in PAIRS
    }
    train_by_symbol = {}
    test_by_symbol = {}
    for symbol, raw in raw_by_symbol.items():
        train_df, test_df = train_test_split_by_time(raw, TRAIN_FRACTION)
        train_by_symbol[symbol] = train_df
        test_by_symbol[symbol] = test_df
        logger.info(
            "%s: train %s -> %s (%d velas), test %s -> %s (%d velas)",
            symbol, train_df.index[0], train_df.index[-1], len(train_df),
            test_df.index[0], test_df.index[-1], len(test_df),
        )

    logger.info("Buscando candidatos en TRAIN unicamente (test no se toca todavia)...")
    hour_results_train = {
        symbol: analyze_by_hour(df, min_samples=MIN_SAMPLES) for symbol, df in train_by_symbol.items()
    }
    dow_results_train = {
        symbol: analyze_by_day_of_week(df, min_samples=MIN_SAMPLES) for symbol, df in train_by_symbol.items()
    }

    hour_candidates = _find_cross_symbol_candidates(hour_results_train)
    dow_candidates = _find_cross_symbol_candidates(dow_results_train)

    logger.info(
        "TRAIN: %d candidatos de hora, %d candidatos de día (excluyen cero y coinciden en signo en ambos símbolos)",
        len(hour_candidates), len(dow_candidates),
    )

    def _evaluate(candidates: list[dict], kind: str, label_fn) -> list[dict]:
        out = []
        for c in candidates:
            group = c["group"]
            test_stats = _test_set_stats(test_by_symbol, group, kind)
            holds_direction = all(
                test_stats[s].get("n", 0) >= MIN_SAMPLES and
                (test_stats[s]["mean"] > 0) == (c["per_symbol"][s]["mean"] > 0)
                for s in c["per_symbol"]
            )
            still_significant = all(test_stats[s].get("excludes_zero", False) for s in test_stats)
            max_abs_mean = max(abs(r["mean"]) for r in c["per_symbol"].values())
            beats_costs = max_abs_mean > ROUND_TRIP_COST_PCT

            entry = {
                "label": label_fn(group),
                "train": c["per_symbol"],
                "test": test_stats,
                "holds_direction_out_of_sample": holds_direction,
                "still_significant_out_of_sample": still_significant,
                "max_abs_mean_pct": max_abs_mean,
                "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
                "economically_viable_before_slippage": beats_costs,
            }
            logger.info(
                "%s: train->test %s | %s | efecto %.4f%% vs costo %.4f%%: %s",
                entry["label"],
                "sostiene dirección" if holds_direction else "NO sostiene dirección",
                "sigue significativo" if still_significant else "ya no es significativo en test",
                max_abs_mean, ROUND_TRIP_COST_PCT,
                "supera el costo" if beats_costs else "NO supera el costo de operar",
            )
            out.append(entry)
        return out

    hour_evaluated = _evaluate(hour_candidates, "hour", lambda g: f"{g:02d}:00 UTC")
    dow_evaluated = _evaluate(dow_candidates, "dow", lambda g: DOW_NAMES[g])

    any_fully_validated = any(
        e["holds_direction_out_of_sample"] and e["still_significant_out_of_sample"] and e["economically_viable_before_slippage"]
        for e in hour_evaluated + dow_evaluated
    )

    report = {
        "years_of_history": YEARS_OF_HISTORY,
        "min_samples": MIN_SAMPLES,
        "train_fraction": TRAIN_FRACTION,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "hour_candidates": hour_evaluated,
        "dow_candidates": dow_evaluated,
        "status": "validated" if any_fully_validated else "failed_out_of_sample_or_uneconomical",
    }

    out_path = RESULTS_DIR / "seasonality_validation.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Reporte guardado en %s", out_path)
    logger.info(
        "%s",
        "AL MENOS UN CANDIDATO sostiene fuera de muestra y supera costos" if any_fully_validated
        else "Ningún candidato sostiene fuera de muestra Y supera el costo de operar a la vez",
    )


if __name__ == "__main__":
    main()
