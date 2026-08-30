"""Random Forest / HistGradientBoosting sobre el mismo set de features y labeling triple-barrera
que el clasificador logístico (`ml_strategy.py`), con selección de hiperparámetros ANIDADA: cada
fold de train se separa en un tramo de ajuste (cronológicamente primero) y uno de validación
interna (el más reciente), nunca al revés — así el grid de hiperparámetros se penaliza por cómo
generaliza a datos que el modelo no vio al ajustarse, no por qué tan bien memoriza el propio fit.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import MLConfig
from src.ml_cv import purged_time_series_splits
from src.ml_features import FEATURE_COLUMNS
from src.ml_strategy import ModelBundle, simulate

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Grids deliberadamente conservadores (árboles poco profundos, hojas grandes, learning rate bajo)
# — la penalización real contra el sobreajuste: dado el tamaño de muestra efectivo (velas de 1h
# autocorrelacionadas), un árbol profundo o muchas iteraciones de boosting memorizan ruido.
PARAM_GRID: tuple[dict, ...] = (
    tuple(
        {"family": "rf", "max_depth": d, "min_samples_leaf": leaf, "n_estimators": 200}
        for d in (3, 5) for leaf in (30, 75)
    )
    + tuple(
        {"family": "hgb", "max_depth": d, "learning_rate": lr, "max_iter": 150}
        for d in (2, 3) for lr in (0.02, 0.05)
    )
)


def _build_model(params: dict):
    if params["family"] == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        min_samples_leaf=50,
        random_state=42,
    )


def _fit_bundle(labeled_df: pd.DataFrame, params: dict, cfg: MLConfig) -> ModelBundle | None:
    X = labeled_df[FEATURE_COLUMNS].to_numpy()
    y_long = (labeled_df["long_pnl_pct"] > 0).to_numpy()
    y_short = (labeled_df["short_pnl_pct"] > 0).to_numpy()

    if y_long.sum() < 5 or (~y_long).sum() < 5 or y_short.sum() < 5 or (~y_short).sum() < 5:
        return None

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model_long = _build_model(params).fit(X_scaled, y_long)
    model_short = _build_model(params).fit(X_scaled, y_short)

    prob_long = model_long.predict_proba(X_scaled)[:, 1]
    prob_short = model_short.predict_proba(X_scaled)[:, 1]
    threshold_long = float(np.quantile(prob_long, 1 - cfg.top_fraction))
    threshold_short = float(np.quantile(prob_short, 1 - cfg.top_fraction))

    return ModelBundle(
        scaler=scaler, model_long=model_long, model_short=model_short,
        threshold_long=threshold_long, threshold_short=threshold_short,
    )


def train_tree_classifiers(
    train_labeled: pd.DataFrame,
    cfg: MLConfig,
    min_val_trades: int = 10,
    val_fraction: float = 0.2,
) -> ModelBundle | None:
    """Prueba cada combinación de `PARAM_GRID`, ajustando SOLO en el tramo de ajuste (más
    antiguo) y midiendo expectancy en el tramo de validación interna (más reciente, nunca visto
    al ajustar). La combinación ganadora se reentrena sobre el fold de train COMPLETO — recién
    ahí se usan los datos de validación interna para entrenar, ya sin influir en qué combinación
    se eligió.
    """
    n = len(train_labeled)
    split = int(n * (1 - val_fraction))
    fit_part, val_part = train_labeled.iloc[:split], train_labeled.iloc[split:]

    if len(fit_part) < cfg.min_train_samples or len(val_part) < min_val_trades * 3:
        return None

    best_params, best_score = None, float("-inf")
    for params in PARAM_GRID:
        bundle = _fit_bundle(fit_part, params, cfg)
        if bundle is None:
            continue
        val_result = simulate(val_part, bundle, cfg)
        if val_result.num_trades < min_val_trades:
            continue
        if val_result.expectancy_pct > best_score:
            best_score, best_params = val_result.expectancy_pct, params

    if best_params is None:
        return None

    return _fit_bundle(train_labeled, best_params, cfg)


_NEG_SENTINEL = -1e6  # peor puntaje posible para Optuna; evita -inf, que algunos samplers manejan mal


def _suggest_params(trial: optuna.Trial) -> dict:
    family = trial.suggest_categorical("family", ["rf", "hgb"])
    if family == "rf":
        return {
            "family": "rf",
            "max_depth": trial.suggest_int("rf_max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 10, 150),
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 300, step=50),
        }
    return {
        "family": "hgb",
        "max_depth": trial.suggest_int("hgb_max_depth", 2, 6),
        "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("hgb_max_iter", 50, 300, step=50),
    }


def _cv_score(
    train_labeled: pd.DataFrame, params: dict, cfg: MLConfig,
    n_cv_splits: int, embargo: int, min_val_trades: int,
) -> float:
    """Peor expectancy entre los pliegues de CV purgada — no el promedio: un hiperparámetro que
    funciona bien en 2 de 3 pliegues y muy mal en el tercero no es más confiable que uno mediocre
    mais consistente. Mismo criterio de robustez usado en toda la sesión (línea 1, calibración
    compartida BTC/ETH: "el peor caso entre símbolos, no el promedio").
    """
    splits = purged_time_series_splits(len(train_labeled), n_cv_splits, embargo)
    if not splits:
        return _NEG_SENTINEL

    scores = []
    for train_idx, val_idx in splits:
        bundle = _fit_bundle(train_labeled.iloc[train_idx], params, cfg)
        if bundle is None:
            return _NEG_SENTINEL
        val_result = simulate(train_labeled.iloc[val_idx], bundle, cfg)
        if val_result.num_trades < min_val_trades:
            return _NEG_SENTINEL
        scores.append(val_result.expectancy_pct)

    return float(np.min(scores)) if scores else _NEG_SENTINEL


def _params_from_optuna(raw: dict) -> dict:
    if raw["family"] == "rf":
        return {
            "family": "rf", "max_depth": raw["rf_max_depth"],
            "min_samples_leaf": raw["rf_min_samples_leaf"], "n_estimators": raw["rf_n_estimators"],
        }
    return {
        "family": "hgb", "max_depth": raw["hgb_max_depth"],
        "learning_rate": raw["hgb_learning_rate"], "max_iter": raw["hgb_max_iter"],
    }


def train_tree_classifiers_optuna(
    train_labeled: pd.DataFrame,
    cfg: MLConfig,
    n_trials: int = 25,
    n_cv_splits: int = 3,
    min_val_trades: int = 8,
    seed: int = 42,
) -> ModelBundle | None:
    """Busca hiperparámetros con Optuna (TPE), puntuando cada intento con CV purgada (peor caso
    entre pliegues) en vez de un único split 80/20 — más robusto y explora un espacio de
    hiperparámetros mucho más amplio que el grid manual de 8 combinaciones. El embargo entre
    train y validación de cada pliegue es `cfg.max_holding_bars`: el mismo horizonte que usa el
    labeling triple-barrera, para que ninguna etiqueta de train mire hacia dentro de la
    validación.
    """
    embargo = cfg.max_holding_bars

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        return _cv_score(train_labeled, params, cfg, n_cv_splits, embargo, min_val_trades)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_value is None or study.best_value <= _NEG_SENTINEL:
        return None

    best_params = _params_from_optuna(study.best_params)
    return _fit_bundle(train_labeled, best_params, cfg)
