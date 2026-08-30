"""Configuración central del sistema de análisis (pares, indicadores, riesgo, backtest)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrendConfig:
    ema_fast: int = 50
    ema_slow: int = 100
    adx_period: int = 14
    # 25 fue el mejor valor encontrado en la búsqueda train/test de 2026-08-29 (ver FINDINGS.md).
    # No está validado como rentable, es el punto de partida menos malo para retomar.
    adx_threshold: float = 25.0


@dataclass(frozen=True)
class PullbackConfig:
    # Capa de timing: comprar el retroceso a la EMA rápida dentro de una tendencia ya
    # confirmada, en vez de perseguir un cruce de osciladores (RSI/MACD no mostró ventaja
    # robusta en la validación train/test — ver resultados de la sesión anterior).
    lookback: int = 3  # nº de velas hacia atrás en las que se admite que el low haya tocado la EMA (mejor valor encontrado)
    require_bullish_candle: bool = True  # exige close > open en la vela de la señal


@dataclass(frozen=True)
class VolumeConfig:
    volume_ma_period: int = 20


@dataclass(frozen=True)
class RegimeConfig:
    # Interruptor global: bloquea TODA señal (larga o corta) cuando la volatilidad relativa
    # reciente (ATR/precio, comparado contra su propio percentil histórico) está en el tramo
    # más alto — apunta a correcciones/crashes de alta volatilidad no direccional, en vez de
    # pedirle al SL/TP que absorba ese ruido (ver intento 10/11 en FINDINGS.md).
    lookback_bars: int = 2160  # ~90 días en velas de 1h
    max_volatility_percentile: float = 0.75


@dataclass(frozen=True)
class RiskConfig:
    atr_period: int = 14
    swing_lookback: int = 5
    # SL/TP calibrados con el análisis de MAE/MFE de src/analyze_excursions.py: la excursión
    # adversa mediana antes de que el precio siga a favor es de ~3.5-4x ATR, muy por encima del
    # 1.5x original. sl_atr_mult=3.0 / tp_atr_mult=8.0 fue la mejor combinación encontrada en el
    # grid search train/test de 2026-08-29 (ver FINDINGS.md) — sigue sin cruzar a expectancy
    # positiva en ambos pares a la vez, no está validada como rentable.
    sl_atr_mult: float = 3.0
    tp_atr_mult: float = 8.0
    # % de equity arriesgado por trade, usado SOLO para poder correr el backtest.
    # El sizing definitivo (Kelly fraccionado) se calcula después con el win rate/RR reales.
    backtest_risk_per_trade: float = 0.01


@dataclass(frozen=True)
class BacktestConfig:
    exchange_id: str = "binance"
    timeframe: str = "1h"
    years_of_history: int = 2
    initial_cash: float = 10_000.0
    commission: float = 0.001  # 0.1% taker fee, Binance spot
    min_trades_for_significance: int = 30
    monte_carlo_iterations: int = 2000
    # Debe coincidir con el fractional_unit pasado a FractionalBacktest: esta última reescala
    # solo Open/High/Low/Close/Volume, así que las columnas de precio propias (atr, swing_low)
    # se reescalan a mano en backtest.py para no desalinearse con Close dentro de la estrategia.
    fractional_unit: float = 1e-8


@dataclass(frozen=True)
class StrategyConfig:
    trend: TrendConfig = field(default_factory=TrendConfig)
    pullback: PullbackConfig = field(default_factory=PullbackConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)


PAIRS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
STRATEGY = StrategyConfig()
BACKTEST = BacktestConfig()


@dataclass(frozen=True)
class PairsConfig:
    """Pairs trading BTC/ETH: opera el spread entre ambas monedas, no la dirección de cada una
    por separado. Market-neutral — no depende de que el mercado suba o baje, aprovecha
    directamente la alta correlación entre BTC/ETH que `contexto proyecto trading.md` marcaba
    como riesgo de diversificación, convirtiéndola en la fuente de la señal.

    Ver FINDINGS.md: la estrategia de seguimiento de tendencia (EMA/ADX/pullback, con o sin
    cortos) fue descartada tras 11 intentos fallidos de validación train/test. Esta es una línea
    de trabajo nueva e independiente, no una variación de esa.
    """

    symbol_a: str = "BTC/USDT"
    symbol_b: str = "ETH/USDT"
    # Más largo que BACKTEST.years_of_history (2): con solo 2 años, el tramo de test del
    # split 70/30 generaba apenas 5 trades — muestra insuficiente para concluir nada.
    years_of_history: int = 4
    beta_window: int = 720  # ~30 días en velas de 1h: ventana de la regresión rodante (hedge ratio)
    z_window: int = 480  # ~20 días en velas de 1h: ventana de media/desvío del spread
    entry_z: float = 2.0  # abre posición cuando |z| supera este umbral
    exit_z: float = 0.5  # cierra cuando |z| revierte por debajo de este umbral (spread converge)
    stop_z: float = 3.5  # cierra en pérdida si |z| sigue divergiendo más allá de este umbral
    risk_per_trade: float = 0.02  # % de equity arriesgado por trade (ambas patas combinadas)


PAIRS_STRATEGY = PairsConfig()


@dataclass(frozen=True)
class BollingerConfig:
    period: int = 20
    num_std: float = 2.0


@dataclass(frozen=True)
class RsiExtremeConfig:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


@dataclass(frozen=True)
class MeanRevRiskConfig:
    atr_period: int = 14
    # SL más allá de la banda tocada, no en la banda misma: si el precio sigue rompiendo después
    # de tocarla, la apuesta de reversión ya está invalidada.
    sl_atr_mult: float = 1.5
    backtest_risk_per_trade: float = 0.01


@dataclass(frozen=True)
class MeanReversionConfig:
    """Reversión a la media: apuesta a que el precio vuelve a la banda central (SMA) tras tocar
    un extremo (Bollinger) confirmado por RSI en sobreventa/sobrecompra y volumen real — hipótesis
    opuesta a la de seguimiento de tendencia (línea 1, descartada) y no depende de una relación
    estable entre dos activos (línea 2, pairs trading, también descartada). Ver FINDINGS.md.
    """

    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    rsi: RsiExtremeConfig = field(default_factory=RsiExtremeConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    risk: MeanRevRiskConfig = field(default_factory=MeanRevRiskConfig)


MEANREV_STRATEGY = MeanReversionConfig()


@dataclass(frozen=True)
class DonchianConfig:
    period: int = 20  # nº de velas para el canal (ruptura de máximo/mínimo de N velas)


@dataclass(frozen=True)
class BreakoutRiskConfig:
    atr_period: int = 14
    # Multiplicadores anchos desde el arranque: la línea 1 (ver FINDINGS.md, análisis MAE/MFE)
    # mostró que en BTC/ETH 1h la excursión adversa mediana antes de que el precio siga a favor
    # es de ~3.5-4x ATR — un SL ajustado solo genera stop-outs por ruido, no por señal invalidada.
    sl_atr_mult: float = 3.0
    tp_atr_mult: float = 6.0
    backtest_risk_per_trade: float = 0.01


@dataclass(frozen=True)
class BreakoutConfig:
    """Breakout de volatilidad: entra en la ruptura del canal de Donchian (máximo/mínimo de N
    velas) confirmada por volumen — apuesta a capturar el inicio real de un impulso, sin
    anticiparlo (momentum, línea 1 descartada) ni esperar un retroceso (pullback, línea 1) ni
    apostar a que revierta (línea 3, descartada). Cuarta y última hipótesis de mercado
    considerada en la sesión. Ver FINDINGS.md.
    """

    donchian: DonchianConfig = field(default_factory=DonchianConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    risk: BreakoutRiskConfig = field(default_factory=BreakoutRiskConfig)


BREAKOUT_STRATEGY = BreakoutConfig()


@dataclass(frozen=True)
class FundingConfig:
    """Estrategia contraria basada en funding rate extremo: cuando el funding lleva sostenido en
    un percentil extremo de su propia historia (posicionamiento de mercado muy sesgado a un lado),
    apuesta a que revierte — una fuente de información (posicionamiento derivado de futuros) nunca
    probada en las 4 líneas anteriores, todas basadas solo en precio/volumen. Ver FINDINGS.md.
    """

    lookback_periods: int = 270  # ~90 días (funding cada 8h, 3/día)
    extreme_percentile: float = 0.90  # top/bottom 10% de su propia historia reciente
    atr_period: int = 14
    sl_atr_mult: float = 3.0
    tp_atr_mult: float = 6.0
    backtest_risk_per_trade: float = 0.01
    volume: VolumeConfig = field(default_factory=VolumeConfig)


FUNDING_STRATEGY = FundingConfig()


@dataclass(frozen=True)
class MLConfig:
    """Clasificador (regresión logística, regularizada) sobre un set de features técnicas
    combinadas — en vez de una regla fija tipo 'si A y B y C', deja que el modelo aprenda los
    pesos. Etiquetado triple-barrera (ver ml_labels.py): para cada barra, simula un long y un
    short hipotéticos con el mismo SL/TP vía ATR de siempre, y aprende a predecir cuál habría
    ganado. Reentrenado en cada fold del walk-forward — nunca usa datos futuros para entrenar.
    """

    sl_atr_mult: float = 2.5
    tp_atr_mult: float = 5.0
    max_holding_bars: int = 72  # ~3 días en velas de 1h
    # Umbral RELATIVO (percentil de las probabilidades predichas en train), no un valor fijo:
    # con una tasa base de acierto ~37-38%, un umbral fijo como 0.55 resultó inalcanzable (el
    # modelo nunca predijo más de ~0.545) — eso daba 0 trades sin decir nada sobre si el modelo
    # tiene alguna señal real en su extremo de mayor confianza.
    top_fraction: float = 0.10  # opera solo el 10% de mayor confianza predicha (por lado)
    min_train_samples: int = 200
    risk_per_trade: float = 0.02


ML_STRATEGY = MLConfig()
