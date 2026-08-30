# Hallazgos del backtest de trading BTC/ETH (sesión 2026-08-29)

Estado: **pausado a propósito, sin ventaja estadísticamente significativa encontrada en ninguna
línea**. Un clasificador (RF/GB + Optuna + CV purgada) dio agregados positivos en dos walk-forwards
de BTC, pero un intervalo de confianza bootstrap sobre los retornos trade-por-trade (sección
"Tercera verificación", la más importante si se retoma esto) mostró que el intervalo incluye cero
en BTC y en ETH — no es distinguible de ruido con rigor estadístico. Este documento existe para
retomar sin perder el contexto de lo ya descartado — **no repetir los caminos fallidos sin una
razón nueva y concreta**.

## Resumen para retomar rápido

14 intentos entre 4 líneas de estrategia completamente distintas, cada una con causa raíz
identificada:

1. **Seguimiento de tendencia** (EMA/ADX + momentum/pullback + volumen, 11 intentos): funciona
   en train, se revierte en test — no sobrevive el cambio de régimen entre el bull market de
   entrenamiento y la corrección de prueba. Ni calibración de SL/TP, ni rediseño de entrada, ni
   calibración por símbolo, ni permitir cortos, ni un filtro de régimen de volatilidad lo arreglaron.
2. **Pairs trading BTC/ETH**: BTC y ETH no formaron una relación estable/mean-reverting en
   2022-2026 — hubo una tendencia estructural de +208% en el ratio. Apostar a la reversión del
   spread es apostar contra una tendencia real, no contra ruido.
3. **Reversión a la media** (Bollinger + RSI en extremos): 0/36 combinaciones de la búsqueda de
   parámetros dan expectancy positiva en train — rechazo uniforme en todo el rango explorado, no
   un problema de calibración fina.
4. **Breakout de volatilidad** (canal de Donchian + volumen): 0/48 combinaciones positivas en
   train, mismo rechazo uniforme que la línea 3 — ni el mejor caso (-0.15% en el peor par) se
   acerca al punto de equilibrio.

**Las 4 hipótesis de mercado consideradas al inicio de la sesión ya se probaron y fallaron.**
Ver "Recomendación final" al cierre del documento antes de intentar una quinta.

## Qué se construyó

Infraestructura común (reutilizada por las 3 líneas):
- `src/data_fetch.py` — descarga y cachea OHLCV de Binance vía `ccxt`, con reintentos y relleno
  incremental **en ambas direcciones** (hacia adelante para velas nuevas, hacia atrás si se pide
  más historia de la que ya hay cacheada — bug corregido en la línea 2, ver abajo).
- `src/monte_carlo.py` — simulación de Monte Carlo (reordena trades) para estimar rango de drawdown.
- `src/validation.py` — `train_test_split_by_time`, utilidad genérica reusada por las 3 líneas.
- `tests/` — 16 tests unitarios, todos pasando (`pytest -q`).

### Línea 1 — Seguimiento de tendencia (`src/config.py: StrategyConfig`)
`indicators.py`, `strategy.py` (`ConfluenceStrategy`, largo+corto), `backtest.py`,
`excursion_analysis.py` + `analyze_excursions.py` (calibración de SL/TP por MAE/MFE),
`train_test_validate.py`, `train_test_validate_per_symbol.py`.
```
.\.venv\Scripts\python.exe -m src.backtest                    # backtest simple, período completo
.\.venv\Scripts\python.exe -m src.analyze_excursions           # MAE/MFE sobre train
.\.venv\Scripts\python.exe -m src.train_test_validate          # grid search en train + validación en test
.\.venv\Scripts\python.exe -m src.train_test_validate_per_symbol
```

### Línea 2 — Pairs trading (`src/config.py: PairsConfig`)
`pairs_strategy.py` (spread vía hedge ratio con regresión rodante + z-score, motor de backtest a
medida), `pairs_backtest.py`, `pairs_validate.py`.
```
.\.venv\Scripts\python.exe -m src.pairs_backtest
.\.venv\Scripts\python.exe -m src.pairs_validate
```

### Línea 3 — Reversión a la media (`src/config.py: MeanReversionConfig`)
`meanrev_indicators.py`, `meanrev_strategy.py` (`MeanReversionStrategy`, largo+corto),
`meanrev_backtest.py`, `meanrev_validate.py`.
```
.\.venv\Scripts\python.exe -m src.meanrev_backtest
.\.venv\Scripts\python.exe -m src.meanrev_validate
```

### Línea 4 — Breakout de volatilidad (`src/config.py: BreakoutConfig`)
`breakout_indicators.py`, `breakout_strategy.py` (`BreakoutStrategy`, largo+corto),
`breakout_backtest.py`, `breakout_validate.py`.
```
.\.venv\Scripts\python.exe -m src.breakout_backtest
.\.venv\Scripts\python.exe -m src.breakout_validate
```

---

## Línea 1: seguimiento de tendencia (EMA/ADX + pullback/momentum + volumen)

**Bug real encontrado y corregido:** `backtesting.py` opera en unidades enteras por defecto. Con
BTC a $78k+, el sizing fraccionario (1% de equity en riesgo) se redondeaba a 0 unidades → 0
trades en todo el período. Se corrigió usando `backtesting.lib.FractionalBacktest`, con una
precaución adicional: esa clase solo reescala `Open/High/Low/Close/Volume`, no columnas propias
— hubo que reescalar `atr`/`swing_low`/`swing_high` a mano (`BACKTEST.fractional_unit`) para que
no quedaran desalineadas del precio dentro de la estrategia.

**Cronología de intentos** (BTC/USDT y ETH/USDT, 1h, ~2 años):

1. **Baseline** (documento original: EMA50/100+ADX20, RSI45+MACD cruce, SL=swing_low-1.5×ATR,
   TP=+2.5×ATR): expectancy negativa en ambos pares (-0.42% BTC, -1.02% ETH).
2. **Ajuste de SL/TP** (ancla en entrada vs. swing_low, TP hasta 6×ATR): mejora pero sin cruzar a
   positivo. Mejor caso: BTC casi neutro (+1.0% return, Sharpe +0.11), ETH sigue negativo.
3. **Grid search de thresholds sobre el período completo** (48 combos): solo 2/48 daban
   expectancy positiva en ambos pares, con ventaja marginal (+0.1-0.3%) e inestable ante cambios
   pequeños de parámetros — señal de sobreajuste, no de ventaja real.
4. **Validación train/test introducida** (70/30, split por fecha, búsqueda solo en train,
   exigiendo expectancy>0 en AMBOS pares): **0/108 combinaciones** del diseño RSI/MACD sobreviven
   ni siquiera en train.
5. **Rediseño de entrada: pullback a EMA** (reemplaza el cruce de RSI/MACD por "retrocedió a
   tocar la EMA50 en las últimas N velas y cerró por encima con vela alcista"): **0/54
   combinaciones** sobreviven en train tampoco. Mejor caso: peor-par expectancy ~-0.40%.
6. **Diagnóstico buy & hold**: durante el train (set-2024 a ene-2026), BTC hizo +51% y ETH +17%
   simplemente comprando y manteniendo. La estrategia perdía dinero en ese mismo período — señal
   de que el problema no es "falta de señal" sino que el SL corta operaciones sanas antes de que
   se desarrollen (whipsaw). El test (ene-2026 a ago-2026) fue de corrección (-13% BTC, -17%
   ETH): el régimen cambió por completo entre train y test.
7. **Análisis MAE/MFE** (`src/analyze_excursions.py`, sobre train): la excursión adversa mediana
   antes de que el precio siga a favor es de **~3.5-4× ATR**, muy por encima del 1.5× original.
   La excursión favorable mediana es ~3-4× ATR, con percentil 75 en ~7× ATR — sí hay upside real
   si el SL sobrevive el ruido inicial.
8. **Recalibración con SL/TP más anchos** (SL 3.0-4.5×ATR, TP 4-8×ATR, ancla en entrada, grid de
   180 combos en train): mejor resultado de toda la línea — BTC casi neutro (ADX>25, lookback=3,
   SL=3×ATR, TP=8×ATR → expectancy -0.16%, return +4.0%), pero ETH se mantiene negativo (-0.23%
   expectancy, -0.97% return). Ninguna combinación cruza a positivo en ambos pares a la vez.
9. **Calibración independiente por símbolo** (sin exigir una config compartida): con
   `min_trades=15`, BTC "ganaba" en train (+0.75%) pero con solo 15 trades y win rate 26.7%
   (apalancado por TP=10×ATR, ganancias raras) — sobreajuste por muestra pequeña; en test, win
   rate cae a 11%, expectancy -0.73%. Con `min_trades=25`: mejora la calidad pero el patrón se
   mantiene (BTC train +0.18%→test -0.99%; ETH train +0.30%→test -0.10%).
10. **Permitir cortos** (`ConfluenceStrategy` largo+corto, capas espejo para tendencia bajista y
    rechazo en rally): por primera vez, TRAIN con ambos pares positivos y muestra sólida (BTC
    +0.156%/58 trades/Sharpe 0.96; ETH +0.749%/58 trades). TEST: ambos se revierten a negativo
    (BTC -0.55%, ETH -0.79%). Ni con cortos durante la corrección se sostiene fuera de muestra.
11. **Filtro de régimen de volatilidad** (`atr_percentile` vs. su propia historia de ~90 días,
    bloquea toda señal en el 25% de mayor volatilidad relativa): **empeora los resultados**. El
    ATR no distingue direccionalidad — un rally fuerte (volatilidad "buena") y un crash
    (volatilidad "mala") generan el mismo percentil alto, así que el filtro bloquea trades
    rentables junto con los dañinos. Mejor caso con el filtro puesto: -0.32% (peor que sin él).

**Conclusión de la línea 1:** cada cambio estructural razonable dentro de esta familia de
estrategia (confluencia técnica de 3 capas, SL/TP vía ATR, con o sin cortos, con o sin filtro de
régimen) falló el mismo criterio train/test. El patrón — funciona en train, se revierte en test —
es consistente con que el train fue un bull market y el test una corrección: la estrategia no
tiene ventaja que sobreviva ese cambio de régimen. `config.py: StrategyConfig` quedó con los
valores menos malos encontrados (`adx_threshold=25`, `sl_atr_mult=3.0`, `tp_atr_mult=8.0`,
`sl_anchor="entry"`), documentados como **no validados como rentables**, solo como punto de
partida si se retoma esta línea.

---

## Línea 2: pairs trading BTC/ETH

A petición explícita del usuario, se abandonó la línea 1 y se construyó una estrategia
completamente distinta, sin reutilizar nada de su código: operar el **spread** entre BTC y ETH
(comprar la rezagada, vender la adelantada cuando la relación se desvía de su media) en vez de
apostar a la dirección de cada moneda por separado — market-neutral, aprovechando directamente la
alta correlación que `contexto proyecto trading.md` marcaba como riesgo de diversificación.

Motor de backtest a medida (`pairs_strategy.py`) porque `backtesting.py` no soporta dos activos
simultáneos: hedge ratio vía regresión rodante (cov/var, sin lookahead) entre `log(BTC)` y
`log(ETH)`, z-score del spread resultante, entrada en `|z| >= entry_z`, salida en reversión
(`|z| <= exit_z`) o stop (`|z| >= stop_z`, la divergencia siguió creciendo).

**Bug de infraestructura encontrado y corregido:** `data_fetch.fetch_ohlcv` solo extendía el
caché hacia adelante (velas nuevas), nunca hacia atrás — pedir más años de historia con datos ya
cacheados no traía historia más antigua. Se corrigió para rellenar en ambas direcciones
(`_fetch_range` reutilizable). Afecta a cualquier pipeline que use `fetch_ohlcv`.

**Resultado:**
- Con 2 años de historia: el test solo generó 5 trades — muestra insuficiente para concluir
  nada. Se amplió a 4 años (`PairsConfig.years_of_history`).
- Con 4 años (35,039 velas, ago-2022 a ago-2026): TRAIN con la mejor config encontrada
  (beta_window=2160, z_window=240, entry_z=1.5, exit_z=0.5, stop_z=3.5) da expectancy
  prácticamente nula (+0.07%, Sharpe 0.01, 108 trades). TEST: -0.34% (50 trades) → no sostiene.
- **Diagnóstico de causa raíz:** el ratio BTC/ETH subió **+208.6%** durante todo el train
  (13.45 → 41.49), acelerando en la segunda mitad (+127% vs +36% en la primera). 75 de 108
  trades (70%) cerraron por stop, no por reversión. El supuesto central del pairs trading — que
  el spread es estacionario y revierte a su media — **no se cumple para BTC/ETH en este
  período**: hay una tendencia estructural de varios años (BTC ganando dominancia sobre ETH), no
  ruido de corto plazo alrededor de un promedio estable.

**Conclusión:** a diferencia de la línea 1 (problema de calibración), acá el problema es
conceptual — ajustar ventanas o umbrales no arregla apostar contra una tendencia real.

---

## Línea 3: reversión a la media (Bollinger + RSI)

Tercera hipótesis, independiente de las dos anteriores: Bandas de Bollinger (20, 2σ) + RSI(14)
en sobreventa/sobrecompra como gatillo, volumen como confirmación, TP = banda central (`bb_basis`,
el objetivo natural de una apuesta de reversión), SL = ATR más allá del extremo tocado. De un
solo activo, así que reutiliza `backtesting.py`/`FractionalBacktest`.

**Bug de sizing encontrado y corregido:** la primera corrida (config por defecto) dio **-95% de
drawdown** en ambos pares con ~1100 trades cada uno. Causa: cuando el ATR es muy pequeño (mercado
en calma), la distancia al SL se vuelve minúscula en precio, y el sizing por riesgo fijo (mismo
patrón que la línea 1) dispara el tamaño de la posición hasta el techo de 99% de equity —
apostando más fuerte justo en las señales más débiles (bandas muy ajustadas = probablemente
ruido). Se corrigió con un techo duro de 20% de equity por trade, independiente del riesgo
calculado. Bajó el drawdown a ~-49%, pero no cambió el signo del resultado.

**Resultado de la búsqueda de parámetros en TRAIN** (36 combinaciones: Bollinger 2.0-3.0σ, RSI
extremos 20/80 a 30/70, SL 1.5-3.0×ATR, exigiendo expectancy positiva en ambos pares): **0/36
combinaciones viables.** Mejor caso (peor de los dos pares): -0.25% expectancy, returns de -2% a
-40% según la combinación. Win rate nunca superó ~45% — notablemente bajo para reversión a la
media, donde se espera win rate alto compensando con TP pequeños (lo contrario de lo observado).

**Conclusión:** a diferencia de la línea 1 (varias combinaciones cerca del punto de equilibrio),
acá el resultado es uniformemente negativo en todo el rango explorado — un rechazo más
contundente. Bollinger+RSI simple tampoco tiene ventaja demostrable en BTC/ETH 1h en este período.

---

## Línea 4: breakout de volatilidad (canal de Donchian)

Cuarta y última hipótesis de mercado considerada en la sesión: entrar en la ruptura del
máximo/mínimo de las N velas previas (canal desplazado 1 vela para no incluirse a sí mismo),
confirmada por volumen — captura el inicio de un impulso en vez de anticiparlo (línea 1,
momentum), esperar un retroceso (línea 1, pullback) o apostar a que revierta (línea 3). SL/TP
simétricos vía ATR, con multiplicadores ya anchos desde el arranque (3×/6×, informados por el
análisis MAE/MFE de la línea 1) y el mismo techo duro de 20% de equity por trade aprendido en la
línea 3. Código: `breakout_indicators.py`, `breakout_strategy.py`, `breakout_backtest.py`,
`breakout_validate.py`, `BreakoutConfig` en `config.py`.

**Resultado de la búsqueda de parámetros en TRAIN** (48 combinaciones: canal de 10 a 55 velas,
SL 2-4×ATR, TP 4-10×ATR, exigiendo expectancy positiva en ambos pares): **0/48 combinaciones
viables.** Mejor caso (peor de los dos pares): -0.145% expectancy, con BTC casi neutro en varias
combinaciones (-0.03% a -0.09%) pero ETH consistentemente arrastrando el resultado a negativo.
Mismo patrón de rechazo uniforme que la línea 3 — ninguna combinación se acerca al punto de
equilibrio en ambos pares a la vez.

**Conclusión:** breakout tampoco tiene ventaja demostrable en BTC/USDT y ETH/USDT 1h en este
período.

---

## Chequeo BTC-solo (2026-08-29, misma sesión, a petición del usuario)

Pregunta: dado que BTC se comportó consistentemente mejor que ETH bajo la misma config en las
líneas 1, 3 y 4, ¿alcanza con enfocarse solo en BTC? Para la línea 1 esto ya se había probado
(intento 9: calibración independiente por símbolo) y falló igual en test — no hacía falta
repetirlo. Para las líneas 3 y 4 nunca se había buscado una config específica para BTC sin exigir
nada a ETH, así que sí era una pregunta nueva.

- **Reversión a la media, BTC solo:** sigue sin haber ninguna combinación con expectancy
  positiva en train. No es que ETH diluyera el resultado — BTC solo tampoco tiene ventaja bajo
  esta hipótesis.
- **Breakout, BTC solo:** mejor resultado de TRAIN de toda la sesión — canal=?, expectancy
  +0.379%, 276 trades, Sharpe 0.855, return +23.6% (ver script de chequeo, no promovido a
  pipeline permanente). En TEST se revierte: expectancy -0.178%, 151 trades, return -5.3%.

**Conclusión:** aislar BTC no resuelve el problema de fondo. Confirma que la causa raíz no es
"ETH arrastra a BTC hacia abajo", sino el mismo patrón de toda la sesión — el cambio de régimen
entre train (bull market) y test (corrección) invalida cualquier ventaja encontrada en train,
incluso la mejor (breakout BTC, Sharpe 0.855) — ni aislando el activo que mejor se comportaba
sobrevive la validación fuera de muestra.

## Walk-forward: breakout en BTC con recalibración periódica (2026-08-29, misma sesión)

El chequeo BTC-solo dejó una pregunta abierta: el mejor resultado de train de toda la sesión
(breakout BTC, Sharpe 0.855) se revirtió en un único split 70/30 — pero un solo split solo
compara "un tramo de bull market" contra "un tramo de corrección". ¿Sostiene si se recalibra
periódicamente en vez de fijar los parámetros una sola vez?

Se construyó un motor genérico de walk-forward (`src/walk_forward.py`, con tests) y se aplicó a
breakout/BTC (`src/breakout_walk_forward.py`): 16 folds sobre 5 años de historia (ago-2021 a
ago-2026, 43,797 velas), cada uno recalibrando con los 365 días previos (grid completo de
Donchian/SL/TP) y evaluando en los 90 días siguientes nunca vistos, sin retocar nada según el
resultado. Los folds cubren múltiples regímenes reales: bull 2021, bear 2022, recuperación
2023-2025 — no solo "bull vs. corrección" como el split único anterior.

**Resultado: -10.58% acumulado sobre los 16 tramos de test concatenados** (400 trades en total).
8 de 16 folds individuales fueron positivos (básicamente una moneda al aire por fold), pero las
pérdidas fueron sistemáticamente mayores que las ganancias: peores folds en -6.44%, -5.09%,
-4.44%, -4.04%, -2.17%; mejores folds en +4.27%, +2.79%, +1.20%, +1.16%, +0.99%. Ni siquiera
recalibrando cada 90 días con el método técnicamente correcto para adaptarse a cambios de régimen,
el mejor candidato de toda la sesión sostiene una ventaja neta positiva.

**Conclusión:** esto cierra la pregunta de fondo. No era un problema de "un split de mala suerte"
ni de "elegir mal el activo" (BTC ya era el mejor caso) — con la metodología más rigurosa posible
dentro de lo razonable (recalibración periódica sobre 5 años, múltiples regímenes), el resultado
neto sigue siendo negativo. Insistir con walk-forward sobre las otras 3 líneas (pairs, mean
reversion, tendencia) no es prioritario: todas partían de resultados peores que breakout/BTC en
el split único, así que es improbable que walk-forward las salve donde no salvó al mejor caso.

## Filtro multi-timeframe sobre breakout/BTC (2026-08-29, misma sesión, última prueba)

A pedido del usuario, se probó una hipótesis más antes de cerrar la búsqueda: exigir que la
ruptura de 1h esté a favor de la tendencia diaria (EMA50 diaria) antes de operar — la idea más
fundamentada en lo ya aprendido (el ruido de 1h fue la causa raíz recurrente de toda la sesión).

Código: `src/htf_filter.py` (`add_htf_trend`, con `_shift_and_align` testeado explícitamente para
garantizar que cada barra de 1h solo usa la vela diaria YA CERRADA — sin lookahead), integrado de
forma retrocompatible en `breakout_strategy.compute_layers` (si las columnas `trend_up_htf` /
`trend_dn_htf` no están presentes, el comportamiento es idéntico al de antes). Se corrió el mismo
walk-forward de 16 folds / 5 años (`src/breakout_htf_walk_forward.py`).

**Resultado: empeora, no mejora.** -14.86% acumulado (vs. -10.58% sin el filtro). Solo 5/16 folds
positivos (vs. 8/16 sin filtro), y en 2 folds no se encontró ninguna combinación con expectancy
positiva en absoluto (el filtro, combinado con el mínimo de 20 trades, dejó muy pocas señales en
esos períodos). El filtro multi-timeframe refutó la hipótesis que lo motivaba.

**Conclusión y cumplimiento del compromiso:** esta era la última hipótesis razonable identificada
(la más barata y mejor fundamentada de las tres opciones discutidas — más barata que agregar
funding rate/open interest, y con más base que un quinto patrón técnico arbitrario). Con este
resultado, se cierra la búsqueda de una regla sistemática con edge positivo dentro de este marco.

## Agotando las dos alternativas restantes (2026-08-29, sesión siguiente)

A petición explícita del usuario ("agotar los dos recursos que quedan antes de rendirnos"), se
construyeron y validaron con la misma disciplina (tests, train/test, walk-forward de 5 años) las
dos alternativas identificadas como más prometedoras: una fuente de datos nueva (funding rate) y
un enfoque de modelado distinto (clasificador). Open Interest se descartó de entrada: la API
pública de Binance solo devuelve ~30 días de historia, insuficiente para cualquier backtest
multi-año.

### Funding rate contrario

Hipótesis: cuando el funding rate de BTC/USDT perpetuo lleva sostenido en un percentil extremo de
su propia historia (posicionamiento de mercado muy sesgado a largos o a cortos), apostar a que
revierte. Código: `funding_data.py` (descarga con caché, historia real desde 2019-09),
`funding_indicators.py` (alineación sin lookahead del funding —evento cada 8h— sobre velas de
1h), `funding_strategy.py`, `funding_validate.py`, `funding_walk_forward.py`.

Split simple: train +0.34% (108 trades) → test -0.23% (44 trades) — mismo patrón de siempre.
**Walk-forward de 5 años (15 folds, recalibrando cada 90 días): -4.81% acumulado**, 135 trades de
test en total, 6/12 folds con trades fueron positivos (básicamente 50/50). Negativo, pero el
segundo mejor resultado de walk-forward de toda la sesión.

### Clasificador (regresión logística sobre features combinadas)

Hipótesis: en vez de una regla fija ("si A y B y C"), dejar que un modelo aprenda los pesos sobre
8 features combinadas de todas las líneas anteriores (tendencia, RSI, posición en bandas de
Bollinger, percentil de ATR, ratio de volumen, posición en el canal de Donchian). Etiquetado
triple-barrera (`ml_labels.py`, testeado exhaustivamente: TP antes que SL, SL antes que TP, ambas
barreras en la misma vela — se asume SL primero por conservador —, y barrera de tiempo) para cada
barra y cada dirección. Reentrenado desde cero en cada fold del walk-forward.

**Dos bugs reales encontrados y corregidos en el camino:**
1. Un umbral de probabilidad fijo (0.55) resultó inalcanzable: con una tasa base de acierto de
   ~37%, el modelo nunca predijo más de ~0.545 de probabilidad, dando 0 trades siempre. Se
   corrigió a un umbral relativo (percentil 90 de las probabilidades predichas en el propio
   train) — así el modelo compite con su propio extremo de mayor confianza, no contra un número
   arbitrario.
2. El labeling triple-barrera no descontaba comisiones de trading — a diferencia de todas las
   demás estrategias de la sesión. La primera corrida del walk-forward dio **+0.39% positivo**;
   tras corregir el descuento de comisión (2x, entrada+salida) en el propio labeling (para que el
   modelo aprenda con el costo real incluido, no con una noción optimista de "trade ganador"), el
   resultado correcto es **-1.95%**.

**Walk-forward de 5 años (16 folds, 571 trades de test en total): -1.95% acumulado**, 6/16 folds
positivos. Negativo, pero el resultado **menos malo de toda la sesión** — más cerca del punto de
equilibrio que cualquiera de las otras 6 líneas/variantes probadas (líneas 1-4, breakout+HTF,
funding).

### Balance de las dos alternativas agotadas

Ninguna de las dos sostiene una ventaja positiva fuera de muestra con la validación más rigurosa
disponible (walk-forward de 5 años). El clasificador es el resultado más cercano a cero de toda
la sesión, pero "más cercano a cero" no es lo mismo que "rentable" — sigue siendo negativo neto.
Se cumplió el objetivo del usuario de agotar los recursos disponibles: se probaron 2 fuentes de
datos (precio/volumen, funding rate; open interest descartado por falta de historia) y 3 enfoques
de modelado (reglas técnicas fijas, reglas contrarias sobre posicionamiento, clasificador
estadístico), todos con la misma disciplina de validación.

## Random Forest / Gradient Boosting (2026-08-29, misma sesión)

A pedido del usuario, se probó si un modelo con más capacidad que la regresión logística
(interacciones no lineales entre features) mejoraba el resultado. Código: `ml_tree_strategy.py`
(RandomForestClassifier y HistGradientBoostingClassifier, grid conservador — árboles poco
profundos, hojas grandes, learning rate bajo — para no memorizar ruido), reutilizando el mismo
labeling/features/simulate de `ml_strategy.py`. Selección de hiperparámetros **anidada**: cada
fold de train se separa en un tramo de ajuste (80%, cronológicamente primero) y uno de validación
interna (20%, el más reciente); se elige la combinación con mejor expectancy en la validación
interna (nunca vista al ajustar), y esa combinación ganadora recién se reentrena sobre el fold de
train completo — el mismo principio de walk-forward, aplicado un nivel más adentro, específicamente
para penalizar la complejidad del modelo por generalización real.

**Resultado: peor que la regresión logística, no mejor.** Walk-forward de 5 años (16 folds, 634
trades de test): **-2.92% acumulado**, solo 4/16 folds positivos (vs. 6/16 de la logística,
-1.95%). La firma es clásica de sobreajuste: el expectancy en TRAIN es notablemente más alto que
el de la logística (0.6%-2.1% vs. 0.05%-0.7% por fold) — el modelo con más capacidad memoriza
mejor el pasado, pero generaliza peor al futuro, incluso con la selección anidada diseñada
explícitamente para evitar eso.

**Conclusión:** confirma la hipótesis planteada antes de construirlo — con el tamaño de muestra
efectivo disponible (velas de 1h autocorrelacionadas, no verdaderamente independientes), un
modelo más flexible no tiene margen para aprender señal real adicional; solo tiene margen para
aprender ruido adicional. Este resultado también hace más pesimista la expectativa sobre redes
neuronales o deep learning (que necesitan aún más datos que Random Forest para no sobreajustar) —
no se recomienda intentarlas dado este resultado.

## Optuna + CV purgada + más historia (2026-08-29, misma sesión) — el hallazgo más sólido de la sesión

A pedido del usuario, se llevó el clasificador a su versión más rigurosa: 9 años de historia de
BTC/USDT (Binance spot arranca en 2017-08, el doble de los 5 años usados hasta acá), ventanas de
entrenamiento de 2 años por fold (antes 1 año), y búsqueda de hiperparámetros con **Optuna** (TPE,
25 intentos) puntuada por **validación cruzada purgada con embargo** (`src/ml_cv.py`, testeado:
3 pliegues secuenciales tipo walk-forward interno, con un hueco de `max_holding_bars` entre cada
train y su validación para que ninguna etiqueta triple-barrera del train mire hacia dentro de la
validación) en vez de un único split 80/20. El puntaje de cada intento es el **peor caso entre
pliegues** (mismo criterio de robustez que la línea 1). Código: `ml_cv.py`,
`train_tree_classifiers_optuna` en `ml_tree_strategy.py`, `ml_tree_optuna_walk_forward.py`.

**Primer resultado (14 folds, 5 años→9 años, train=2 años, step=180 días): +0.44% acumulado**,
7/14 folds positivos (exactamente la mitad, no cumple `holds_up` estricto), 510 trades de test.
Es el primer resultado NO NEGATIVO de toda la sesión. Se verificó que la distribución de folds
está razonablemente balanceada (ningún fold individual domina el resultado) y que las comisiones
siguen descontadas correctamente (reutiliza el `prepare_labeled` ya corregido).

**Prueba de robustez pedida antes de aceptar el resultado:** repetir exactamente la misma
metodología (mismos hiperparámetros de búsqueda, mismo train/test/step) pero desplazando el
origen de los folds 75 días (`ml_tree_optuna_walk_forward_shifted.py`) — sin retocar nada del
proceso de búsqueda. Criterio de falsación acordado de antemano: si el resultado se derrumba a
algo claramente negativo, confirma que el +0.44% original era sensible a una elección arbitraria
(dónde arranca el primer fold); si sigue rondando cero o positivo, es evidencia de que no es puro
ruido.

**Resultado desplazado: +0.98% acumulado, 8/14 folds positivos (mayoría) — `holds_up: true`.**
No solo no se derrumbó, sino que mejoró. Es la primera vez en toda la sesión (14+ intentos, 4
líneas de estrategia, funding rate, 3 variantes de clasificador) que dos corridas de walk-forward
independientes, con la misma metodología, dan positivo sin haber tocado ningún hiperparámetro de
búsqueda entre medio.

**Calibración honesta del hallazgo — qué demuestra y qué no:**
- Sí demuestra: el resultado no es puramente un artefacto de dónde arranca el primer fold. Con
  dos orígenes distintos, ambos dan positivo y de magnitud similar.
- No demuestra: una ventaja grande o lista para operar. +0.44%/+0.98% acumulados equivalen a unas
  pocas décimas de porcentaje anualizadas sobre ~3.5 años de exposición de test — lejos de un
  edge sustancial. Los folds individuales siguen teniendo pérdidas notables intercaladas (peor
  fold: -1.14%), no es una curva de equity suave.
- Son solo 2 puntos de datos (2 orígenes de fold), no una prueba estadística formal de que el
  efecto es real y no ruido con suerte en ambas direcciones. Por acuerdo explícito con el
  usuario, no se siguió ajustando la configuración del walk-forward después de esta verificación
  — seguir buscando un tercer/cuarto desplazamiento hasta confirmar sería el mismo tipo de
  sobreajuste que se evitó toda la sesión.

**Conclusión:** de las ~15 combinaciones probadas en la sesión (4 líneas de estrategia técnica,
funding rate, 3 variantes de clasificador), esta es la única que sostuvo un resultado positivo
bajo un chequeo de robustez razonable. No es evidencia suficiente para pasar a producción, pero
es la línea de trabajo con la que valdría la pena continuar si se retoma el proyecto — con más
verificaciones (más orígenes de fold, quizás ETH además de BTC) antes de cualquier decisión de
uso real.

## Tercera verificación: ETH + significancia estadística bootstrap (2026-08-29, misma sesión) — revierte la conclusión anterior

En vez de un tercer desplazamiento de folds sobre BTC (que ya empezaba a parecerse a "buscar
hasta que confirme"), se hicieron dos cosas genuinamente distintas y más rigurosas:

1. **La misma metodología exacta aplicada a ETH/USDT** (activo independiente, no otro recorte de
   los mismos datos de BTC): walk-forward de -0.05% acumulado, 6/14 folds positivos — no replica
   el patrón positivo de BTC.
2. **Intervalo de confianza bootstrap (5000 remuestreos) sobre los retornos trade-por-trade**
   de cada símbolo (`src/ml_significance.py`, testeado), en vez de conformarse con que el
   agregado del walk-forward diera positivo. Código: `ml_tree_optuna_significance_check.py`,
   reutilizando `simulate_with_trades` (nueva variante de `simulate` que además devuelve el PnL
   de cada trade individual).

**Resultado: el intervalo de confianza al 95% INCLUYE CERO en ambos símbolos.**
- BTC: media +0.044%/trade, IC95% = [-0.249%, +0.347%] — no excluye cero.
- ETH: media -0.002%/trade, IC95% = [-0.393%, +0.412%] — no excluye cero.

**Esto revierte la lectura de la sección anterior.** Que el walk-forward diera +0.44% y luego
+0.98% en dos orígenes distintos de BTC no era, después de todo, evidencia suficiente — el
intervalo de confianza sobre los datos crudos (510-519 trades por símbolo) es demasiado ancho
para distinguir la expectancy real de cero. Estadísticamente, no se puede rechazar que el "edge"
observado sea puro ruido de muestra. Y ETH, con la misma metodología, ni siquiera reproduce el
signo positivo.

**Conclusión definitiva de la línea de clasificadores:** de las ~16 combinaciones probadas en
toda la sesión (4 líneas de estrategia técnica, funding rate, y 4 variantes de clasificador
incluyendo la más rigurosa con Optuna+CV purgada+9 años de historia), **ninguna sostiene una
ventaja estadísticamente significativa**. Los dos walk-forwards positivos de BTC fueron el
resultado más prometedor de la sesión en términos de agregado, pero no superan el estándar de
significancia estadística cuando se los somete a un chequeo diseñado específicamente para eso.
Esto es exactamente lo que se pidió al plantear una verificación "extensa y robusta": templar el
entusiasmo de un hallazgo prometedor con menos escrutinio.

## Recomendación final

**Las 4 hipótesis de mercado consideradas — seguimiento de tendencia, relación de valor relativo
(pairs), reversión a la media, y breakout — fallaron con evidencia consistente y disciplina
train/test.** El mejor candidato de toda la sesión (breakout en BTC solo) se sometió además a
walk-forward con recalibración cada 90 días sobre 5 años y múltiples regímenes reales — el
estándar más riguroso razonable — y también dio neto negativo (-10.58%). Esto descarta que el
problema fuera "un split de mala suerte" o "el activo equivocado": ni la validación más exigente
que se le puede pedir a este enfoque lo salva.

**No se recomienda seguir buscando una regla de entrada/salida sistemática dentro de este marco**
(indicadores técnicos simples sobre precio/volumen, BTC/USDT y ETH/USDT, 1h) — se agotaron tanto
las hipótesis de mercado razonables como el método de validación más riguroso disponible. Insistir
con una quinta variación, o con walk-forward sobre las otras 3 líneas (todas partían de resultados
peores que breakout/BTC en su split único), tiene cada vez menos retorno esperado sobre el
esfuerzo.

Caminos honestos si se retoma el proyecto:
- **Reconsiderar el objetivo** (recomendado): un dashboard de contexto/monitoreo — mostrar
  indicadores, capas de confluencia, niveles de SL/TP en tiempo real vía TradingView MCP — como
  apoyo a decisión manual, sin afirmar una ventaja sistemática que no existe. Es justo lo que
  `contexto proyecto trading.md` pedía desde el inicio ("apoyo a decisión manual, no ejecución
  automática") y aprovecha todo el código de indicadores/riesgo ya construido.
- Otro universo de activos con dinámicas genuinamente distintas entre sí (los pares
  correlacionados del documento original — SOL, BNB, XRP — probablemente comparten el mismo
  problema que BTC/ETH).
- Otra fuente de datos (on-chain, sentimiento, order flow) en vez de solo precio/volumen — un
  cambio de información de entrada, no de patrón técnico. *(Actualización 2026-08-30: se probó
  la variante de order flow — open interest de Bybit + taker buy/sell ratio de Binance, sustituto
  del ratio long/short por cuenta, que no tiene historia suficiente en ningún exchange revisado —
  y se backtesteó con el mismo rigor de las 4 líneas anteriores. No sostiene fuera de muestra.
  Ver "Línea 5" más abajo.)*
- **No** volver a probar 4h (descartado explícitamente por el usuario), ni recalibrar dentro de
  las 4 hipótesis ya rechazadas, ni walk-forward sobre las líneas que ya partían peor que
  breakout/BTC en su split único.

## Principio que se respetó durante toda la sesión

Ninguna métrica de win rate/expectancy se reportó sin venir de un backtest real, y ninguna
configuración se declaró "buena" sin pasar por la separación train/test — exactamente lo que
pide `contexto proyecto trading.md`. El resultado no es el que se esperaba, pero es el resultado
real, en las 4 líneas probadas.

---

## Línea 5: flujo institucional (open interest + taker buy/sell ratio) — backtesteada y rechazada (2026-08-30, sesión de automatización del dashboard)

Una de las vías "honestas" señaladas arriba en la Recomendación final ("otra fuente de datos...
order flow, en vez de solo precio/volumen"). A diferencia de las 4 líneas anteriores, esta pasó
por dos etapas: primero un intento fallido por falta de datos, después un camino alternativo real
que sí permitió backtestear con el mismo rigor de siempre.

### Etapa 1 — el dato "literal" no tiene historia suficiente en ningún exchange

Verificado vía `ccxt` en Binance (`fetch_open_interest_history`,
`fapiDataGetGlobalLongShortAccountRatio`), Bybit, Bitget y OKX (`fetchLongShortRatioHistory`):
**ninguno** retiene más de ~21 días gratis para el ratio long/short por cuenta (Binance y Bybit:
~21 días; OKX: ~4 días; Bitget: ~1 día). Pedir explícitamente `since` más viejo es rechazado
directamente por el servidor (`BadRequest: parameter 'startTime' is invalid`, código -1130 en
Binance) — no es un límite de paginación sorteable como el de OHLCV, es la retención real del
endpoint. También se revisaron dYdX y Hyperliquid (perpetuos on-chain): ninguno de los dos expone
esta métrica vía `ccxt`. Con solo ~500-720 barras horarias, ningún walk-forward/CV purgada/
bootstrap tendría el poder estadístico que usaron las 4 líneas anteriores (la verificación
bootstrap de la línea de clasificadores ya mostró un intervalo demasiado ancho para concluir nada
con ~510-519 trades — acá la muestra sería una fracción de eso).

### Etapa 2 — dos sustitutos reales con historia completa

- **Interés abierto de Bybit** (no Binance): mismo endpoint tipo, pero con retención de ~4.9 años
  (desde sept. 2021) en vez de ~21 días. Usado como proxy de posicionamiento agregado del mercado,
  aplicado sobre el precio de Binance — mismo criterio que ya usaba `funding_data.py` (una fuente
  de información distinta no necesita venir del mismo exchange que el precio).
- **Ratio de compra/venta agresiva (taker)** de Binance: no es el ratio long/short por cuenta,
  pero es un sustituto legítimo con la MISMA historia completa que ya usa `data_fetch.py` (mismo
  endpoint de velas, campo "taker buy base volume") — mide presión de compra/venta real en vez de
  posición declarada, concepto de microestructura de mercado conocido como order flow imbalance.

Código: `orderflow_data.py` (fetch + caché parquet, igual patrón que `funding_data.py`),
`orderflow_indicators.py` (percentil rodante del OI + z-score rodante del ratio taker, alineados
sobre el índice de 1h), `orderflow_strategy.py` (`OrderflowConfig` en `config.py`).

**Hipótesis (contraria, "posicionamiento saturado"):** interés abierto en percentil ≥90% de su
propia historia (mucho apalancamiento acumulado) + flujo de órdenes fuertemente sesgado a un lado
(z-score del ratio taker por encima/debajo de un umbral) + volumen real → apuesta a que el
mercado revierte contra esa mayoría (short/long squeeze). 3 capas: OI saturado, desequilibrio de
flujo extremo, volumen — mismo patrón de confluencia que el resto de las líneas.

**Resultado de la búsqueda de parámetros en TRAIN** (81 combinaciones: percentil de OI 0.85-0.95,
z-score extremo 1.0-2.0, SL 2-4×ATR, TP 4-8×ATR, exigiendo expectancy positiva en ambos pares,
4 años de historia — límite real de la retención de Bybit, split 70/30):

```
Mejor config: oi_extreme_percentile=0.95, imbalance_extreme_z=1.5, sl_atr_mult=2.0, tp_atr_mult=6.0

TRAIN:
  BTC/USDT: 34 trades, win rate 35.3%, expectancy +0.0085%
  ETH/USDT: 57 trades, win rate 28.1%, expectancy +0.024%

TEST:
  BTC/USDT: 25 trades, win rate 20.0%, expectancy -0.452%
  ETH/USDT:  2 trades, win rate  0.0%, expectancy -1.864%   <- muestra insuficiente además
```

**No sostiene fuera de muestra.** La "mejor" config de todo el grid ya arrancaba con expectancy
casi cero en train (0.0085%, 0.024% — apenas por encima del punto de corte, no una señal fuerte),
y en test se revierte a claramente negativo en ambos pares — mismo patrón que las 4 líneas
anteriores. ETH además da apenas 2 trades en test, ni siquiera alcanza el mínimo para confiar en
el número aunque hubiera sido positivo.

**Conclusión: la línea de flujo institucional tampoco tiene ventaja demostrable en BTC/USDT y
ETH/USDT 1h en este período**, ni usando el sustituto de taker buy/sell ratio combinado con
interés abierto de Bybit. Se detuvo acá (sin escalar a walk-forward completo ni bootstrap) porque
ya falló en el primer filtro barato (test fuera de muestra) — misma disciplina de "no invertir en
validación cara sobre algo que ya falló en la barata" aplicada en las líneas 3 y 4.

**Nota sobre qué quedó sin probar:** el ratio long/short "literal" por cuenta sigue sin
respuesta — no se pudo backtestear con datos gratuitos de ningún exchange revisado. El resultado
de esta línea es sobre el sustituto (taker buy/sell ratio + open interest de Bybit), no sobre esa
métrica específica. Caminos si se quiere ese dato en particular en el futuro: un proveedor de
pago (Coinglass, Glassnode, etc.), o recolección propia hacia adelante (guardarlo cada hora desde
ahora y esperar meses/años antes de poder backtestear con rigor).

---

## Línea 6: estacionalidad (hora del día / día de la semana) — el único hallazgo real de toda la sesión, pero no operable (2026-08-30, misma sesión)

A diferencia de las líneas 1-5 (todas del tipo "regla de entrada con SL/TP vía backtesting.py"),
acá se probó la hipótesis directamente como pregunta estadística: ¿el retorno de la vela
siguiente tiene un sesgo según la hora del día o el día de la semana en que ocurre? No hace falta
ningún dato nuevo — 8 años de OHLCV ya cacheados alcanzan.

**Método** (`seasonality_analysis.py`, `seasonality_validate.py`): retorno de cierre a cierre de
la vela siguiente, agrupado por hora (0-23) y por día de la semana (0-6), con el mismo bootstrap
de `ml_significance.py` ya testeado. Con 24+7=31 grupos probados, un IC al 95% sin corregir daría
~1.55 "positivos" esperados por puro azar — por eso el criterio de éxito exige que el MISMO grupo
excluya cero, con el MISMO signo, en BTC y ETH a la vez (split train/test 70/30, candidatos
buscados solo en train, igual que las líneas anteriores).

**Candidatos encontrados en TRAIN:** hora 20:00 UTC, hora 21:00 UTC, día viernes — los tres
excluían cero con el mismo signo en ambos símbolos.

**Verificación en TEST (nunca tocado hasta este punto):**

```
20:00 UTC: sostiene dirección Y sigue significativo en test
  train -> BTC +0.043% [0.008, 0.079] | ETH +0.046% [0.005, 0.089]
  test  -> BTC +0.042% [0.012, 0.072] | ETH +0.080% [0.034, 0.125]

21:00 UTC: sostiene dirección pero YA NO es significativo en test (ETH: CI [-0.002, 0.086], incluye cero)
viernes:   sostiene dirección pero YA NO es significativo en test (ambos símbolos ~0%, sin efecto)
```

**21:00 UTC y viernes quedan descartados** como ruido de la muestra de train — exactamente el
comportamiento que se esperaba filtrar con el split train/test.

**20:00 UTC es distinto: es el primer resultado de toda la sesión que sostiene fuera de
muestra.** No es una casualidad de train — el efecto se reencontró, con el mismo signo y
significativo, en datos que el descubrimiento nunca vio. Explicación económica plausible (no solo
estadística): 20:00 UTC coincide con el cierre del mercado bursátil de EE. UU. en horario de
verano, un solapamiento de mesas de trading institucionales documentado en la literatura de
microestructura de mercado cripto.

**Pero no es operable así de simple.** El efecto (~0.046% promedio, el menor de los dos símbolos)
es más chico que el costo de una operación completa ida y vuelta en Binance spot (0.1% de
comisión taker por lado × 2 = 0.2%) — la comisión sola es ~4 veces el tamaño del efecto. Operar
esto tal cual con una entrada/salida simple da pérdida neta esperada, no ganancia.

**Conclusión: hallazgo real y confirmado estadísticamente, rechazado por motivos económicos, no
estadísticos** — una categoría distinta de las 5 líneas anteriores (que fallaron el chequeo
estadístico mismo). No se implementó como señal del dashboard porque haría perder plata en
promedio después de comisiones tal como está.

**Caminos si se quiere seguir esto en el futuro** (ninguno implementado, quedan como ideas):
- Probar si una orden maker (comisión típicamente menor, a veces ~0%) en vez de taker cierra la
  brecha — introduce riesgo de que la orden no se ejecute, no es gratis en otro sentido.
- Probar una ventana de retorno más larga que 1 hora (ej. mantener varias horas alrededor de las
  20:00 UTC) para ver si el efecto acumulado supera el costo sin necesitar múltiples operaciones.
- Mostrarlo en el dashboard como **dato de contexto informativo**, no como señal de entrada — ej.
  "históricamente, 20:00 UTC (cierre bursátil EE. UU.) mostró un sesgo alcista pequeño y
  consistente, aunque no supera comisiones de trading" — es información real, honesta, y
  consistente con el resto del panel, sin prometer una ventaja operable.

---

## Línea 7: sentimiento (Fear & Greed Index) — rechazada (2026-08-30, misma sesión)

Hipótesis contraria clásica: cuando el índice Fear & Greed (alternative.me, sentimiento agregado
de todo el mercado cripto, gratis, historia diaria completa desde 2018-02-01, sin límite de
retención) está en miedo extremo o codicia extrema, el mercado revierte — confirmado por vela de
rechazo y volumen real. Misma estructura que `funding_strategy.py` (3 capas: sentimiento extremo,
vela de confirmación, volumen), fuente de información distinta (sentimiento declarado, no
derivado de precio/volumen/derivados). Código: `sentiment_data.py`, `sentiment_indicators.py`,
`sentiment_strategy.py` (`SentimentConfig` en `config.py`).

**Resultado de la búsqueda de parámetros en TRAIN** (144 combinaciones: umbral de miedo 10-25,
umbral de codicia 75-90, SL 2-4×ATR, TP 4-8×ATR, exigiendo expectancy positiva en ambos pares,
8 años de historia, split 70/30):

```
Mejor config: miedo<=10, codicia>=90, sl_atr_mult=3.0, tp_atr_mult=6.0

TRAIN:
  BTC/USDT: 65 trades, expectancy +0.526%
  ETH/USDT: 62 trades, expectancy +1.583%

TEST:
  BTC/USDT: 20 trades, expectancy +0.547%   <- sostiene
  ETH/USDT: 22 trades, expectancy -0.174%   <- se revierte
```

**No sostiene fuera de muestra.** A diferencia de las líneas 1-6, acá TRAIN fue el más
prometedor de toda la sesión (expectancy claramente positiva en ambos pares, no un valor apenas
por encima de cero) — y BTC de hecho sostiene esa dirección en test. Pero ETH se revierte a
negativo, y el criterio de esta sesión siempre exigió que la MISMA config funcione en ambos pares
a la vez, no en uno solo por separado (evita elegir una config que solo funciona en un activo por
azar — la misma razón por la que se rechazó aislar BTC en el "Chequeo BTC-solo" de más arriba).

**Conclusión: la línea de sentimiento tampoco tiene ventaja demostrable en ambos pares a la vez
en este período.** Se detuvo acá (sin escalar a walk-forward ni bootstrap) por la misma
disciplina de "no invertir en validación cara sobre algo que ya falló en la barata" — aunque acá
el fallo es más ajustado que en las líneas 1-5 (BTC solo sí hubiera sostenido), no cambia la
conclusión bajo el estándar de "ambos pares" aplicado en toda la sesión.

---

## Línea 8: macro (índice del dólar, DXY) — parecía la mejor de la sesión, cae con bootstrap (2026-08-30, misma sesión)

Hipótesis contraria: el índice del dólar (DXY, vía la API pública de gráficos de Yahoo Finance,
gratis, historia diaria completa desde 2017) en un extremo de su propia historia reciente
(dólar inusualmente fuerte o débil) apuesta a que el cripto revierte en sentido contrario,
confirmado por vela de rechazo y volumen real. Primera fuente de información de mercados
tradicionales probada en toda la sesión (todo lo anterior era cripto: precio, volumen,
derivados, sentimiento cripto). Código: `macro_data.py`, `macro_indicators.py`,
`macro_strategy.py` (`MacroConfig` en `config.py`).

**Resultado de la búsqueda de parámetros en TRAIN** (108 combinaciones: lookback de percentil
90-270 días, extremo 0.85-0.95, SL 2-4×ATR, TP 4-8×ATR, exigiendo expectancy positiva en ambos
pares, 8 años de historia, split 70/30):

```
Mejor config: lookback=270, extreme_percentile=0.85, sl_atr_mult=4.0, tp_atr_mult=8.0

TRAIN:
  BTC/USDT: 226 trades, expectancy +0.762%, Sharpe 0.52
  ETH/USDT: 257 trades, expectancy +0.721%, Sharpe 0.39

TEST:
  BTC/USDT: 109 trades, expectancy +0.213%   <- sostiene
  ETH/USDT:  95 trades, expectancy +1.395%   <- sostiene, incluso mejor que en train
```

**Esta fue la única línea de toda la sesión que sostuvo fuera de muestra en AMBOS pares a la
vez** con el criterio de expectancy agregada — mejor resultado que cualquiera de las líneas 1-7,
incluidos los dos walk-forwards positivos de BTC que en su momento parecieron el hallazgo más
sólido de la sesión.

**Por eso mismo se aplicó la misma escalada de rigor que se le aplicó a esa línea de
clasificadores cuando pasó lo mismo** (`ml_tree_optuna_significance_check.py`): un resultado
agregado positivo no alcanza, hay que ver si el intervalo de confianza bootstrap sobre los
retornos trade-por-trade EN TEST excluye cero. Se corrió (`macro_significance_check.py`,
reutilizando `ml_significance.bootstrap_mean_ci`, ya testeado):

```
BTC/USDT (TEST, 109 trades): media 0.213%, IC95% [-0.710%, 1.205%] -> incluye cero
ETH/USDT (TEST,  95 trades): media 1.395%, IC95% [-0.193%, 3.038%] -> incluye cero
```

**Ambos intervalos incluyen cero.** Exactamente el mismo desenlace que la línea de
clasificadores: la expectancy agregada positiva no es evidencia suficiente cuando la muestra de
trades (95-109 en test) es chica en relación a la variabilidad de los retornos — el intervalo de
confianza es demasiado ancho para distinguir esta ganancia de pura casualidad de muestra.

**Conclusión: el resultado más prometedor de toda la sesión tampoco resiste el escrutinio
estadístico correcto.** Sirve como recordatorio de por qué el paso de bootstrap importa siempre,
no solo cuando "se ve sospechoso" — este resultado, mirado solo por expectancy agregada y split
train/test, se habría reportado como el único éxito de las 8 líneas probadas. No se implementó
como señal del dashboard.
