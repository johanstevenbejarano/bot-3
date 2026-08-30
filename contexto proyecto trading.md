# Contexto del Proyecto: Dashboard de Análisis de Trading (Binance)

## Objetivo general
Construir un sistema que:
1. Se conecte a TradingView Desktop vía MCP (Model Context Protocol) para lectura de datos en tiempo real.
2. Ejecute 3 estrategias técnicas en confluencia sobre pares de Binance.
3. Muestre un dashboard en tiempo real con señales, métricas históricas (win rate, expectancy, drawdown) validadas por backtest, y niveles concretos de SL/TP.
4. Sirva como apoyo a decisiones — el usuario ejecuta manualmente, el sistema NO opera de forma automática.

**Importante:** el usuario opera manualmente sus entradas/salidas. Este sistema es de análisis y apoyo a decisión, no un bot de ejecución automática.

---

## Mercado y alcance
- **Exchange:** Binance (spot, vía API pública de datos — no se requiere API key con permisos de trading para la fase de análisis)
- **Pares iniciales:** BTC/USDT y ETH/USDT (fase de validación)
- **Pares de expansión** (una vez validado el sistema): SOL/USDT, BNB/USDT, XRP/USDT, posiblemente AVAX/USDT o LINK/USDT
- **Timeframe principal:** 1h (punto de partida; evaluar bajar a 15m tras validar en 1h)
- **Nota de riesgo conocida:** los pares elegidos tienen alta correlación entre sí — no representan diversificación real, hay que tenerlo en cuenta en el position sizing agregado.

---

## Estrategias (diseñadas para confluencia, no redundancia)

Cada estrategia cubre un ángulo distinto: dirección, timing, y confirmación.

### 1. Tendencia — filtro direccional
- EMA 50 y EMA 100 (se eligió 100 en vez de 200 por ser más ágil para timeframes intradía en cripto)
- Solo se consideran señales largas si EMA50 > EMA100 (y cortas si EMA50 < EMA100, si se decide operar ambos lados)
- ADX(14) > 20 como filtro de que existe tendencia real (evita operar en lateralización)

### 2. Momentum/Timing — disparador de entrada
- RSI(14) cruzando sobre 45 al alza (para largos), evitando compras en sobrecompra extrema
- MACD (12,26,9): cruce alcista de la línea MACD sobre la señal
- Ambas condiciones deben cumplirse en el mismo periodo o con máximo 1-2 velas de diferencia

### 3. Confirmación — filtro de calidad
- Volumen de la vela actual > promedio móvil de volumen de 20 periodos
- Objetivo: descartar rupturas o cruces sin participación real de mercado

### Regla de entrada
Señal válida = Tendencia (1) confirmada + Momentum (2) disparado + Volumen (3) confirmado.
El dashboard debe mostrar cuántas de las 3 capas se cumplen, no solo un semáforo binario, para que el usuario vea el nivel de confluencia real.

---

## Gestión de riesgo (SL/TP y sizing)

- **Stop Loss:** basado en ATR(14) — SL = 1.5 × ATR por debajo del swing low reciente (para largos)
- **Take Profit:** TP = 2.5 × ATR desde el punto de entrada (ratio riesgo:beneficio inicial ≈ 1:1.67, ajustable según resultados del backtest)
- **Position sizing:** pendiente de calcular con Kelly fraccionado (25-50% del Kelly completo) una vez que el backtest entregue win rate y ratio riesgo/beneficio reales — no usar tamaño fijo arbitrario
- **Validación de robustez:** correr simulación de Monte Carlo (1000-5000 iteraciones reordenando los trades del backtest) para estimar rango real de drawdown esperado, no solo el drawdown del orden histórico real

---

## Backtest (paso previo obligatorio antes de operar en vivo)

- **Periodo:** mínimo 2 años de histórico en 1h, para cruzar al menos un ciclo alcista y uno bajista/lateral
- **Fuente de datos:** API pública de Binance (vía librería `ccxt` en Python) — ejecutar localmente, ya que el sandbox de Claude.ai no tiene acceso de red a api.binance.com
- **Librerías sugeridas:** `ccxt` (datos), `pandas`/`numpy` (procesamiento), `ta` (indicadores técnicos), `backtesting.py` (motor de backtest)
- **Métricas que debe entregar el backtest:**
  - Win rate
  - Expectancy (ganancia/pérdida promedio esperada por trade)
  - Máximo drawdown
  - Ratio de Sharpe o similar
  - Número total de trades (para verificar significancia estadística — evitar conclusiones con muestras pequeñas, como ocurrió en un intento previo con solo 5 operaciones en 6 meses)
  - Resultado de Monte Carlo (rango de drawdown probable)

**Antecedente relevante:** un backtest previo de una estrategia más rígida (más condiciones exigidas) sobre 6 meses arrojó solo 5 operaciones, todas perdedoras — muestra insuficiente para concluir si el problema fue el diseño de la estrategia o simple mala suerte estadística. Por eso ahora se exige mínimo 2 años de histórico antes de sacar conclusiones.

---

## Conexión a TradingView (tiempo real)

- Herramienta: MCP server de TradingView para Claude Code (ej. proyecto `tradingview-mcp`), que conecta vía Chrome DevTools Protocol (CDP) a la app TradingView Desktop corriendo localmente
- Requiere: TradingView Desktop instalado y corriendo con `--remote-debugging-port=9222`, suscripción válida de TradingView
- Uso previsto: solo lectura (precio, indicadores, estructura de velas) — sin ejecución de órdenes automática
- Consideraciones de seguridad/ToS ya evaluadas y aceptadas por el usuario: es un conector no oficial de terceros, revisar Términos de Uso de TradingView, mantener permisos en modo lectura

---

## Dashboard (entregable final)

Debe mostrar, por par monitoreado:
- Estado de las 3 capas de la estrategia (cumple/no cumple cada una, no solo semáforo final)
- Señal activa o no (y de qué fuerza: 1/3, 2/3, 3/3 condiciones)
- SL y TP calculados dinámicamente vía ATR
- Métricas históricas de la estrategia (win rate, expectancy, drawdown) provenientes del backtest — no estimaciones improvisadas en tiempo real
- Position size sugerido (una vez implementado Kelly fraccionado)
- Actualización en tiempo real vía conexión MCP a TradingView

---

## Orden de ejecución acordado

1. ✅ Definir estrategias y pares (completado)
2. ⏳ Construir y correr el backtest en Python localmente (BTC/USDT y ETH/USDT, 1h, 2 años) — **siguiente paso**
3. Analizar resultados: si son sólidos, calibrar SL/TP y sizing con Kelly + Monte Carlo
4. Si los resultados no son sólidos, ajustar parámetros (ej. thresholds de RSI/ADX, timeframe) y volver a backtestear — no lanzar a producción sin validación
5. Instalar TradingView MCP en Claude Code local
6. Construir el dashboard en tiempo real que consuma la conexión MCP + las reglas ya validadas por el backtest
7. Expandir a pares adicionales (SOL, BNB, XRP) una vez validado el sistema en BTC/ETH

---

## Principios que deben respetarse durante el desarrollo

- Ninguna métrica de "probabilidad" o win rate debe mostrarse en el dashboard sin estar respaldada por el backtest real — nada de estimaciones improvisadas.
- No agregar más condiciones/indicadores solo para "verse más robusto" — cada capa debe aportar una señal distinta (evitar redundancia).
- Los pares monitoreados están correlacionados: no tratar el conjunto como diversificación real al dimensionar el riesgo total expuesto.
- El sistema es de apoyo a decisión manual, no de ejecución automática de órdenes.
