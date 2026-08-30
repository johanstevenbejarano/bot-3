"""Arma el HTML completo del dashboard a partir de un snapshot ya calculado
(`live_snapshot.build_snapshot()`). Mismo diseño visual aprobado manualmente — acá solo se
insertan los valores dinámicos.
"""
from __future__ import annotations

import json
from datetime import datetime

from src.dashboard_render import format_price, layers_met_fraction

_MESES = [
    "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic",
]


def _display_timestamp(dt: datetime) -> str:
    return f"{dt.day} {_MESES[dt.month - 1]} {dt.year}, {dt.strftime('%H:%M')} UTC"


def _dots_html(flags: list[bool]) -> str:
    return "".join(f'<span class="dot{" on" if f else ""}"></span>' for f in flags)


def _signal_banner(d: dict) -> str:
    if d["signal_long"] and d["signal_short"]:
        return '<div class="signal-banner">Señales mixtas entre estrategias — revisar manualmente antes de actuar.</div>'
    if d["signal_long"]:
        nombres = ", ".join(d["active_names"])
        return f'<div class="signal-banner active-long">Confluencia de <strong>LARGO</strong> activa: {nombres}.</div>'
    if d["signal_short"]:
        nombres = ", ".join(d["active_names"])
        return f'<div class="signal-banner active-short">Confluencia de <strong>CORTO</strong> activa: {nombres}.</div>'
    return '<div class="signal-banner">Sin confluencia activa ahora mismo — ninguna estrategia dispara señal.</div>'


def _asset_card_html(d: dict, delay_ms: int) -> str:
    spark = d["sparkline"]
    change = d["change_72h"]
    change_cls = "up" if change >= 0 else "down"
    change_arrow = "▲" if change >= 0 else "▼"
    funding_pct_display = "s/d" if d["funding_percentile"] != d["funding_percentile"] else f"{d['funding_percentile'] * 100:.0f}"

    return f"""
    <div class="card" style="animation-delay: {delay_ms}ms">
      <div class="asset-head">
        <div>
          <div class="asset-name">{d["symbol"]}</div>
          <span class="asset-price">{d["price_fmt"]}</span>
          <span class="asset-change {change_cls}">{change_arrow} {abs(change):.2f}% · 72h</span>
        </div>
        <svg class="sparkline" width="140" height="40" viewBox="0 0 280 56" style="--spark-len:600" aria-label="Precio {d["symbol"]}, últimas 72 horas">
          <polygon class="fill" points="{spark["area_points"]}"></polygon>
          <polyline class="line" points="{spark["line_points"]}"></polyline>
          <circle class="endpoint" cx="{spark["endpoint_x"]}" cy="{spark["endpoint_y"]}" r="2.6"></circle>
        </svg>
      </div>
      <div class="asset-meta">
        <span>ADX {d["adx"]:.1f}</span>
        <span>ATR {d["atr_pct_of_price"]:.2f}% del precio</span>
        <span>Funding pct. {funding_pct_display}</span>
      </div>

      <div class="layer-row">
        <span class="layer-name">Tendencia (EMA + ADX + rebote)</span>
        <span class="layer-detail">{layers_met_fraction(d["trend_flags"])}</span>
        <span class="dots">{_dots_html(d["trend_flags"])}</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Reversión (Bollinger + RSI)</span>
        <span class="layer-detail">{layers_met_fraction(d["meanrev_flags"])}</span>
        <span class="dots">{_dots_html(d["meanrev_flags"])}</span>
      </div>
      <div class="layer-row">
        <span class="layer-name">Breakout (canal + volumen)</span>
        <span class="layer-detail">{layers_met_fraction(d["breakout_flags"])}</span>
        <span class="dots">{_dots_html(d["breakout_flags"])}</span>
      </div>

      {_signal_banner(d)}

      <div class="interpretation">
        <span class="interp-label">Interpretación de este momento <em>(lectura del presente, no una predicción)</em></span>
        <p class="interp-text">Interpretación pendiente de esta actualización para {d["symbol"]}.</p>
      </div>
    </div>"""


def _risk_rows_html(d: dict) -> str:
    """Una fila por estrategia (cada una con su propia calibración de SL/TP) -- ver
    StrategyRisk en dashboard_render.py. La de Reversión marca su TP como dinámico."""
    rows = []
    for r in d["strategy_risks"]:
        tp_long = f"{format_price(r.long_tp)}<span class='dyn-mark'>*</span>" if r.tp_is_dynamic else format_price(r.long_tp)
        tp_short = f"{format_price(r.short_tp)}<span class='dyn-mark'>*</span>" if r.tp_is_dynamic else format_price(r.short_tp)
        rows.append(f"""        <tr>
          <td>{d["symbol"]}</td>
          <td>{r.name}</td>
          <td>{d["price_fmt"]}</td>
          <td class="sl">{format_price(r.long_sl)}</td>
          <td class="tp">{tp_long}</td>
          <td class="sl">{format_price(r.short_sl)}</td>
          <td class="tp">{tp_short}</td>
        </tr>""")
    return "\n".join(rows)


def _calculator_data_js(symbols: list[dict]) -> str:
    """Datos que la calculadora de tamaño de posición necesita, embebidos como JS -- todo el
    cálculo corre en el navegador del usuario, nada de su capital sale de la página."""
    data = {
        d["symbol"]: {
            "price": d["price"],
            "strategies": {r.name: {"sl_dist": r.sl_dist} for r in d["strategy_risks"]},
        }
        for d in symbols
    }
    return json.dumps(data, ensure_ascii=False)


def render_html(snapshot: dict) -> str:
    symbols = list(snapshot["symbols"].values())
    asset_cards = "\n".join(_asset_card_html(d, i * 70) for i, d in enumerate(symbols))
    risk_rows = "\n".join(_risk_rows_html(d) for d in symbols)
    calc_data_js = _calculator_data_js(symbols)

    correlation = snapshot["correlation"]
    avg_adx = sum(d["adx"] for d in symbols) / len(symbols)
    avg_atr_pct = sum(d["atr_pct_of_price"] for d in symbols) / len(symbols)

    generated_iso = snapshot["generated_at"].strftime("%Y-%m-%dT%H:%M:%SZ")
    generated_display = _display_timestamp(snapshot["generated_at"])

    return f"""<title>Panel de Confluencia</title>
<style>
  :root {{
    --bg: #F4F6F9;
    --surface: #FFFFFF;
    --surface-2: #EBEEF3;
    --border: #DCE1E8;
    --text: #12151C;
    --text-muted: #5B6472;
    --accent: #A6811C;
    --accent-soft: #F2E9D2;
    --good: #1F7A57;
    --good-soft: #DFF3EA;
    --bad: #B14640;
    --bad-soft: #FBE6E4;
    --warn: #A85D1F;
    --warn-soft: #F7E7D6;
    --neutral-soft: #E7E9EE;
    --font-sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
  }}

  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0B0E14; --surface: #131722; --surface-2: #1B2130; --border: #262E40;
      --text: #E8EAF0; --text-muted: #8891A5; --accent: #D4B24C; --accent-soft: #2E2712;
      --good: #4FC08D; --good-soft: #143026; --bad: #E2726C; --bad-soft: #331A18;
      --warn: #E0A560; --warn-soft: #332411; --neutral-soft: #1B2130;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0B0E14; --surface: #131722; --surface-2: #1B2130; --border: #262E40;
    --text: #E8EAF0; --text-muted: #8891A5; --accent: #D4B24C; --accent-soft: #2E2712;
    --good: #4FC08D; --good-soft: #143026; --bad: #E2726C; --bad-soft: #331A18;
    --warn: #E0A560; --warn-soft: #332411; --neutral-soft: #1B2130;
  }}

  * {{ box-sizing: border-box; }}
  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
  }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font-sans); margin: 0; padding: 32px 20px 80px; line-height: 1.5; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}

  @keyframes fade-up {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  @keyframes pulse-ring {{
    0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 45%, transparent); }}
    100% {{ box-shadow: 0 0 0 8px color-mix(in srgb, currentColor 0%, transparent); }}
  }}
  @keyframes dash-in {{ from {{ stroke-dashoffset: var(--spark-len); }} to {{ stroke-dashoffset: 0; }} }}

  header {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }}
  .brand {{ display: flex; align-items: baseline; gap: 10px; }}
  .brand-mark {{ width: 10px; height: 10px; border-radius: 2px; background: var(--accent); display: inline-block; transform: rotate(45deg); }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }}
  .subtitle {{ color: var(--text-muted); font-size: 13px; margin-top: 2px; }}
  .updated {{ font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); text-align: right; }}
  .updated strong {{ color: var(--text); font-weight: 500; }}

  .section-label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); margin: 36px 0 14px; }}
  .section-label:first-of-type {{ margin-top: 0; }}

  .asset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}

  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; transition: border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease; animation: fade-up 420ms ease both; }}
  .card:hover {{ border-color: color-mix(in srgb, var(--accent) 45%, var(--border)); transform: translateY(-2px); box-shadow: 0 8px 24px -12px rgba(0, 0, 0, 0.25); }}

  .asset-head {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; gap: 14px; }}
  .asset-name {{ font-size: 17px; font-weight: 600; }}
  .asset-price {{ font-family: var(--font-mono); font-size: 20px; font-weight: 500; font-variant-numeric: tabular-nums; display: block; }}
  .asset-change {{ font-size: 11.5px; font-family: var(--font-mono); text-align: right; }}
  .asset-change.up {{ color: var(--good); }}
  .asset-change.down {{ color: var(--bad); }}

  .sparkline {{ display: block; overflow: visible; }}
  .sparkline .fill {{ fill: color-mix(in srgb, var(--accent) 14%, transparent); }}
  .sparkline .line {{ fill: none; stroke: var(--accent); stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: var(--spark-len); stroke-dashoffset: var(--spark-len); animation: dash-in 1100ms ease-out 120ms forwards; }}
  .sparkline .endpoint {{ fill: var(--accent); }}

  .asset-meta {{ font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); margin-bottom: 18px; display: flex; gap: 14px; flex-wrap: wrap; }}

  .layer-row {{ display: flex; align-items: center; gap: 12px; padding: 9px 0; border-top: 1px solid var(--border); }}
  .layer-row:first-of-type {{ border-top: none; }}
  .layer-name {{ font-size: 13.5px; flex: 1; }}
  .layer-detail {{ font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); }}

  .dots {{ display: flex; gap: 4px; }}
  .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--neutral-soft); border: 1px solid var(--border); transition: background 300ms ease, border-color 300ms ease, transform 300ms ease; }}
  .dot.on {{ background: var(--accent); border-color: var(--accent); transform: scale(1.15); }}

  .signal-banner {{ margin-top: 16px; padding: 12px 14px; border-radius: 8px; background: var(--neutral-soft); font-size: 13px; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
  .signal-banner.active-long {{ background: var(--good-soft); color: var(--good); animation: fade-up 420ms ease both, pulse-ring 1.8s ease-out 1; }}
  .signal-banner.active-short {{ background: var(--bad-soft); color: var(--bad); animation: fade-up 420ms ease both, pulse-ring 1.8s ease-out 1; }}
  .signal-banner strong {{ font-weight: 600; }}

  .interpretation {{ margin-top: 12px; padding: 12px 14px; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--border); }}
  .interp-label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.03em; color: var(--text-muted); display: block; margin-bottom: 6px; }}
  .interp-label em {{ font-weight: 400; font-style: normal; color: var(--text-muted); opacity: 0.8; }}
  .interp-text {{ font-size: 13px; color: var(--text); margin: 0; line-height: 1.55; }}

  .regime-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
  .metric-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; transition: border-color 180ms ease, transform 180ms ease; animation: fade-up 420ms ease both; }}
  .metric-card:hover {{ border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); transform: translateY(-2px); }}
  .metric-label {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; }}
  .metric-value {{ font-family: var(--font-mono); font-size: 24px; font-weight: 500; font-variant-numeric: tabular-nums; }}
  .metric-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

  .gauge-track {{ height: 6px; border-radius: 4px; background: var(--neutral-soft); margin-top: 10px; overflow: hidden; }}
  .gauge-fill {{ height: 100%; border-radius: 4px; background: var(--accent); width: 0; transition: width 900ms cubic-bezier(0.22, 1, 0.36, 1) 150ms; }}
  .gauge-fill.warn {{ background: var(--warn); }}

  .card {{ overflow-x: auto; }}
  .risk-table {{ width: 100%; min-width: 560px; border-collapse: collapse; }}
  .risk-table th {{ text-align: left; font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-muted); padding: 0 12px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .risk-table td {{ padding: 12px; font-family: var(--font-mono); font-size: 13.5px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  .risk-table tr:last-child td {{ border-bottom: none; }}
  .risk-table .sl {{ color: var(--bad); }}
  .risk-table .tp {{ color: var(--good); }}
  .dyn-mark {{ color: var(--accent); font-weight: 700; }}
  .risk-table-note {{ padding: 10px 12px 2px; font-size: 12px; color: var(--text-muted); }}

  .calculator .calc-intro {{ font-size: 12.5px; color: var(--text-muted); margin: 0 0 16px; line-height: 1.5; }}
  .calc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 4px; }}
  .calc-grid label {{ display: flex; flex-direction: column; gap: 6px; font-size: 12px; color: var(--text-muted); }}
  .calc-grid select, .calc-grid input {{ font: inherit; font-family: var(--font-mono); font-size: 13.5px; color: var(--text); background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }}
  .calc-grid select:focus, .calc-grid input:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  .calc-result {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border); }}
  .calc-result .stat-line span:last-child {{ font-weight: 500; color: var(--text); }}
  .calc-leverage-note {{ margin-top: 10px; padding: 8px 10px; border-radius: 6px; background: var(--warn-soft); color: var(--warn); font-size: 12px; }}

  details.methodology {{ margin-top: 40px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); overflow: hidden; }}
  details.methodology summary {{ cursor: pointer; padding: 16px 20px; font-size: 13.5px; font-weight: 600; color: var(--text-muted); list-style: none; display: flex; align-items: center; gap: 8px; }}
  details.methodology summary::-webkit-details-marker {{ display: none; }}
  details.methodology summary::before {{ content: "+"; font-family: var(--font-mono); color: var(--accent); font-size: 15px; width: 16px; }}
  details.methodology[open] summary::before {{ content: "\\2013"; }}
  .methodology-body {{ padding: 4px 20px 22px; color: var(--text-muted); font-size: 13.5px; }}
  .methodology-body h4 {{ color: var(--text); font-size: 13px; margin: 18px 0 6px; }}
  .methodology-body p {{ margin: 6px 0; }}
  .methodology-body ul {{ margin: 6px 0; padding-left: 20px; }}
  .methodology-body li {{ margin: 4px 0; }}
  .stat-line {{ display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 12.5px; padding: 4px 0; }}

  footer {{ margin-top: 32px; text-align: center; font-size: 11.5px; color: var(--text-muted); }}

  @media (max-width: 600px) {{
    body {{ padding: 20px 14px 60px; }}
    header {{ flex-direction: column; align-items: flex-start; }}
    .updated {{ text-align: left; }}
  }}
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">

<div class="wrap">
  <header>
    <div class="brand">
      <span class="brand-mark"></span>
      <div>
        <h1>Confluencia BTC / ETH</h1>
        <div class="subtitle">Apoyo a decisión manual — no ejecuta órdenes</div>
      </div>
    </div>
    <div class="updated">Última actualización<br><strong id="updated-time" data-generated="{generated_iso}">{generated_display}</strong><br><span id="updated-ago" style="color: var(--text-muted); font-weight: 400;"></span></div>
  </header>

  <div class="section-label">Señales en vivo</div>
  <div class="asset-grid">
{asset_cards}
  </div>

  <div class="section-label">Contexto de régimen</div>
  <div class="regime-grid">
    <div class="metric-card">
      <div class="metric-label">Correlación BTC–ETH (30 días)</div>
      <div class="metric-value">{correlation:.2f}</div>
      <div class="gauge-track"><div class="gauge-fill" data-w="{abs(correlation) * 100:.0f}"></div></div>
      <div class="metric-sub">{"Alta" if abs(correlation) > 0.6 else "Moderada" if abs(correlation) > 0.3 else "Baja"} — no tratar ambos pares como diversificación real.</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Fuerza de tendencia (ADX)</div>
      <div class="metric-value">~{avg_adx:.0f}</div>
      <div class="gauge-track"><div class="gauge-fill{" warn" if avg_adx < 25 else ""}" data-w="{min(avg_adx * 2, 100):.0f}"></div></div>
      <div class="metric-sub">{"Débil" if avg_adx < 25 else "Confirmada"} en promedio — {"mercado sin dirección clara." if avg_adx < 25 else "hay tendencia real en curso."}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Volatilidad relativa (ATR)</div>
      <div class="metric-value">{avg_atr_pct:.2f}%</div>
      <div class="gauge-track"><div class="gauge-fill" data-w="{min(avg_atr_pct * 40, 100):.0f}"></div></div>
      <div class="metric-sub">Respecto al precio, promedio de ambos pares.</div>
    </div>
  </div>

  <div class="section-label">Niveles de riesgo de referencia</div>
  <div class="card">
    <table class="risk-table">
      <thead><tr><th>Par</th><th>Estrategia</th><th>Precio</th><th>SL largo</th><th>TP largo</th><th>SL corto</th><th>TP corto</th></tr></thead>
      <tbody>
{risk_rows}
      </tbody>
    </table>
    <div class="risk-table-note"><span class="dyn-mark">*</span> Reversión no usa un TP fijo por ATR: el objetivo es la media móvil actual, que se mueve con el mercado.</div>
  </div>

  <div class="section-label">Calculadora de tamaño de posición</div>
  <div class="card calculator">
    <p class="calc-intro">Esto es matemática de gestión de riesgo, no una predicción — sirve
    para cualquier operación, tenga o no ventaja comprobada la señal que la originó. Todo el
    cálculo ocurre en tu navegador: tu capital nunca se envía a ningún lado.</p>
    <div class="calc-grid">
      <label>Par
        <select id="calc-symbol"></select>
      </label>
      <label>Estrategia (define la distancia al SL)
        <select id="calc-strategy"></select>
      </label>
      <label>Capital disponible (USD)
        <input id="calc-capital" type="number" min="0" step="any" placeholder="ej. 1000">
      </label>
      <label>Riesgo máximo por operación (%)
        <input id="calc-risk-pct" type="number" min="0" max="100" step="any" value="1">
      </label>
    </div>
    <div id="calc-result" class="calc-result" hidden>
      <div class="stat-line"><span>Dinero en riesgo si toca el SL</span><span id="calc-risk-amount">—</span></div>
      <div class="stat-line"><span>Tamaño de posición sugerido</span><span id="calc-position-value">—</span></div>
      <div class="stat-line"><span>Unidades aproximadas</span><span id="calc-units">—</span></div>
      <div id="calc-leverage-note" class="calc-leverage-note" hidden></div>
    </div>
  </div>

  <details class="methodology">
    <summary>Metodología y honestidad estadística</summary>
    <div class="methodology-body">
      <p>Estas 3 estrategias (tendencia, reversión a la media, breakout) y un filtro de funding
      fueron sometidas a una investigación exhaustiva de backtest: 4 hipótesis de mercado, una
      fuente de datos alternativa, y 4 variantes de clasificador (incluyendo Random Forest /
      Gradient Boosting con Optuna y validación cruzada purgada), validadas con walk-forward de
      hasta 9 años de historia en BTC y ETH.</p>

      <h4>Resultado del backtest</h4>
      <div class="stat-line"><span>Intervalo de confianza (bootstrap, 95%) sobre BTC</span><span>incluye cero</span></div>
      <div class="stat-line"><span>Intervalo de confianza (bootstrap, 95%) sobre ETH</span><span>incluye cero</span></div>
      <div class="stat-line"><span>Mejor resultado agregado de walk-forward encontrado</span><span>+0.98% / 3.5 años</span></div>

      <p><strong>Ninguna configuración probada mostró una ventaja estadísticamente
      significativa.</strong> Las señales de este panel son el estado actual de reglas técnicas
      bien definidas — no una probabilidad de acierto validada. Este dashboard es una herramienta
      de contexto para apoyar una decisión manual, no un sistema de señales con ventaja
      demostrada.</p>

      <h4>Qué sí se validó</h4>
      <ul>
        <li>La lógica de cada capa está testeada unitariamente y es reproducible.</li>
        <li>Los niveles de riesgo se calculan con ATR real de cada par, no estimaciones fijas.</li>
        <li>El dato de precio, volumen y funding viene directo de la API pública de Binance.</li>
      </ul>

      <h4>Qué no se validó</h4>
      <ul>
        <li>Que seguir estas señales produzca ganancias por encima del azar.</li>
        <li>El modelo de aprendizaje automático (no mostrado como señal activa por esta razón).</li>
      </ul>
    </div>
  </details>

  <footer>Actualizado automáticamente cada hora vía datos públicos de Binance · Uso exclusivo como apoyo a decisión manual</footer>
</div>

<script>
(function () {{
  requestAnimationFrame(function () {{
    requestAnimationFrame(function () {{
      document.querySelectorAll(".gauge-fill[data-w]").forEach(function (el) {{
        el.style.width = el.getAttribute("data-w") + "%";
      }});
    }});
  }});

  function tickFreshness() {{
    var el = document.getElementById("updated-time");
    var out = document.getElementById("updated-ago");
    if (!el || !out) return;
    var ts = el.getAttribute("data-generated");
    if (!ts) return;
    var diffMin = Math.max(0, Math.round((Date.now() - new Date(ts).getTime()) / 60000));
    var label;
    if (diffMin < 1) label = "justo ahora";
    else if (diffMin === 1) label = "hace 1 minuto";
    else if (diffMin < 60) label = "hace " + diffMin + " minutos";
    else label = "hace " + Math.round(diffMin / 60) + " h";
    out.textContent = label;
  }}
  tickFreshness();
  setInterval(tickFreshness, 30000);

  // Calculadora de tamaño de posición -- todo corre acá, en el navegador; el capital que
  // escribe el usuario nunca sale de esta página.
  var CALC_DATA = {calc_data_js};
  var symSel = document.getElementById("calc-symbol");
  var stratSel = document.getElementById("calc-strategy");
  var capitalInput = document.getElementById("calc-capital");
  var riskPctInput = document.getElementById("calc-risk-pct");
  var resultBox = document.getElementById("calc-result");
  var riskAmountEl = document.getElementById("calc-risk-amount");
  var positionValueEl = document.getElementById("calc-position-value");
  var unitsEl = document.getElementById("calc-units");
  var leverageNoteEl = document.getElementById("calc-leverage-note");

  if (symSel && CALC_DATA) {{
    Object.keys(CALC_DATA).forEach(function (sym) {{
      var opt = document.createElement("option");
      opt.value = sym;
      opt.textContent = sym;
      symSel.appendChild(opt);
    }});

    function populateStrategies() {{
      var sym = CALC_DATA[symSel.value];
      stratSel.innerHTML = "";
      if (!sym) return;
      Object.keys(sym.strategies).forEach(function (name) {{
        var opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        stratSel.appendChild(opt);
      }});
    }}

    function fmtUsd(n) {{
      return "$" + n.toLocaleString("en-US", {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    }}

    function recalc() {{
      var sym = CALC_DATA[symSel.value];
      var strategy = sym ? sym.strategies[stratSel.value] : null;
      var capital = parseFloat(capitalInput.value);
      var riskPct = parseFloat(riskPctInput.value);

      if (!sym || !strategy || !(capital > 0) || !(riskPct > 0) || !(strategy.sl_dist > 0)) {{
        resultBox.hidden = true;
        return;
      }}

      var riskAmount = capital * (riskPct / 100);
      var units = riskAmount / strategy.sl_dist;
      var positionValue = units * sym.price;

      riskAmountEl.textContent = fmtUsd(riskAmount);
      positionValueEl.textContent = fmtUsd(positionValue);
      unitsEl.textContent = units.toLocaleString("en-US", {{ maximumFractionDigits: 6 }});

      if (positionValue > capital) {{
        var mult = (positionValue / capital).toFixed(2);
        leverageNoteEl.textContent = "Esto supera tu capital disponible — implicaria apalancamiento de ~" + mult + "x para mantener ese riesgo con este SL. Considera bajar el % de riesgo o aceptar un SL mas ajustado.";
        leverageNoteEl.hidden = false;
      }} else {{
        leverageNoteEl.hidden = true;
      }}

      resultBox.hidden = false;
    }}

    populateStrategies();
    recalc();
    symSel.addEventListener("change", function () {{ populateStrategies(); recalc(); }});
    [stratSel, capitalInput, riskPctInput].forEach(function (el) {{
      el.addEventListener("input", recalc);
      el.addEventListener("change", recalc);
    }});
  }}

  var details = document.querySelector("details.methodology");
  if (details) {{
    var body = details.querySelector(".methodology-body");
    var animating = false;
    details.addEventListener("click", function (e) {{
      if (animating) {{ e.preventDefault(); return; }}
      var summary = details.querySelector("summary");
      if (!summary.contains(e.target)) return;
      e.preventDefault();
      animating = true;
      var opening = !details.open;
      if (opening) details.open = true;
      var startHeight = opening ? 0 : body.offsetHeight;
      var endHeight = opening ? body.offsetHeight : 0;
      var anim = body.animate(
        [{{ height: startHeight + "px", opacity: opening ? 0 : 1 }}, {{ height: endHeight + "px", opacity: opening ? 1 : 0 }}],
        {{ duration: 240, easing: "cubic-bezier(0.22, 1, 0.36, 1)" }}
      );
      anim.onfinish = function () {{
        if (!opening) details.open = false;
        animating = false;
      }};
    }});
  }}
}})();
</script>
"""
