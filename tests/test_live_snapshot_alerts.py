from src.live_snapshot import _build_alert, _notify_active_signals


def _fake_symbol_data(signal_long=False, signal_short=False, active_names=None):
    return {
        "symbol": "BTC/USDT",
        "price_fmt": "$78,123.45",
        "signal_long": signal_long,
        "signal_short": signal_short,
        "active_names": active_names or [],
    }


def test_build_alert_never_uses_predictive_language():
    d = _fake_symbol_data(signal_long=True, active_names=["tendencia"])
    title, message = _build_alert("BTC/USDT", d)

    banned_phrases = ["creo que", "es probable", "recomiendo comprar", "recomiendo vender", "va a subir", "va a bajar"]
    combined = (title + " " + message).lower()
    for phrase in banned_phrases:
        assert phrase not in combined
    assert "no es una predicción" in message.lower()


def test_build_alert_describes_current_conditions():
    d = _fake_symbol_data(signal_long=True, active_names=["tendencia"])
    title, message = _build_alert("BTC/USDT", d)

    assert "BTC/USDT" in title
    assert "LARGO" in title
    assert "tendencia" in message
    assert "no es una predicción" in message


def test_build_alert_reflects_short_direction():
    d = _fake_symbol_data(signal_short=True, active_names=["breakout"])
    title, _ = _build_alert("ETH/USDT", d)

    assert "CORTO" in title


def test_notify_active_signals_only_fires_for_active_symbols(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "src.live_snapshot.send_ntfy_alert",
        lambda title, message, tags=None: sent.append((title, message)) or True,
    )

    symbols_data = {
        "BTC/USDT": _fake_symbol_data(signal_long=True, active_names=["tendencia"]),
        "ETH/USDT": _fake_symbol_data(),  # sin señal
    }
    _notify_active_signals(symbols_data)

    assert len(sent) == 1
    assert "BTC/USDT" in sent[0][0]
