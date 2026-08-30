from src.ntfy_notify import send_ntfy_alert


def test_returns_false_when_alerts_disabled(monkeypatch):
    monkeypatch.delenv("ALERT_ENABLED", raising=False)
    monkeypatch.setenv("NTFY_TOPIC", "some-topic")

    assert send_ntfy_alert("titulo", "mensaje") is False


def test_returns_false_when_enabled_but_no_topic(monkeypatch):
    monkeypatch.setenv("ALERT_ENABLED", "true")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)

    assert send_ntfy_alert("titulo", "mensaje") is False


def test_sends_json_payload_when_configured(monkeypatch):
    monkeypatch.setenv("ALERT_ENABLED", "true")
    monkeypatch.setenv("NTFY_TOPIC", "btc-eth-alertas-x7f9k2")
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    monkeypatch.delenv("NTFY_USER", raising=False)
    monkeypatch.delenv("NTFY_PASSWORD", raising=False)
    monkeypatch.delenv("NTFY_TOKEN", raising=False)

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

    def _fake_post(url, json=None, auth=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = auth
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr("src.ntfy_notify.requests.post", _fake_post)

    result = send_ntfy_alert("BTC/USDT: confluencia LARGO activa", "mensaje con ñ y tildes", tags=["rotating_light"])

    assert result is True
    assert captured["url"] == "https://ntfy.sh/"
    assert captured["json"]["topic"] == "btc-eth-alertas-x7f9k2"
    assert captured["json"]["title"] == "BTC/USDT: confluencia LARGO activa"
    assert captured["json"]["message"] == "mensaje con ñ y tildes"
    assert captured["json"]["tags"] == ["rotating_light"]
    assert captured["auth"] is None
    assert captured["headers"] is None


def test_uses_custom_server_and_basic_auth_when_provided(monkeypatch):
    monkeypatch.setenv("ALERT_ENABLED", "true")
    monkeypatch.setenv("NTFY_TOPIC", "mytopic")
    monkeypatch.setenv("NTFY_SERVER", "https://ntfy.example.com/")
    monkeypatch.setenv("NTFY_USER", "juan")
    monkeypatch.setenv("NTFY_PASSWORD", "secreta")
    monkeypatch.delenv("NTFY_TOKEN", raising=False)

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

    def _fake_post(url, json=None, auth=None, headers=None, timeout=None):
        captured["url"] = url
        captured["auth"] = auth
        return _FakeResponse()

    monkeypatch.setattr("src.ntfy_notify.requests.post", _fake_post)

    assert send_ntfy_alert("t", "m") is True
    assert captured["url"] == "https://ntfy.example.com/"
    assert captured["auth"] == ("juan", "secreta")


def test_returns_false_without_raising_on_request_failure(monkeypatch):
    monkeypatch.setenv("ALERT_ENABLED", "true")
    monkeypatch.setenv("NTFY_TOPIC", "mytopic")

    def _fake_post(*args, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr("src.ntfy_notify.requests.post", _fake_post)

    assert send_ntfy_alert("t", "m") is False
