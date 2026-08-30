"""Notificaciones push a ntfy (https://ntfy.sh) cuando se activa una confluencia de señal.

Best-effort a propósito: la notificación es un extra sobre el dashboard, no algo de lo que
dependa generarlo. Si `ALERT_ENABLED`/`NTFY_TOPIC` no están configurados, o el request falla
(sin red, topic inválido, etc.), `send_ntfy_alert` devuelve `False` en vez de lanzar -- nunca
debe tumbar `live_snapshot.py`.

Usa el endpoint JSON de ntfy (`POST {server}/`, no headers) para evitar problemas de
codificación con tildes/ñ en título y mensaje, que si van como headers HTTP hay que
manipular a mano.
"""
from __future__ import annotations

import os

import requests


def send_ntfy_alert(title: str, message: str, tags: list[str] | None = None, priority: int = 4) -> bool:
    if os.environ.get("ALERT_ENABLED", "").strip().lower() != "true":
        return False

    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False

    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")

    payload: dict = {"topic": topic, "title": title, "message": message, "priority": priority}
    if tags:
        payload["tags"] = tags

    auth = None
    user = os.environ.get("NTFY_USER")
    password = os.environ.get("NTFY_PASSWORD")
    if user and password:
        auth = (user, password)

    headers = None
    token = os.environ.get("NTFY_TOKEN")
    if token:
        headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.post(server + "/", json=payload, auth=auth, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False
