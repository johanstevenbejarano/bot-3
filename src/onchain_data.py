"""Descarga y cachea "direcciones activas" on-chain -- BTC vía blockchain.info (gratis, sin clave,
histórico diario completo), ETH vía la exportación CSV pública de Etherscan (gratis, sin clave,
histórico diario completo desde 2015). Es la primera fuente de información on-chain probada en
toda la sesión: actividad real de la red, no precio/volumen/derivados/sentimiento/macro.

A diferencia de funding rate o el Fear & Greed Index (una sola serie compartida por ambos
símbolos), acá cada símbolo tiene su PROPIA cadena y su propia fuente -- no hay atajo, hay que
traer los dos por separado.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

BLOCKCHAIN_INFO_URL = "https://api.blockchain.info/charts/n-unique-addresses"
ETHERSCAN_CSV_URL = "https://etherscan.io/chart/active-address"

_BTC_CACHE_FILE = DATA_DIR / "btc_active_addresses.parquet"
_ETH_CACHE_FILE = DATA_DIR / "eth_active_addresses.parquet"


def fetch_btc_active_addresses(years: int = 9, force_refresh: bool = False) -> pd.Series:
    if _BTC_CACHE_FILE.exists() and not force_refresh:
        return pd.read_parquet(_BTC_CACHE_FILE)["value"]

    response = requests.get(
        BLOCKCHAIN_INFO_URL,
        params={"timespan": f"{years}years", "format": "json", "sampled": "false"},
        timeout=20,
    )
    response.raise_for_status()
    values = response.json()["values"]

    df = pd.DataFrame({"timestamp": [v["x"] for v in values], "value": [float(v["y"]) for v in values]})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df.to_parquet(_BTC_CACHE_FILE)
    logger.info("BTC direcciones activas: %d registros en caché (%s -> %s)", len(df), df.index[0], df.index[-1])
    return df["value"]


def fetch_eth_active_addresses(force_refresh: bool = False) -> pd.Series:
    if _ETH_CACHE_FILE.exists() and not force_refresh:
        return pd.read_parquet(_ETH_CACHE_FILE)["value"]

    response = requests.get(
        ETHERSCAN_CSV_URL, params={"output": "csv"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=20
    )
    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.text))
    df = df.rename(columns={"Date(UTC)": "date", "Unique Address Total Count": "value"})
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp")[["value"]].sort_index()
    df["value"] = df["value"].astype(float)
    df = df[~df.index.duplicated(keep="last")]

    df.to_parquet(_ETH_CACHE_FILE)
    logger.info("ETH direcciones activas: %d registros en caché (%s -> %s)", len(df), df.index[0], df.index[-1])
    return df["value"]


def fetch_active_addresses(symbol: str, force_refresh: bool = False) -> pd.Series:
    if symbol.startswith("BTC"):
        return fetch_btc_active_addresses(force_refresh=force_refresh)
    if symbol.startswith("ETH"):
        return fetch_eth_active_addresses(force_refresh=force_refresh)
    raise ValueError(f"Sin fuente on-chain configurada para {symbol!r}")
