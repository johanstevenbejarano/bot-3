import requests.adapters

from src.live_snapshot import _patch_requests_ca_bundle_for_sandbox


def test_patches_default_bundle_when_env_var_points_to_real_file(tmp_path, monkeypatch):
    fake_bundle = tmp_path / "ca-bundle.crt"
    fake_bundle.write_text("fake cert")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(fake_bundle))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    original = requests.adapters.DEFAULT_CA_BUNDLE_PATH
    try:
        _patch_requests_ca_bundle_for_sandbox()
        assert requests.adapters.DEFAULT_CA_BUNDLE_PATH == str(fake_bundle)
    finally:
        requests.adapters.DEFAULT_CA_BUNDLE_PATH = original


def test_falls_back_to_ssl_cert_file_when_requests_ca_bundle_unset(tmp_path, monkeypatch):
    fake_bundle = tmp_path / "ca-bundle.crt"
    fake_bundle.write_text("fake cert")
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(fake_bundle))
    original = requests.adapters.DEFAULT_CA_BUNDLE_PATH
    try:
        _patch_requests_ca_bundle_for_sandbox()
        assert requests.adapters.DEFAULT_CA_BUNDLE_PATH == str(fake_bundle)
    finally:
        requests.adapters.DEFAULT_CA_BUNDLE_PATH = original


def test_noop_when_no_env_var_set(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    original = requests.adapters.DEFAULT_CA_BUNDLE_PATH
    try:
        _patch_requests_ca_bundle_for_sandbox()
        assert requests.adapters.DEFAULT_CA_BUNDLE_PATH == original
    finally:
        requests.adapters.DEFAULT_CA_BUNDLE_PATH = original


def test_noop_when_env_var_points_to_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "does-not-exist.crt"))
    original = requests.adapters.DEFAULT_CA_BUNDLE_PATH
    try:
        _patch_requests_ca_bundle_for_sandbox()
        assert requests.adapters.DEFAULT_CA_BUNDLE_PATH == original
    finally:
        requests.adapters.DEFAULT_CA_BUNDLE_PATH = original
