"""Tests for the FreeToken local-network model adapter.

No network, no real LLM: the /models lookup is faked with an httpx
MockTransport-backed client, and endpoint policy is validated without
contacting a real server.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest import mock

from radar.provider import freetoken
from radar.provider.freetoken import (
    EXAMPLE_BASE_URL,
    FreeTokenConfig,
    FreeTokenError,
    check_local_network,
    list_models,
    resolve_base_url,
    resolve_model,
)


@contextmanager
def _fake_models_client(body: bytes | Exception):
    """Serve a canned /models payload without touching the network."""
    import httpx

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None, timeout=None):
            import httpx as _real_httpx

            if isinstance(body, Exception):
                raise body
            return _real_httpx.Response(
                200, content=body,
                request=_real_httpx.Request("GET", "http://127.0.0.1:1919/v1/models"))

    with mock.patch("radar.provider.strata._httpx.Client", _FakeClient):
        yield


class _EnvCleaner:
    """Temporarily remove env vars (restoring afterwards)."""

    def __init__(self, names):
        self.names = names
        self.saved = {}

    def __enter__(self):
        for name in self.names:
            self.saved[name] = os.environ.pop(name, None)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


class TestLocalNetworkPolicy(unittest.TestCase):
    def test_loopback_and_private_ip_hosts_accepted(self):
        for url in (
            "http://127.0.0.1:1919/v1",
            "http://localhost:1919/v1",
            "http://[::1]:1919/v1",
            "https://127.0.0.1:1919/v1",
            "http://192.168.1.10:1919/v1",
            "http://10.0.0.5:1919/v1",
            "http://172.20.4.8:1919/v1",
            "http://100.100.20.30:1919/v1",
            "http://[fd12:3456::10]:1919/v1",
        ):
            self.assertTrue(check_local_network(url).startswith("http"))

    def test_private_lan_hostname_accepted_after_resolution(self):
        answer = [(2, 1, 6, "", ("192.168.50.8", 0))]
        with mock.patch("socket.getaddrinfo", return_value=answer):
            self.assertEqual(
                check_local_network("http://freetoken.home:1919/v1"),
                "http://freetoken.home:1919/v1",
            )

    def test_missing_endpoint_is_actionable(self):
        with _EnvCleaner(("FREETOKEN_BASE_URL",)):
            with self.assertRaises(FreeTokenError) as ctx:
                resolve_base_url()
            self.assertIn("FREETOKEN_BASE_URL", str(ctx.exception))

    def test_public_or_unsafe_hosts_rejected(self):
        bad = [
            "https://api.openai.com/v1",
            "http://8.8.8.8/v1",
            "http://127.0.0.1.evil.com/v1",
            "http://169.254.169.254/v1",
            "http://0.0.0.0:1919/v1",
            "not-a-url",
            "ftp://127.0.0.1/v1",
        ]
        for url in bad:
            with self.subTest(url=url):
                with self.assertRaises(FreeTokenError):
                    check_local_network(url)
                with self.assertRaises(FreeTokenError):
                    FreeTokenConfig.resolve(base_url=url, model="m")

    def test_hostname_resolving_public_address_is_rejected(self):
        answer = [(2, 1, 6, "", ("93.184.216.34", 0))]
        with mock.patch("socket.getaddrinfo", return_value=answer):
            with self.assertRaises(FreeTokenError):
                check_local_network("http://model.example:1919/v1")

    def test_empty_base_url_means_unset(self):
        # Empty/whitespace behaves as "not provided" and requires configuration.
        with _EnvCleaner(("FREETOKEN_BASE_URL",)):
            with self.assertRaises(FreeTokenError):
                resolve_base_url("")
            with self.assertRaises(FreeTokenError):
                check_local_network("")

    def test_link_local_env_rejected(self):
        with _EnvCleaner(("FREETOKEN_BASE_URL",)):
            os.environ["FREETOKEN_BASE_URL"] = "http://169.254.169.254/v1"
            with self.assertRaises(FreeTokenError):
                resolve_base_url()

    def test_explicit_overrides_env(self):
        with _EnvCleaner(("FREETOKEN_BASE_URL",)):
            os.environ["FREETOKEN_BASE_URL"] = "http://8.8.8.8/v1"
            resolved = resolve_base_url("http://127.0.0.1:1919/v1")
            self.assertEqual(resolved, "http://127.0.0.1:1919/v1")


class TestModelResolution(unittest.TestCase):
    def test_env_model_skips_network(self):
        with _EnvCleaner(("FREETOKEN_MODEL",)):
            os.environ["FREETOKEN_MODEL"] = "local-qwen"
            with mock.patch("radar.provider.strata._httpx.Client") as fake:
                self.assertEqual(resolve_model(EXAMPLE_BASE_URL), "local-qwen")
                fake.assert_not_called()

    def test_models_endpoint_parsed(self):
        body = b'{"data": [{"id": "model-a"}, {"id": "model-b"}]}'
        with _EnvCleaner(("FREETOKEN_MODEL",)):
            with _fake_models_client(body):
                self.assertEqual(resolve_model(EXAMPLE_BASE_URL), "model-a")

    def test_empty_models_list_rejected(self):
        with _EnvCleaner(("FREETOKEN_MODEL",)):
            with _fake_models_client(b'{"data": []}'):
                with self.assertRaises(FreeTokenError):
                    resolve_model(EXAMPLE_BASE_URL)

    def test_unreachable_endpoint_actionable(self):
        with _EnvCleaner(("FREETOKEN_MODEL",)):
            with _fake_models_client(ConnectionRefusedError("refused")):
                with self.assertRaises(FreeTokenError) as ctx:
                    list_models(EXAMPLE_BASE_URL)
                self.assertIn(EXAMPLE_BASE_URL, str(ctx.exception))

    def test_config_resolve_binds_all(self):
        with _EnvCleaner(("FREETOKEN_MODEL", "FREETOKEN_BASE_URL", "FREETOKEN_API_KEY")):
            os.environ["FREETOKEN_MODEL"] = "m1"
            os.environ["FREETOKEN_BASE_URL"] = "http://192.168.1.20:1919/v1"
            cfg = FreeTokenConfig.resolve()
            self.assertEqual(cfg.base_url, "http://192.168.1.20:1919/v1")
            self.assertEqual(cfg.model, "m1")
            self.assertTrue(cfg.api_key)


class TestNoDirectOpenAISDK(unittest.TestCase):
    def test_no_direct_openai_imports(self):
        """All LLM inference must go through PydanticAI (no openai SDK use)."""
        import pathlib

        package = pathlib.Path(freetoken.__file__).parent
        offenders: list[str] = []
        for path in sorted(package.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "from openai" in stripped or stripped.startswith("import openai"):
                    # 'pydantic_ai.models.openai' / 'providers.openai' are fine.
                    if "pydantic_ai" not in stripped and "providers.openai" not in stripped:
                        offenders.append(f"{path.name}:{lineno}: {stripped}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
