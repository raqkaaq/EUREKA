"""Strata configuration and legacy compatibility; no external services."""

import os
import unittest
from unittest import mock

from radar.provider import strata


class TestStrataConfiguration(unittest.TestCase):
    def test_strata_settings_win_over_retired_provider(self):
        with mock.patch.dict(os.environ, {
            "STRATA_BASE_URL": "http://192.168.50.8:8080/v1",
            "STRATA_MODEL": "strata-qwen",
            "STRATA_API_KEY": "strata-test-key",
            "FREETOKEN_BASE_URL": "http://127.0.0.1:1919/v1",
            "FREETOKEN_MODEL": "retired-qwen",
            "FREETOKEN_API_KEY": "retired-test-key",
        }, clear=True):
            config = strata.StrataConfig.resolve()
        self.assertEqual(config.base_url, "http://192.168.50.8:8080/v1")
        self.assertEqual(config.model, "strata-qwen")
        self.assertEqual(config.api_key, "strata-test-key")

    def test_explicit_settings_win_and_legacy_settings_still_work(self):
        with mock.patch.dict(os.environ, {
            "FREETOKEN_BASE_URL": "http://127.0.0.1:1919/v1",
            "FREETOKEN_MODEL": "legacy-qwen",
            "FREETOKEN_API_KEY": "legacy-test-key",
        }, clear=True):
            legacy = strata.StrataConfig.resolve()
            explicit = strata.StrataConfig.resolve(
                base_url="http://127.0.0.1:8080/v1", model="explicit-qwen",
                api_key="explicit-test-key")
        self.assertEqual(legacy.model, "legacy-qwen")
        self.assertEqual(legacy.api_key, "legacy-test-key")
        self.assertEqual(explicit.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(explicit.model, "explicit-qwen")
        self.assertEqual(explicit.api_key, "explicit-test-key")

    def test_strata_false_disables_legacy_thinking_override(self):
        with mock.patch.dict(os.environ, {
            "STRATA_DISABLE_THINKING": "false",
            "FREETOKEN_DISABLE_THINKING": "1",
        }, clear=True):
            self.assertFalse(strata.resolve_disable_thinking())
            self.assertTrue(strata.resolve_disable_thinking(True))

    def test_missing_endpoint_stays_actionable_and_no_lan_default_is_assumed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(strata.StrataError) as caught:
                strata.resolve_base_url()
        self.assertIn("STRATA_BASE_URL", str(caught.exception))

    def test_legacy_names_refer_to_the_same_contracts(self):
        from radar.provider import freetoken
        self.assertIs(freetoken.FreeTokenConfig, strata.StrataConfig)
        self.assertIs(freetoken.FreeTokenError, strata.StrataError)
        self.assertIs(freetoken.FreeTokenSession, strata.StrataSession)

    def test_invalid_strata_endpoint_does_not_fall_back_to_legacy(self):
        with mock.patch.dict(os.environ, {
            "STRATA_BASE_URL": "https://8.8.8.8/v1",
            "FREETOKEN_BASE_URL": "http://127.0.0.1:1919/v1",
        }, clear=True):
            with self.assertRaises(strata.StrataError):
                strata.resolve_base_url()

    def test_default_key_and_thinking_setting_are_provider_neutral(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = strata.StrataConfig.resolve(
                base_url="http://127.0.0.1:8080/v1", model="test-qwen")
            self.assertFalse(strata.resolve_disable_thinking())
        self.assertEqual(config.api_key, "strata-local")

    def test_model_discovery_uses_the_selected_api_key(self):
        import httpx

        def serve(request):
            self.assertEqual(request.headers.get("authorization"), "Bearer explicit-test-key")
            return httpx.Response(200, json={"data": [{"id": "discovered-qwen"}]})

        client = httpx.Client(transport=httpx.MockTransport(serve))
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "env-test-key"}, clear=True), \
             mock.patch.object(strata._httpx, "Client", return_value=client):
            config = strata.StrataConfig.resolve(
                base_url="http://127.0.0.1:8080/v1", api_key="explicit-test-key")
        self.assertEqual(config.model, "discovered-qwen")
        self.assertEqual(config.api_key, "explicit-test-key")


if __name__ == "__main__":
    unittest.main()
class TestBoundedTransport(unittest.TestCase):
    def test_provider_does_not_hide_network_retries_from_stage_budget(self):
        from unittest import mock
        from radar.provider.strata import StrataConfig, build_session, close_session
        import httpx2

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(
            lambda request: httpx2.Response(503)))
        config = StrataConfig(base_url="http://127.0.0.1:8080/v1", model="test")
        with mock.patch("radar.provider.strata._provider_http_client", return_value=client):
            session = build_session(config)
        try:
            self.assertEqual(session.model.client.max_retries, 0)
        finally:
            close_session(session)
