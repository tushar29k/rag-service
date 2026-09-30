"""Tests for llm_client.py — the free-tier provider client.

    python3 test_llm_client.py
Never touches a real API: urlopen is stubbed, so these run offline.
"""
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import llm_client
from llm_client import FreeLLMClient, FreeLLMError


def _resp(payload):
    r = mock.Mock()
    r.read.return_value = json.dumps(payload).encode()
    return r


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO())


class ClientTest(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        for var in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                    "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            os.environ.pop(var, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_no_key_means_no_client(self):
        # the whole point: unset key -> caller keeps its mock path
        self.assertIsNone(FreeLLMClient.from_env())

    def test_unknown_provider_rejected(self):
        os.environ["LLM_API_KEY"] = "k"
        with self.assertRaises(FreeLLMError):
            FreeLLMClient(provider="nope")

    def test_gemini_request_shape(self):
        os.environ.update(LLM_API_KEY="secret-key", LLM_PROVIDER="gemini")
        payload = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        with mock.patch.object(llm_client.urllib.request, "urlopen",
                               return_value=_resp(payload)) as u, \
             mock.patch.object(llm_client.time, "sleep"):
            c = FreeLLMClient.from_env()
            self.assertEqual(c.generate("hello"), "hi")
            req = u.call_args[0][0]
            url = req.full_url
            self.assertIn("generativelanguage.googleapis.com", url)
            self.assertIn("gemini-3.8-flash", url)  # default model
            body = json.loads(req.data.decode())
            self.assertEqual(body["contents"][0]["parts"][0]["text"], "hello")

    def test_openrouter_request_shape(self):
        os.environ.update(LLM_API_KEY="secret-key",
                          LLM_PROVIDER="openrouter")
        payload = {"choices": [{"message": {"content": "yo"}}]}
        with mock.patch.object(llm_client.urllib.request, "urlopen",
                               return_value=_resp(payload)) as u, \
             mock.patch.object(llm_client.time, "sleep"):
            c = FreeLLMClient.from_env()
            self.assertEqual(c.generate("hello"), "yo")
            req = u.call_args[0][0]
            self.assertEqual(req.full_url,
                             "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(req.get_header("Authorization"),
                             "Bearer secret-key")
            body = json.loads(req.data.decode())
            self.assertIn(":free", body["model"])  # free-tier slug only

    def test_model_override(self):
        os.environ.update(LLM_API_KEY="k", LLM_MODEL="custom-model")
        c = FreeLLMClient.from_env()
        self.assertEqual(c.model, "custom-model")

    def test_retry_once_on_429_then_success(self):
        os.environ.update(LLM_API_KEY="k", LLM_PROVIDER="openrouter")
        payload = {"choices": [{"message": {"content": "recovered"}}]}
        with mock.patch.object(
                llm_client.urllib.request, "urlopen",
                side_effect=[_http_error(429), _resp(payload)]) as u, \
             mock.patch.object(llm_client.time, "sleep") as s:
            c = FreeLLMClient.from_env()
            self.assertEqual(c.generate("hello"), "recovered")
            self.assertEqual(u.call_count, 2)
            s.assert_called_once()  # one backoff, not a hammer loop

    def test_total_failure_raises_without_key_in_message(self):
        os.environ.update(LLM_API_KEY="super-secret-key",
                          LLM_PROVIDER="gemini")
        with mock.patch.object(llm_client.urllib.request, "urlopen",
                               side_effect=[_http_error(500),
                                            _http_error(503)]), \
             mock.patch.object(llm_client.time, "sleep"):
            c = FreeLLMClient.from_env()
            with self.assertRaises(FreeLLMError) as ctx:
                c.generate("hello")
            # the key must never appear in error text (logs, UI, anywhere)
            self.assertNotIn("super-secret-key", str(ctx.exception))

    def test_bad_response_shape_raises(self):
        os.environ.update(LLM_API_KEY="k", LLM_PROVIDER="gemini")
        with mock.patch.object(llm_client.urllib.request, "urlopen",
                               return_value=_resp({"nope": 1})), \
             mock.patch.object(llm_client.time, "sleep"):
            c = FreeLLMClient.from_env()
            with self.assertRaises(FreeLLMError):
                c.generate("hello")

    def test_openai_request_shape(self):
        os.environ.update(OPENAI_API_KEY="sk-test", LLM_MODEL="gpt-4o-mini")
        payload = {"choices": [{"message": {"content": "cited answer"}}]}
        with mock.patch.object(llm_client.urllib.request, "urlopen",
                               return_value=_resp(payload)) as u, \
             mock.patch.object(llm_client.time, "sleep"):
            c = FreeLLMClient.for_generator("openai")
            self.assertEqual(c.generate("hello"), "cited answer")
            req = u.call_args[0][0]
            self.assertEqual(req.full_url,
                             "https://api.openai.com/v1/chat/completions")
            self.assertEqual(req.get_header("Authorization"),
                             "Bearer sk-test")
            body = json.loads(req.data.decode())
            self.assertEqual(body["model"], "gpt-4o-mini")
            self.assertEqual(body["messages"][0]["role"], "user")

    def test_openai_accepts_shared_llm_key(self):
        # no OPENAI_API_KEY — the shared LLM_API_KEY still works
        os.environ.update(LLM_API_KEY="shared")
        c = FreeLLMClient.for_generator("openai")
        self.assertEqual(c.provider, "openai")
        self.assertEqual(c.api_key, "shared")

    def test_openai_prefers_openai_key(self):
        os.environ.update(OPENAI_API_KEY="sk-openai", LLM_API_KEY="shared")
        c = FreeLLMClient.for_generator("openai")
        self.assertEqual(c.api_key, "sk-openai")

    def test_openai_base_url_override(self):
        os.environ.update(OPENAI_API_KEY="k",
                          OPENAI_BASE_URL="http://localhost:11434/v1")
        c = FreeLLMClient.for_generator("openai")
        self.assertEqual(c._request("q", 8, 0).full_url,
                         "http://localhost:11434/v1/chat/completions")

    def test_openai_no_key_falls_back_to_none(self):
        # no key at all -> None, the caller keeps its mock path
        self.assertIsNone(FreeLLMClient.for_generator("openai"))

    def test_generator_mock_ignores_key(self):
        os.environ.update(LLM_API_KEY="k")
        self.assertIsNone(FreeLLMClient.for_generator("mock"))

    def test_generator_auto_without_key_is_none(self):
        self.assertIsNone(FreeLLMClient.for_generator("auto"))

    def test_generator_typo_fails_loud(self):
        os.environ.update(LLM_API_KEY="k")
        with self.assertRaises(FreeLLMError):
            FreeLLMClient.for_generator("openi")


if __name__ == "__main__":
    unittest.main(verbosity=1)
