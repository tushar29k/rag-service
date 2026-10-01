"""Free-tier LLM client. stdlib only — no new deps, keeps Render light.

Env:
  LLM_PROVIDER  gemini (default) | openrouter | openai
  LLM_MODEL     override the default model for the provider
  LLM_API_KEY   the key for gemini/openrouter. FreeLLMClient.from_env()
                returns None without it, so callers keep their mock path
                with zero behavior change.
  OPENAI_API_KEY  key for the openai provider (LLM_API_KEY also works)
  OPENAI_BASE_URL point the openai provider at an OpenAI-compatible
                server (ollama, vLLM, ...) instead of api.openai.com

Gemini = Google AI Studio free tier (no card needed), key rides in the
query param — that's how their REST API takes it; it's never logged.
OpenRouter = OpenAI-compatible /chat/completions with a :free model slug.
OpenAI = the real /chat/completions API; same request shape as OpenRouter.

Resilience: 20s timeout, one retry with backoff on 429/5xx, then
FreeLLMError whose message never contains the key.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error


class FreeLLMError(RuntimeError):
    pass


_TIMEOUT = 20
_BACKOFF = 1.5

_DEFAULTS = {
    "gemini": "gemini-3.8-flash",
    "openrouter": "openai/gpt-oss-20b:free",
    "openai": "gpt-4o-mini",
}


def _base_url(provider):
    # openai-compatible servers (ollama, vLLM, proxies) can stand in
    # for the real API by setting OPENAI_BASE_URL — same request shape
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    return os.environ.get("OPENAI_BASE_URL",
                          "https://api.openai.com/v1")


def _key_for(provider):
    # OPENAI_API_KEY is the conventional name; the shared LLM_API_KEY
    # still works so one secret serves every provider
    if provider == "openai":
        return (os.environ.get("OPENAI_API_KEY")
                or os.environ.get("LLM_API_KEY"))
    return os.environ.get("LLM_API_KEY")


class FreeLLMClient:
    def __init__(self, provider=None, model=None, api_key=None):
        self.provider = (provider or os.environ.get("LLM_PROVIDER",
                                                     "gemini")).lower()
        if self.provider not in _DEFAULTS:
            raise FreeLLMError(f"unknown LLM_PROVIDER '{self.provider}' "
                               f"— want one of {sorted(_DEFAULTS)}")
        self.model = model or os.environ.get("LLM_MODEL") or \
            _DEFAULTS[self.provider]
        self.api_key = api_key or _key_for(self.provider)
        if not self.api_key:
            raise FreeLLMError("no API key in the environment")

    @classmethod
    def for_generator(cls, generator):
        """Map the config.yaml `generator:` value to a client or None.

        mock  -> None always: the extractive stand-in answers everything
        openai -> an openai provider client when a key is configured, else
                  None (caller falls back to mock — no key, no crash)
        auto  -> the pre-generator behavior: whatever LLM_PROVIDER + a key
                  configure, else None
        A typo'd value fails loud at startup instead of silently mocking.
        """
        generator = (generator or "auto").lower()
        if generator == "mock":
            return None
        if generator == "auto":
            return cls.from_env()
        if generator not in _DEFAULTS:
            raise FreeLLMError(
                f"unknown generator '{generator}' — want one of "
                f"mock, auto, {sorted(_DEFAULTS)}")
        if not _key_for(generator):
            print(f"rag: no key for generator '{generator}' — mock instead",
                  file=sys.stderr)
            return None
        return cls(provider=generator)

    @classmethod
    def from_env(cls):
        # None = no key configured → caller stays on its mock path
        if not os.environ.get("LLM_API_KEY"):
            return None
        return cls()

    @property
    def label(self):
        return f"{self.provider}:{self.model}"

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        req = self._request(prompt, max_tokens, temperature)
        return self._send(req)

    def generate_stream(self, prompt, max_tokens=512, temperature=0.2):
        """Yield answer text chunks as they arrive from the provider.

        The real-LLM path uses the provider's streaming HTTP endpoint
        (gemini streamGenerateContent, or "stream": true for the
        openai-compatible chat completions shape) and parses the SSE
        frames with one line-based parser — both shapes boil down to
        `data: {json}\n` lines ending in `data: [DONE]`.
        """
        req = self._request(prompt, max_tokens, temperature,
                            stream=True)
        try:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise FreeLLMError(
                f"{self.provider} unreachable ({type(e).__name__})") from None
        buf = ""
        try:
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    text = self._sse_text(line)
                    if text:
                        yield text
        except urllib.error.HTTPError as e:
            raise FreeLLMError(
                f"{self.provider} rejected the request "
                f"(HTTP {e.code})") from None

    def _sse_text(self, line):
        # one line from a streaming response: blank, a [DONE] sentinel,
        # or data: {json} — try the gemini shape, then the delta shape
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        try:
            return payload["choices"][0]["delta"].get("content") or None
        except (KeyError, IndexError, TypeError):
            return None

    def _request(self, prompt, max_tokens, temperature, stream=False):
        if self.provider == "gemini":
            action = "streamGenerateContent" if stream \
                else "generateContent"
            # alt=sse is what makes gemini actually speak SSE — without it
            # the endpoint returns a JSON array and the frame parser below
            # sees zero data: lines
            alt = "alt=sse&" if stream else ""
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model}:{action}?{alt}key={self.api_key}")
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens,
                                        "temperature": temperature}}
            data = json.dumps(body).encode()
            return urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json",
                                         "Accept": "text/event-stream"})
        # openrouter + openai: same openai-compatible chat completions
        # shape, different base URL — the key rides in the Authorization
        # header for both, never in the URL
        url = _base_url(self.provider) + "/chat/completions"
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature,
                "stream": stream}
        return urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})

    def _send(self, req):
        last = None
        for attempt in (0, 1):
            try:
                resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
                return self._parse(json.loads(resp.read().decode()))
            except urllib.error.HTTPError as e:
                last = e
                # 429/5xx are worth one retry; anything else is final
                if e.code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(_BACKOFF)
                    continue
                raise FreeLLMError(
                    f"{self.provider} rejected the request "
                    f"(HTTP {e.code})") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                if attempt == 0:
                    time.sleep(_BACKOFF)
                    continue
                raise FreeLLMError(
                    f"{self.provider} unreachable ({type(e).__name__})") from None
        # unreachable — both attempts raised without the typed branches
        # catching them (defensive; keeps the type checker honest)
        raise FreeLLMError(f"{self.provider} call failed ({last})") from None

    def _parse(self, payload):
        try:
            if self.provider == "gemini":
                return payload["candidates"][0]["content"]["parts"][0]["text"]
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise FreeLLMError(
                f"{self.provider} returned an unexpected response shape") \
                from None
