from types import SimpleNamespace

import pytest

from llm.client import LLMClient
from llm.providers.gemini_provider import DEFAULT_GEMINI_MODEL
from llm.providers.openai_provider import DEFAULT_OPENAI_MODEL


class FakeOpenAIResponses:
    def __init__(self, output_text="OpenAI response"):
        self.output_text = output_text
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeGeminiModels:
    def __init__(self, text="Gemini response"):
        self.text = text
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text=self.text)


def make_openai_client(output_text="OpenAI response"):
    return SimpleNamespace(responses=FakeOpenAIResponses(output_text))


def make_gemini_client(text="Gemini response"):
    return SimpleNamespace(models=FakeGeminiModels(text))


def test_missing_openai_api_key_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("llm.client.load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be set"):
        LLMClient(provider="openai")


def test_missing_gemini_api_key_is_clear(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("llm.client.load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY must be set"):
        LLMClient(provider="gemini")


def test_openai_provider_initializes_with_fake_client():
    client = LLMClient(provider="openai", model="openai-test", api_client=make_openai_client())

    assert client.provider_name == "openai"
    assert client.model == "openai-test"


def test_gemini_provider_initializes_with_fake_client():
    client = LLMClient(provider="gemini", model="gemini-test", api_client=make_gemini_client())

    assert client.provider_name == "gemini"
    assert client.model == "gemini-test"


def test_openai_generate_returns_mocked_text():
    fake_client = make_openai_client("OpenAI RSI explanation")
    client = LLMClient(provider="openai", api_client=fake_client)

    assert client.generate("Explain RSI.", system_prompt="Be concise.") == "OpenAI RSI explanation"
    assert fake_client.responses.requests == [
        {
            "model": DEFAULT_OPENAI_MODEL,
            "input": "Explain RSI.",
            "instructions": "Be concise.",
        }
    ]


def test_gemini_generate_returns_mocked_text():
    fake_client = make_gemini_client("Gemini RSI explanation")
    client = LLMClient(provider="gemini", api_client=fake_client)

    assert client.generate("Explain RSI.", system_prompt="Be concise.") == "Gemini RSI explanation"
    assert fake_client.models.requests == [
        {
            "model": DEFAULT_GEMINI_MODEL,
            "contents": "Explain RSI.",
            "config": {"system_instruction": "Be concise."},
        }
    ]


def test_provider_selection_uses_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    client = LLMClient(api_client=make_gemini_client())

    assert client.provider_name == "gemini"


def test_explicit_model_overrides_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "environment-model")

    client = LLMClient(provider="gemini", model="explicit-model", api_client=make_gemini_client())

    assert client.model == "explicit-model"


def test_unsupported_provider_is_clear():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMClient(provider="unsupported", api_client=object())


@pytest.mark.parametrize(
    ("provider", "api_client", "message"),
    [
        ("openai", make_openai_client(output_text=""), "OpenAI returned a response without text output"),
        ("gemini", make_gemini_client(text=None), "Gemini returned a response without text output"),
    ],
)
def test_empty_provider_response_is_clear(provider, api_client, message):
    client = LLMClient(provider=provider, api_client=api_client)

    with pytest.raises(RuntimeError, match=message):
        client.generate("Explain RSI.")
