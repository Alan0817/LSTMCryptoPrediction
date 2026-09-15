"""Provider-agnostic text generation facade with no tool or agent behavior."""

import os
from pathlib import Path

from dotenv import load_dotenv

from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_PROVIDER = "openai"


class LLMClient:
    """Generate text through the configured provider.

    ``api_client`` is dependency injection for tests and local adapters. It is
    passed only to the selected provider and never used by the ML pipeline.
    """

    def __init__(self, provider: str | None = None, model: str | None = None, api_client=None):
        load_dotenv(ENV_FILE)
        self.provider_name = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
        provider_class = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
        }.get(self.provider_name)
        if provider_class is None:
            raise ValueError(
                "Unsupported LLM provider: {!r}. Supported providers are 'openai' and 'gemini'."
                .format(self.provider_name)
            )
        self._provider = provider_class(model=model, api_client=api_client)
        self.model = self._provider.model

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text without tools, retrieval, or conversation state."""
        return self._provider.generate(prompt, system_prompt)
