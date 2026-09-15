"""OpenAI Responses API adapter."""

import os

from openai import OpenAI


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


class OpenAIProvider:
    """Generate text with the official OpenAI Responses API."""

    def __init__(self, model: str | None = None, api_client=None):
        self.model = (
            model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        )
        if api_client is not None:
            self._client = api_client
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set to initialize the OpenAI provider.")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        request = {"model": self.model, "input": prompt}
        if system_prompt is not None:
            request["instructions"] = system_prompt

        response = self._client.responses.create(**request)
        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("OpenAI returned a response without text output.")
        return text

    def generate_with_tools(self, *args, **kwargs) -> str:
        """Reject tool-enabled generation until an OpenAI-specific adapter exists."""
        raise NotImplementedError("Tool-enabled generation is not supported for the OpenAI provider.")
