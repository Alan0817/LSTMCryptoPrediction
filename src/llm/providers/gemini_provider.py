"""Google Gemini API adapter."""

import os

from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class GeminiProvider:
    """Generate text with the official Google Gen AI SDK."""

    def __init__(self, model: str | None = None, api_client=None):
        self.model = (
            model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        )
        if api_client is not None:
            self._client = api_client
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set to initialize the Gemini provider.")
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        request = {"model": self.model, "contents": prompt}
        if system_prompt is not None:
            request["config"] = {"system_instruction": system_prompt}

        response = self._client.models.generate_content(**request)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned a response without text output.")
        return text
