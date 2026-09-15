"""Google Gemini API adapter, including manual registry-backed function calling."""

import os
import json

from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_MAX_TOOL_ROUNDS = 5


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

    def generate_with_tools(
        self,
        prompt: str,
        registry,
        system_prompt: str | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        trace: list[dict] | None = None,
    ) -> str:
        """Use Gemini's manual stateful function-call loop through ``registry`` only."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if not isinstance(max_tool_rounds, int) or max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be a positive integer.")
        if trace is not None and not isinstance(trace, list):
            raise TypeError("trace must be a list when provided.")
        if not hasattr(registry, "list_tools") or not hasattr(registry, "execute"):
            raise TypeError("registry must provide list_tools() and execute().")

        tools = [_to_gemini_tool_definition(tool) for tool in registry.list_tools()]
        current_input = prompt
        previous_interaction_id = None
        for _ in range(max_tool_rounds):
            request = {
                "model": self.model,
                "input": current_input,
                "tools": tools,
                "previous_interaction_id": previous_interaction_id,
            }
            if system_prompt is not None:
                request["system_instruction"] = system_prompt
            interaction = self._client.interactions.create(**request)
            function_calls = [
                step for step in getattr(interaction, "steps", []) if getattr(step, "type", None) == "function_call"
            ]
            if not function_calls:
                text = getattr(interaction, "output_text", None)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Gemini returned a response without text output.")
                _append_trace(trace, {"event": "final_response"})
                return text

            function_results = []
            for call in function_calls:
                name = getattr(call, "name", None)
                arguments = getattr(call, "arguments", None)
                _append_trace(trace, {"event": "tool_requested", "name": name, "arguments": arguments})
                result = registry.execute(name, arguments)
                # Registry results are JSON-safe, so a caller can inspect structured
                # limitations without trying to infer them from the model's prose.
                _append_trace(trace, {"event": "tool_completed", "name": name, "result": result})
                function_results.append(
                    {
                        "type": "function_result",
                        "name": name,
                        "call_id": getattr(call, "id", None),
                        "result": [{"type": "text", "text": json.dumps(result)}],
                    }
                )

            current_input = function_results
            previous_interaction_id = getattr(interaction, "id", None)

        raise RuntimeError("Gemini exceeded the maximum of {} tool rounds.".format(max_tool_rounds))


def _to_gemini_tool_definition(tool_definition) -> dict:
    """Translate a provider-neutral definition without changing its schema."""
    return {"type": "function", **tool_definition.as_dict()}


def _append_trace(trace: list[dict] | None, event: dict) -> None:
    if trace is not None:
        trace.append(event)
