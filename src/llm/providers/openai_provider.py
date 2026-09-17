"""OpenAI Responses API adapter."""

import os
import json

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

    def generate_with_tools(
        self,
        prompt: str,
        registry,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
        trace: list[dict] | None = None,
    ) -> str:
        """Use a manual Responses API function-call loop through ``registry`` only."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if not isinstance(max_tool_rounds, int) or max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be a positive integer.")
        if trace is not None and not isinstance(trace, list):
            raise TypeError("trace must be a list when provided.")
        if not hasattr(registry, "list_tools") or not hasattr(registry, "execute"):
            raise TypeError("registry must provide list_tools() and execute().")

        tools = [_to_openai_tool_definition(tool) for tool in registry.list_tools()]
        current_input = prompt
        previous_response_id = None
        for round_number in range(1, max_tool_rounds + 1):
            request = {
                "model": self.model,
                "input": current_input,
                "tools": tools,
            }
            if previous_response_id is not None:
                request["previous_response_id"] = previous_response_id
            if system_prompt is not None:
                request["instructions"] = system_prompt

            response = self._client.responses.create(**request)
            function_calls = [
                item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                text = getattr(response, "output_text", None)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("OpenAI returned a response without text output.")
                _append_trace(trace, {"event": "final_response"})
                return text

            response_id = getattr(response, "id", None)
            if not isinstance(response_id, str) or not response_id:
                raise RuntimeError("OpenAI response with function calls did not include an id.")
            function_outputs = []
            for call in function_calls:
                name = getattr(call, "name", None)
                arguments = _parse_function_arguments(getattr(call, "arguments", None), name)
                call_id = getattr(call, "call_id", None)
                if not isinstance(call_id, str) or not call_id:
                    raise RuntimeError("OpenAI function call did not include a call_id.")
                _append_trace(
                    trace,
                    {"event": "tool_requested", "name": name, "arguments": arguments, "round": round_number},
                )
                result = registry.execute(name, arguments)
                _append_trace(
                    trace,
                    {"event": "tool_completed", "name": name, "result": result, "round": round_number},
                )
                function_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    }
                )

            current_input = function_outputs
            previous_response_id = response_id

        raise RuntimeError("OpenAI exceeded the maximum of {} tool rounds.".format(max_tool_rounds))


def _to_openai_tool_definition(tool_definition) -> dict:
    """Translate a provider-neutral definition without duplicating its schema."""
    return {"type": "function", **tool_definition.as_dict()}


def _parse_function_arguments(arguments, name) -> dict:
    if not isinstance(arguments, str):
        raise TypeError("OpenAI function arguments for {!r} must be a JSON string.".format(name))
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError("OpenAI returned malformed JSON arguments for tool {!r}.".format(name)) from error
    if not isinstance(parsed, dict):
        raise TypeError("OpenAI function arguments for {!r} must decode to a JSON object.".format(name))
    return parsed


def _append_trace(trace: list[dict] | None, event: dict) -> None:
    if trace is not None:
        trace.append(event)
