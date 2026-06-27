import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if os.getenv("MSWEA_PERF_ENABLE_STREAMING") == "1":
            return self._query_streaming(messages, **kwargs)

        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    @staticmethod
    def _as_dict(value: Any) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }

    @staticmethod
    def _delta_has_output(delta: dict) -> bool:
        if delta.get("content"):
            return True
        for tool_call in delta.get("tool_calls") or []:
            tool_call = LitellmModel._as_dict(tool_call)
            function = LitellmModel._as_dict(tool_call.get("function"))
            if tool_call.get("id") or tool_call.get("type"):
                return True
            if function.get("name") or function.get("arguments"):
                return True
        return False

    @staticmethod
    def _tool_call_namespace(tool_call: dict) -> SimpleNamespace:
        function = tool_call.get("function") or {}
        return SimpleNamespace(
            id=tool_call.get("id"),
            function=SimpleNamespace(
                name=function.get("name"),
                arguments=function.get("arguments", ""),
            ),
        )

    def _query_streaming(self, messages: list[dict[str, str]], **kwargs) -> dict:
        prepared_messages = self._prepare_messages_for_api(messages)
        request_start_ts = time.time()

        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                content_parts: list[str] = []
                tool_calls_by_index: dict[int, dict] = {}
                first_token_ts = None
                response_id = None
                response_model = None
                response_created = None
                finish_reason = None
                usage = None

                stream = self._query(prepared_messages, stream=True, **kwargs)
                for chunk in stream:
                    chunk_ts = time.time()
                    chunk_dict = self._as_dict(chunk)
                    response_id = response_id or chunk_dict.get("id")
                    response_model = response_model or chunk_dict.get("model")
                    response_created = response_created or chunk_dict.get("created")
                    usage = chunk_dict.get("usage") or usage

                    choices = chunk_dict.get("choices") or []
                    if not choices:
                        continue
                    choice = self._as_dict(choices[0])
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = self._as_dict(choice.get("delta"))

                    if first_token_ts is None and self._delta_has_output(delta):
                        first_token_ts = chunk_ts

                    if delta.get("content"):
                        content_parts.append(delta["content"])

                    for tool_call in delta.get("tool_calls") or []:
                        tool_call = self._as_dict(tool_call)
                        index = int(tool_call.get("index") or 0)
                        merged = tool_calls_by_index.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_call.get("id"):
                            merged["id"] = tool_call["id"]
                        if tool_call.get("type"):
                            merged["type"] = tool_call["type"]

                        function_delta = self._as_dict(tool_call.get("function"))
                        if function_delta.get("name"):
                            merged["function"]["name"] += function_delta["name"]
                        if function_delta.get("arguments"):
                            merged["function"]["arguments"] += function_delta[
                                "arguments"
                            ]

        content = "".join(content_parts)
        tool_calls = [
            tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
        ]
        action_tool_calls = [self._tool_call_namespace(tc) for tc in tool_calls]
        message = {
            "content": content,
            "role": "assistant",
            "tool_calls": tool_calls,
            "function_call": None,
        }
        response_dump = {
            "id": response_id,
            "created": response_created,
            "model": response_model,
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": message.copy(),
                }
            ],
            "usage": usage,
        }
        message["extra"] = {
            "actions": parse_toolcall_actions(
                action_tool_calls,
                format_error_template=self.config.format_error_template,
            ),
            "response": response_dump,
            "cost": 0.0,
            "timestamp": time.time(),
            "request_start_timestamp": request_start_ts,
            "first_token_timestamp": first_token_ts,
        }
        GLOBAL_MODEL_STATS.add(0.0)
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
