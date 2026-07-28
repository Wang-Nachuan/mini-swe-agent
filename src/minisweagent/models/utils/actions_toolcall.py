"""Parse actions & format observations with toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    tool_name: str = "bash",
    argument_name: str = "command",
    allow_empty: bool = False,
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        if allow_empty:
            return []
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if tool_call.function.name != tool_name:
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or argument_name not in args:
            error_msg += f"Missing '{argument_name}' argument in {tool_name} tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append({"command": args[argument_name], "tool_call_id": tool_call.id})
    return actions


def parse_dynamic_toolcall_actions(
    tool_calls: list,
    *,
    tools: list[dict],
    format_error_template: str,
    allow_empty: bool = False,
) -> list[dict]:
    """Parse calls for query-scoped OpenAI tools while retaining typed arguments."""
    if not tool_calls:
        if allow_empty:
            return []
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )

    tool_names = {
        tool["function"]["name"]
        for tool in tools
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict)
    }
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        arguments = {}
        arguments_json = tool_call.function.arguments
        try:
            arguments = json.loads(arguments_json)
        except Exception as error:
            error_msg = f"Error parsing tool call arguments: {error}."
        if tool_call.function.name not in tool_names:
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(arguments, dict):
            error_msg += "Tool call arguments must be a JSON object."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append(
            {
                "name": tool_call.function.name,
                "arguments": arguments,
                "arguments_json": arguments_json,
                "tool_call_id": tool_call.id,
            }
        )
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
