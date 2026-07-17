from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent


class _ProgressManager:
    def update_instance_status(self, instance_id: str, message: str) -> None:
        pass


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            model_name="hosted_vllm/test-model",
            model_kwargs={"extra_headers": {"X-Test": "preserved", "x-request-id": "replaced"}},
        )
        self.query_kwargs = None

    def query(self, messages: list[dict], **kwargs) -> dict:
        self.query_kwargs = kwargs
        return {
            "role": "assistant",
            "content": "",
            "extra": {
                "actions": [{"command": "first"}, {"command": "second"}],
                "cost": 0.0,
                "response": {"id": "chatcmpl-mswea-task__1-1"},
            },
        }

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict) -> list[dict]:
        return [{"role": "user", "content": output["output"]} for output in outputs]

    def get_template_vars(self) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}


class _Environment:
    def __init__(self):
        self.actions = []

    def execute(self, action: dict) -> dict:
        self.actions.append(action)
        return {"output": action["command"], "returncode": 0, "exception_info": ""}

    def get_template_vars(self) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}


def _agent(env=None) -> ProgressTrackingAgent:
    return ProgressTrackingAgent(
        _Model(),
        env or _Environment(),
        progress_manager=_ProgressManager(),
        instance_id="task__1",
        system_template="system",
        instance_template="task",
        cost_limit=0,
    )


def test_records_model_and_each_tool_call_wall_time():
    agent = _agent()

    with patch(
        "minisweagent.run.benchmarks.utils.common.time.perf_counter",
        side_effect=[10.0, 10.25, 20.0, 20.1, 30.0, 30.2],
    ):
        agent.step()

    assert agent.model.query_kwargs == {"extra_headers": {"X-Test": "preserved", "X-Request-Id": "mswea-task__1-1"}}
    assert agent.model_query_s == pytest.approx(0.25)
    assert agent.tool_execution_s == pytest.approx(0.3)
    assert agent.tool_calls == 2
    assert agent.model_call_records[0]["request_id"] == "chatcmpl-mswea-task__1-1"
    assert agent.model_call_records[0]["client_request_id"] == "mswea-task__1-1"
    assert agent.model_call_records[0]["client_duration_s"] == pytest.approx(0.25)
    assert agent.model_call_records[0]["status"] == "ok"


def test_tool_time_is_recorded_when_execute_raises():
    class _SubmittingEnvironment(_Environment):
        def execute(self, action: dict) -> dict:
            raise Submitted({"role": "exit", "content": "done", "extra": {}})

    agent = _agent(_SubmittingEnvironment())
    message = {"extra": {"actions": [{"command": "submit"}]}}

    with (
        patch("minisweagent.run.benchmarks.utils.common.time.perf_counter", side_effect=[1.0, 1.4]),
        pytest.raises(Submitted),
    ):
        agent.execute_actions(message)

    assert agent.tool_calls == 1
    assert agent.tool_execution_s == pytest.approx(0.4)
