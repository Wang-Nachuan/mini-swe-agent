"""Shared agent utilities for benchmark runners."""

import time

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


class ProgressTrackingAgent(DefaultAgent):
    """Agent that reports per-step progress via :class:`RunBatchProgressManager`."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id
        self.model_query_s = 0.0
        self.tool_execution_s = 0.0
        self.tool_calls = 0
        self.model_call_records: list[dict] = []

    def query(self, **model_kwargs) -> dict:
        call_index = self.n_calls + 1
        client_request_id = f"mswea-{self.instance_id}-{call_index}"
        model_name = getattr(self.model.config, "model_name", "")
        query_kwargs = dict(model_kwargs)
        if model_name.startswith("hosted_vllm/"):
            config_model_kwargs = getattr(self.model.config, "model_kwargs", {})
            headers = {
                key: value
                for key, value in config_model_kwargs.get("extra_headers", {}).items()
                if key.lower() != "x-request-id"
            }
            headers.update(
                {
                    key: value
                    for key, value in query_kwargs.get("extra_headers", {}).items()
                    if key.lower() != "x-request-id"
                }
            )
            headers["X-Request-Id"] = client_request_id
            query_kwargs["extra_headers"] = headers

        start_ts = time.time()
        start_time = time.perf_counter()
        message = None
        status = "ok"
        try:
            message = super().query(**query_kwargs)
        except BaseException as e:
            status = type(e).__name__
            raise
        finally:
            if self.n_calls == call_index:
                end_ts = time.time()
                duration_s = time.perf_counter() - start_time
                response = (message or {}).get("extra", {}).get("response", {})
                usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
                request_id = response.get("id") if isinstance(response, dict) else None
                if not request_id:
                    request_id = (
                        f"chatcmpl-{client_request_id}" if model_name.startswith("hosted_vllm/") else client_request_id
                    )
                self.model_query_s += duration_s
                self.model_call_records.append(
                    {
                        "instance_id": self.instance_id,
                        "call_index": call_index,
                        "request_id": request_id,
                        "client_request_id": client_request_id,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "client_duration_s": duration_s,
                        "status": status,
                        "prompt_tokens": usage.get("prompt_tokens", ""),
                        "completion_tokens": usage.get("completion_tokens", ""),
                    }
                )
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            start_time = time.perf_counter()
            try:
                outputs.append(self.env.execute(action))
            finally:
                self.tool_execution_s += time.perf_counter() - start_time
                self.tool_calls += 1
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.n_calls + 1:3d} (${self.cost:.2f})",
        )
        return super().step()
