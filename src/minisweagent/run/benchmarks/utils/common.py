"""Shared agent utilities for benchmark runners."""

import time

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils import perf_metrics
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
        self._last_tool_end_ts: float | None = None

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.n_calls + 1:3d} (${self.cost:.2f})",
        )
        return super().step()

    def query(self) -> dict:
        prev_tool_end_ts = self._last_tool_end_ts
        message = super().query()
        first_token_ts = message.get("extra", {}).get("first_token_timestamp")
        if prev_tool_end_ts is not None and first_token_ts is not None:
            perf_metrics.record_tool_iteration_ttft(
                instance_id=self.instance_id,
                step=self.n_calls,
                tool_end_ts=prev_tool_end_ts,
                first_token_ts=first_token_ts,
            )
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        last_tool_end_ts = None
        for action in message.get("extra", {}).get("actions", []):
            outputs.append(self.env.execute(action))
            last_tool_end_ts = time.time()
        if last_tool_end_ts is not None:
            self._last_tool_end_ts = last_tool_end_ts
        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )
