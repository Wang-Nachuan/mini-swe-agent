"""Small CSV helpers for SWE-bench performance telemetry."""

import csv
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _metrics_dir() -> Path | None:
    path = os.getenv("MSWEA_PERF_METRICS_DIR", "")
    return Path(path) if path else None


def enabled() -> bool:
    return _metrics_dir() is not None


def _append_csv_rows(filename: str, header: list[str], rows: list[list[Any]]) -> None:
    if not rows:
        return
    metrics_dir = _metrics_dir()
    if metrics_dir is None:
        return

    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / filename
    with _LOCK:
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerows(rows)


def _append_csv(filename: str, header: list[str], row: list[Any]) -> None:
    _append_csv_rows(filename, header, [row])


def record_model_calls(records: list[dict[str, Any]]) -> None:
    _append_csv_rows(
        "swe_model_calls.csv",
        [
            "instance_id",
            "call_index",
            "request_id",
            "client_request_id",
            "start_ts",
            "end_ts",
            "client_duration_s",
            "status",
            "turn_index",
            "step_index",
            "prompt_tokens",
            "completion_tokens",
        ],
        [
            [
                record["instance_id"],
                record["call_index"],
                record["request_id"],
                record["client_request_id"],
                f"{record['start_ts']:.6f}",
                f"{record['end_ts']:.6f}",
                f"{record['client_duration_s']:.6f}",
                record["status"],
                record.get("turn_index", ""),
                record.get("step_index", ""),
                record.get("prompt_tokens", ""),
                record.get("completion_tokens", ""),
            ]
            for record in records
        ],
    )


def record_task_completion(
    *,
    instance_id: str,
    start_ts: float,
    end_ts: float,
    duration_s: float | None = None,
    exit_status: str | None,
    api_calls: int,
    model_query_s: float = 0.0,
    tool_execution_s: float = 0.0,
    tool_calls: int = 0,
) -> None:
    duration_s = end_ts - start_ts if duration_s is None else duration_s
    frontend_other_s = duration_s - model_query_s - tool_execution_s
    _append_csv(
        "swe_task_completion.csv",
        [
            "instance_id",
            "start_ts",
            "end_ts",
            "duration_s",
            "exit_status",
            "api_calls",
            "tool_calls",
            "model_query_s",
            "tool_execution_s",
            "frontend_other_s",
        ],
        [
            instance_id,
            f"{start_ts:.6f}",
            f"{end_ts:.6f}",
            f"{duration_s:.6f}",
            exit_status or "",
            api_calls,
            tool_calls,
            f"{model_query_s:.6f}",
            f"{tool_execution_s:.6f}",
            f"{frontend_other_s:.6f}",
        ],
    )
