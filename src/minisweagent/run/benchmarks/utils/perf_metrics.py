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


def _append_csv(filename: str, header: list[str], row: list[Any]) -> None:
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
            writer.writerow(row)


def record_task_completion(
    *,
    instance_id: str,
    start_ts: float,
    end_ts: float,
    exit_status: str | None,
    api_calls: int,
) -> None:
    _append_csv(
        "swe_task_completion.csv",
        ["instance_id", "start_ts", "end_ts", "duration_s", "exit_status", "api_calls"],
        [
            instance_id,
            f"{start_ts:.6f}",
            f"{end_ts:.6f}",
            f"{end_ts - start_ts:.6f}",
            exit_status or "",
            api_calls,
        ],
    )


def record_tool_iteration_ttft(
    *,
    instance_id: str,
    step: int,
    tool_end_ts: float,
    first_token_ts: float,
) -> None:
    _append_csv(
        "tool_iteration_ttft.csv",
        ["instance_id", "step", "tool_end_ts", "first_token_ts", "ttft_s"],
        [
            instance_id,
            step,
            f"{tool_end_ts:.6f}",
            f"{first_token_ts:.6f}",
            f"{first_token_ts - tool_end_ts:.6f}",
        ],
    )
