import concurrent.futures
import csv

import pytest

from minisweagent.run.benchmarks.utils import perf_metrics


def _model_call_record(index: int) -> dict:
    return {
        "instance_id": f"task-{index}",
        "call_index": 1,
        "request_id": f"chatcmpl-task-{index}-1",
        "client_request_id": f"task-{index}-1",
        "start_ts": 10.0,
        "end_ts": 11.0,
        "client_duration_s": 1.0,
        "status": "ok",
    }


def test_record_model_calls_is_concurrency_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_PERF_METRICS_DIR", str(tmp_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: perf_metrics.record_model_calls([_model_call_record(index)]), range(12)))

    with (tmp_path / "swe_model_calls.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12
    assert {row["instance_id"] for row in rows} == {f"task-{index}" for index in range(12)}
    assert all(float(row["client_duration_s"]) == 1.0 for row in rows)


def test_record_task_completion_writes_frontend_timing_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MSWEA_PERF_METRICS_DIR", str(tmp_path))

    perf_metrics.record_task_completion(
        instance_id="task-1",
        start_ts=100.0,
        end_ts=200.0,
        duration_s=10.0,
        exit_status="Submitted",
        api_calls=2,
        model_query_s=4.0,
        tool_execution_s=3.0,
        tool_calls=2,
    )

    with (tmp_path / "swe_task_completion.csv").open(newline="") as f:
        row = next(csv.DictReader(f))

    assert float(row["duration_s"]) == 10.0
    assert row["api_calls"] == "2"
    assert row["tool_calls"] == "2"
    assert float(row["model_query_s"]) == 4.0
    assert float(row["tool_execution_s"]) == 3.0
    assert float(row["frontend_other_s"]) == pytest.approx(3.0)
