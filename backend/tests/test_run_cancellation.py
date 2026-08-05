import json

import pytest

from cvmodellearning.jobs import run_control
from routers import runs


@pytest.fixture(autouse=True)
def clear_cancel_flag():
    run_control.clear_cancellation("job")
    yield
    run_control.clear_cancellation("job")


def _patch_run_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(run_control, "run_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(runs, "RUNS_ROOT", tmp_path)


def test_run_state_is_atomic_and_cancellation_is_durable(monkeypatch, tmp_path):
    run = tmp_path / "job"
    run.mkdir()
    _patch_run_roots(monkeypatch, tmp_path)

    running = run_control.write_run_state("job", "running", active_step="download-data")
    cancelling = run_control.request_cancellation("job")

    assert running["attempt"] == 1
    assert cancelling["status"] == "cancelling"
    assert cancelling["active_step"] == "download-data"
    assert run_control.cancellation_requested("job") is True
    assert not (run / "run_state.json.tmp").exists()
    assert json.loads((run / "run_state.json").read_text())["status"] == "cancelling"


def test_cancel_endpoint_is_idempotent(monkeypatch, tmp_path):
    run = tmp_path / "job"
    run.mkdir()
    _patch_run_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(runs, "_step_running", lambda _job_id, _step_id: True)
    run_control.write_run_state("job", "running", active_step="train-model")

    first = runs.cancel_run("job")
    second = runs.cancel_run("job")

    assert first["status"] == "cancelling"
    assert second["status"] == "cancelling"
    assert second["active_step"] == "train-model"


def test_repeated_cancel_stops_stale_download(monkeypatch, tmp_path):
    run = tmp_path / "job"
    run.mkdir()
    _patch_run_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(runs, "_step_running", lambda _job_id, _step_id: False)
    run_control.write_run_state("job", "running", active_step="download-data")
    run_control.request_cancellation("job")

    state = runs.cancel_run("job")

    assert state["status"] == "stopped"
    assert state["active_step"] == "download-data"


def test_cancel_idle_run_stops_immediately(monkeypatch, tmp_path):
    run = tmp_path / "job"
    run.mkdir()
    _patch_run_roots(monkeypatch, tmp_path)
    run_control.write_run_state("job", "waiting", active_step=None)

    state = runs.cancel_run("job")

    assert state["status"] == "stopped"


def test_cancelled_exception_is_not_a_failure():
    assert issubclass(run_control.PipelineCancelled, Exception)
    with pytest.raises(run_control.PipelineCancelled):
        raise run_control.PipelineCancelled("stopped")


def test_cancellation_wins_step_completion_race(monkeypatch, tmp_path):
    run = tmp_path / "job"
    run.mkdir()
    _patch_run_roots(monkeypatch, tmp_path)
    run_control.write_run_state("job", "running", active_step="prepare-data")
    run_control.request_cancellation("job")

    with pytest.raises(run_control.PipelineCancelled):
        run_control.finish_or_stop("job", "prepare-data")

    assert run_control.read_run_state("job")["status"] == "stopped"
