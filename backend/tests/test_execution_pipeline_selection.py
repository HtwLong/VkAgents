import json

from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline
from cvmodellearning import paths
from routers.execution import get_pipeline_by_task


def _write_interpretation(tmp_path, monkeypatch, filename, task):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    planning_dir = paths.planning_artifacts_dir("job")
    (planning_dir / filename).write_text(json.dumps({"task": task}), encoding="utf-8")


def test_pipeline_uses_current_interpretation_checkpoint(tmp_path, monkeypatch):
    _write_interpretation(
        tmp_path, monkeypatch, "STATE_01_INTERPRETATION.json", "detection"
    )

    assert isinstance(get_pipeline_by_task("job"), DetectionPipeline)


def test_pipeline_supports_legacy_interpretation_result(tmp_path, monkeypatch):
    _write_interpretation(
        tmp_path, monkeypatch, "RESULT_INTERPRETATION.json", "classification"
    )

    assert isinstance(get_pipeline_by_task("job"), ClassificationPipeline)
