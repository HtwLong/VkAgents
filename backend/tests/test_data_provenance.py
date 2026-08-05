import hashlib
import json

import pytest

from cvmodellearning.datasets import provenance
from cvmodellearning.pipelines import classification_pipe


def _setup(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    audit_path = tmp_path / "audit.json"
    manifest_path.write_text(json.dumps({
        "assignment_fingerprint": "fingerprint",
        "samples": [
            {"assigned_split": "train", "assignment_type": "official_split"},
            {"assigned_split": "validation", "assignment_type": "official_split"},
            {"assigned_split": "validation", "assignment_type": "derived_from_train"},
            {"assigned_split": "test", "assignment_type": "derived_from_train"},
        ],
    }))
    monkeypatch.setattr(provenance, "dataset_manifest_path", lambda _job_id: manifest_path)
    monkeypatch.setattr(provenance, "data_provenance_path", lambda _job_id: audit_path)
    return audit_path


def test_provenance_records_disjoint_training_and_evaluation_inputs(monkeypatch, tmp_path):
    audit_path = _setup(monkeypatch, tmp_path)
    artifacts = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.data"
        path.write_text(split)
        artifacts[split] = path
    preparation = {
        "assignment_fingerprint": "fingerprint",
        "manifest_sha256": provenance.file_sha256(provenance.dataset_manifest_path("job")),
        "counts": {"train": 1, "validation": 2, "test": 1},
    }

    provenance.record_split_access(
        "job",
        task="classification",
        stage="training",
        preparation=preparation,
        split_artifacts={key: artifacts[key] for key in ("train", "validation")},
    )
    provenance.record_split_access(
        "job",
        task="classification",
        stage="evaluation",
        preparation=preparation,
        split_artifacts={"test": artifacts["test"]},
    )

    audit = json.loads(audit_path.read_text())
    assert set(audit["stages"]["training"]["splits"]) == {"train", "validation"}
    assert set(audit["stages"]["evaluation"]["splits"]) == {"test"}
    assert audit["official_counts"]["validation"] == 1
    assert audit["derived_counts"]["validation"] == 1
    assert audit["stages"]["evaluation"]["splits"]["test"]["sha256"] == hashlib.sha256(b"test").hexdigest()


def test_provenance_rejects_test_access_during_training(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    test_path = tmp_path / "test.data"
    test_path.write_text("test")

    with pytest.raises(ValueError, match="Training must consume exactly"):
        provenance.record_split_access(
            "job",
            task="detection",
            stage="training",
            preparation={
                "assignment_fingerprint": "fingerprint",
                "manifest_sha256": provenance.file_sha256(provenance.dataset_manifest_path("job")),
                "counts": {"train": 1, "validation": 1, "test": 1},
            },
            split_artifacts={"test": test_path},
        )


def test_provenance_rejects_evaluation_if_dataset_changed_after_training(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    artifacts = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.data"
        path.write_text(split)
        artifacts[split] = path
    preparation = {
        "assignment_fingerprint": "fingerprint",
        "manifest_sha256": provenance.file_sha256(provenance.dataset_manifest_path("job")),
        "counts": {"train": 1, "validation": 2, "test": 1},
    }
    provenance.record_split_access(
        "job",
        task="classification",
        stage="training",
        preparation=preparation,
        split_artifacts={key: artifacts[key] for key in ("train", "validation")},
    )
    manifest_path = provenance.dataset_manifest_path("job")
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][0]["sample_id"] = "replacement"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match the current dataset manifest"):
        provenance.record_split_access(
            "job",
            task="classification",
            stage="evaluation",
            preparation=preparation,
            split_artifacts={"test": artifacts["test"]},
        )


def test_failed_evaluation_does_not_claim_test_access(monkeypatch, tmp_path):
    monkeypatch.setattr(
        classification_pipe.ClassificationPipeline,
        "_require_prepared_data",
        lambda *_args: {"assignment_fingerprint": "fingerprint"},
    )
    monkeypatch.setattr(
        classification_pipe,
        "best_model_path",
        lambda _job_id: tmp_path / "missing-model.pth",
    )
    monkeypatch.setattr(
        classification_pipe,
        "record_split_access",
        lambda *_args, **_kwargs: pytest.fail("failed evaluation recorded provenance"),
    )

    with pytest.raises(FileNotFoundError, match="Best model not found"):
        classification_pipe.ClassificationPipeline().evaluate_model_step({}, "job")
