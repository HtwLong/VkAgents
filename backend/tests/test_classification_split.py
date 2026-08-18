import json

import pandas as pd
import pytest

from cvmodellearning.download.assignment_manifest import assignment_fingerprint
from cvmodellearning.pipelines import classification_pipe


def _assignments(labels=("cat", "dog")):
    return [
        {
            "class_name": label,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [
                    {"split": "train", "count": 1, "assignment_type": "official_split"},
                    {"split": "validation", "count": 1, "assignment_type": "derived_from_train"},
                    {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
                ],
            }],
        }
        for label in labels
    ]


def _write_download_artifacts(tmp_path, selected_data):
    samples = []
    rows = []
    for label in (item["class_name"] for item in selected_data):
        for split in ("train", "validation", "test"):
            filename = f"{label}_{split}.jpg"
            (tmp_path / filename).write_bytes(f"image:{filename}".encode("utf-8"))
            rows.append({"image_filename": filename, "labels": label})
            samples.append({
                "sample_id": filename,
                "image_path": filename,
                "class_names": [label],
                "dataset_name": "bdd_100k_det_train",
                "source_role": "train",
                "assigned_split": split,
                "assignment_type": "official_split" if split == "train" else "derived_from_train",
            })
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 1,
        "job_id": "test-job",
        "task": "classification",
        "assignment_fingerprint": assignment_fingerprint(selected_data),
        "samples": samples,
    }))


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(classification_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(classification_pipe, "csv_labels_path", lambda _job_id: tmp_path / "labels.csv")
    monkeypatch.setattr(classification_pipe, "dataset_manifest_path", lambda _job_id: tmp_path / "manifest.json")
    monkeypatch.setattr(classification_pipe, "preparation_summary_path", lambda _job_id: tmp_path / "summary.json")
    monkeypatch.setattr(classification_pipe, "train_csv_path", lambda _job_id: tmp_path / "train.csv")
    monkeypatch.setattr(classification_pipe, "val_csv_path", lambda _job_id: tmp_path / "val.csv")
    monkeypatch.setattr(classification_pipe, "test_csv_path", lambda _job_id: tmp_path / "test.csv")


def test_prepare_data_preserves_manifest_assignments(tmp_path, monkeypatch):
    selected_data = _assignments()
    _write_download_artifacts(tmp_path, selected_data)
    _patch_paths(monkeypatch, tmp_path)

    result = classification_pipe.ClassificationPipeline().prepare_data_step(
        {"selected_data": selected_data, "classes": ["cat", "dog"]},
        "test-job",
    )

    assert result["counts"] == {"train": 2, "validation": 2, "test": 2}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(tmp_path / ("val.csv" if split == "validation" else f"{split}.csv"))
        assert set(frame["image_filename"]) == {f"cat_{split}.jpg", f"dog_{split}.jpg"}
    assert json.loads((tmp_path / "summary.json").read_text())["assignment_fingerprint"] == assignment_fingerprint(selected_data)
    classification_pipe.ClassificationPipeline()._require_prepared_data(
        {"selected_data": selected_data},
        "test-job",
    )

    (tmp_path / "test.csv").unlink()
    with pytest.raises(FileNotFoundError, match="test"):
        classification_pipe.ClassificationPipeline()._require_prepared_data(
            {"selected_data": selected_data},
            "test-job",
        )


def test_prepare_data_preserves_numeric_class_names_as_strings(tmp_path, monkeypatch):
    selected_data = _assignments(("0", "1"))
    _write_download_artifacts(tmp_path, selected_data)
    _patch_paths(monkeypatch, tmp_path)

    classification_pipe.ClassificationPipeline().prepare_data_step(
        {"selected_data": selected_data, "classes": ["0", "1"]},
        "test-job",
    )

    assert pd.read_csv(tmp_path / "train.csv", dtype={"labels": str})["labels"].tolist() == ["0", "1"]
    dataset = classification_pipe.CocoImageDataset(
        tmp_path / "train.csv",
        tmp_path,
        class_to_idx={"0": 0, "1": 1},
    )
    assert dataset.data["label_enc"].tolist() == [0, 1]


def test_prepare_data_rejects_stale_manifest(tmp_path, monkeypatch):
    selected_data = _assignments()
    _write_download_artifacts(tmp_path, selected_data)
    _patch_paths(monkeypatch, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["assignment_fingerprint"] = "stale"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="current selected_data"):
        classification_pipe.ClassificationPipeline().prepare_data_step(
            {"selected_data": selected_data, "classes": ["cat", "dog"]},
            "test-job",
        )


def test_readiness_rejects_new_manifest_for_same_assignment_plan(tmp_path, monkeypatch):
    selected_data = _assignments()
    _write_download_artifacts(tmp_path, selected_data)
    _patch_paths(monkeypatch, tmp_path)
    pipeline = classification_pipe.ClassificationPipeline()
    config = {"selected_data": selected_data, "classes": ["cat", "dog"]}
    pipeline.prepare_data_step(config, "test-job")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["samples"][0]["sample_id"] = "replacement-sample"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="current dataset manifest"):
        pipeline._require_prepared_data(config, "test-job")
