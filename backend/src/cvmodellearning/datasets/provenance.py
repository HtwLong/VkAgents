"""Audit the prepared split artifacts consumed by training and evaluation."""

import json
from pathlib import Path
from typing import Any, Mapping

from cvmodellearning.paths import data_provenance_path, dataset_manifest_path
from cvmodellearning.download.assignment_manifest import file_sha256


_STAGE_SPLITS = {
    "training": {"train", "validation"},
    "evaluation": {"test"},
}


def record_split_access(
    job_id: str,
    *,
    task: str,
    stage: str,
    preparation: Mapping[str, Any],
    split_artifacts: Mapping[str, Path],
) -> Path:
    """Persist which immutable prepared splits a runtime stage is allowed to read."""

    expected_splits = _STAGE_SPLITS.get(stage)
    if expected_splits is None:
        raise ValueError(f"Unsupported provenance stage: {stage!r}.")
    if set(split_artifacts) != expected_splits:
        raise ValueError(
            f"{stage.capitalize()} must consume exactly: {', '.join(sorted(expected_splits))}."
        )

    fingerprint = str(preparation.get("assignment_fingerprint") or "")
    counts = preparation.get("counts") or {}
    if not fingerprint:
        raise ValueError("Preparation summary has no assignment fingerprint.")

    manifest_path = dataset_manifest_path(job_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = file_sha256(manifest_path)
    if manifest.get("assignment_fingerprint") != fingerprint:
        raise ValueError("Dataset manifest and preparation summary fingerprints differ.")
    if preparation.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Preparation summary does not match the current dataset manifest.")
    official_counts = {split: 0 for split in ("train", "validation", "test")}
    derived_counts = {split: 0 for split in ("train", "validation", "test")}
    for sample in manifest.get("samples") or []:
        split = sample.get("assigned_split")
        if split not in official_counts:
            raise ValueError("Dataset manifest contains an unsupported split.")
        target = (
            derived_counts
            if sample.get("assignment_type") == "derived_from_train"
            else official_counts
        )
        target[split] += 1

    stage_record = {
        "splits": {
            split: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "sample_count": counts[split],
            }
            for split, path in sorted(split_artifacts.items())
        }
    }
    output_path = data_provenance_path(job_id)
    audit: dict[str, Any] = {}
    if output_path.exists():
        audit = json.loads(output_path.read_text(encoding="utf-8"))
    dataset_changed = (
        audit
        and audit.get("assignment_fingerprint") == fingerprint
        and audit.get("manifest_sha256") != manifest_sha256
    )
    if dataset_changed and stage == "evaluation" and "training" in audit.get("stages", {}):
        raise ValueError("Prepared dataset changed after training; evaluation cannot continue.")
    if audit.get("assignment_fingerprint") != fingerprint or dataset_changed:
        audit = {
            "version": 1,
            "job_id": job_id,
            "task": task,
            "assignment_fingerprint": fingerprint,
            "manifest_sha256": manifest_sha256,
            "official_counts": official_counts,
            "derived_counts": derived_counts,
            "stages": {},
        }
    elif audit.get("task") != task:
        raise ValueError("Existing data provenance audit belongs to a different task.")
    audit["stages"][stage] = stage_record
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return output_path
