"""Deterministic, planning-only HPO review and bounded repair."""

from __future__ import annotations

from typing import Any


def _finding(field: str, severity: str, reason: str, recommended: Any = None, rule_id: str | None = None):
    return {
        "field": field,
        "severity": severity,
        "reason": reason,
        "recommended_value": None if recommended is None else str(recommended),
        "rule_id": rule_id,
    }


def evaluate_hpo(
    config: dict[str, Any], context: dict[str, Any], recipe: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    findings = []
    selected_id = (context.get("selected_model_info") or {}).get("id")
    if config.get("model_name") != selected_id:
        findings.append(_finding("model_name", "hard_error", "Configuration changed the selected model.", selected_id, "selected_model_identity"))
    selected_data = config.get("selected_data") or []
    for class_name in context.get("classes") or []:
        assignment = next((item for item in selected_data if item.get("class_name") == class_name), None)
        splits = {allocation.get("split") for source in (assignment or {}).get("sources", []) for allocation in source.get("allocations", [])}
        missing = {"train", "validation", "test"} - splits
        if missing:
            findings.append(_finding("selected_data", "hard_error", f"Class {class_name!r} is missing splits: {sorted(missing)}.", rule_id="split_coverage"))
    numeric = {
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "num_epochs": "epochs",
        "image_size": "image_size",
        "input_size": "image_size",
    }
    if recipe:
        for field, recipe_field in numeric.items():
            if field not in config:
                continue
            value = float(config[field])
            minimum = recipe.get(f"{recipe_field}_min")
            maximum = recipe.get(f"{recipe_field}_max")
            if minimum not in (None, "") and value < float(minimum):
                findings.append(_finding(field, "hard_error", f"Value {value:g} is below recipe minimum {float(minimum):g}.", minimum, recipe.get("id")))
            if maximum not in (None, "") and value > float(maximum):
                findings.append(_finding(field, "hard_error", f"Value {value:g} is above recipe maximum {float(maximum):g}.", maximum, recipe.get("id")))
    epochs = int(config.get("num_epochs") or 0)
    patience = int(config.get("patience") or 0)
    if patience and epochs and patience >= epochs:
        findings.append(_finding("patience", "hard_error", "Patience must be lower than the epoch limit.", max(0, epochs // 5), "early_stopping"))
    hardware = context.get("training_hardware") or {}
    max_batch = hardware.get("max_batch_size")
    if max_batch and int(config.get("batch_size") or 0) > int(max_batch):
        findings.append(_finding("batch_size", "hard_error", "Batch size exceeds the declared planning hardware limit.", max_batch, "hardware_batch_limit"))
    if context.get("task") == "classification" and config.get("criterion_name") != "cross_entropy":
        findings.append(_finding("criterion_name", "hard_error", "The planned single-label classification data requires cross entropy.", "cross_entropy", "classification_target_contract"))
    if context.get("task") == "detection" and "small" in (context.get("robustness_requirements") or {}).get("object_scale", []):
        if int(config.get("input_size") or 0) < 640:
            findings.append(_finding("input_size", "safety_warning", "Small-object requirements may be underserved below 640px.", 640, "small_object_resolution"))
    return findings


def repair_hpo(config: dict[str, Any], findings: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    repaired = dict(config)
    changed = []
    for finding in findings:
        if finding["severity"] != "hard_error" or finding.get("recommended_value") is None:
            continue
        field = finding["field"]
        current = repaired.get(field)
        recommendation = finding["recommended_value"]
        if isinstance(current, bool):
            value = recommendation.lower() == "true"
        elif isinstance(current, int):
            value = int(float(recommendation))
        elif isinstance(current, float):
            value = float(recommendation)
        else:
            value = recommendation
        if current != value:
            repaired[field] = value
            changed.append(field)
    return repaired, changed


def decision_from_findings(findings: list[dict[str, Any]], repaired_fields: list[str]):
    blocking = [item for item in findings if item["severity"] == "hard_error"]
    warnings = [item for item in findings if item["severity"] != "hard_error"]
    return {
        "accept": not blocking,
        "reason": (
            "The proposal passed schema, dataset, hardware, and ontology-policy validation."
            if not blocking else "The proposal still contains blocking planning conflicts after one bounded repair round."
        ),
        "findings": findings,
        "suggestions": [item["reason"] for item in warnings],
        "repair": {"attempted": bool(repaired_fields), "changed_fields": repaired_fields, "rounds": 1 if repaired_fields else 0},
    }
