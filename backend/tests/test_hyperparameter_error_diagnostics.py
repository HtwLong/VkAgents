import pytest
from pydantic import ValidationError

from cvmodellearning.agents.hyperparameter_agents import (
    HPO_REVIEW_HISTORY_LIMIT,
    append_rejected_review_record,
    build_evaluator_messages,
    build_validation_error_summary,
    demote_external_deployment_findings,
    discard_inactive_schema_findings,
    discard_equivalent_model_findings,
    evaluator_active_configuration,
    format_pydantic_validation_errors,
    normalize_hpo_decision,
    rejected_review_record,
    training_hardware_role_findings,
)
from cvmodellearning.graphrag.hyperparameter_context import (
    _normalize_inactive_classification_fields,
)
from cvmodellearning.schemas.classification_hpo import ClassificationConfigModel
from cvmodellearning.schemas.detection_hpo import DetectionConfigModel
from cvmodellearning.schemas.decision_schema import HpoDecision, HpoFinding


def test_reviewer_history_is_bounded_to_latest_three_rejections():
    history = []
    for round_idx in range(1, 6):
        append_rejected_review_record(history, {"round": round_idx})
    assert HPO_REVIEW_HISTORY_LIMIT == 3
    assert [item["round"] for item in history] == [3, 4, 5]


def test_rejected_review_record_tracks_config_findings_and_changes():
    decision = HpoDecision(
        accept=False,
        reason="Learning rate is unsafe.",
        findings=[
            HpoFinding(field="learning_rate", severity="safety_warning",
                       reason="Too high for stable fine-tuning.", recommended_value="0.001"),
            HpoFinding(field="batch_size", severity="preference",
                       reason="A larger batch may be faster."),
        ],
    )
    record = rejected_review_record(
        round_idx=2,
        candidate={"learning_rate": 0.005, "batch_size": 8},
        decision=decision,
        previous_candidate={"learning_rate": 0.01, "batch_size": 8},
    )
    assert len(record["candidate_fingerprint"]) == 16
    assert record["active_configuration"]["learning_rate"] == 0.005
    assert record["changes_from_previous_rejected_candidate"] == {
        "learning_rate": {"from": 0.01, "to": 0.005}
    }
    assert [finding["field"] for finding in record["blocking_findings"]] == ["learning_rate"]


def test_evaluator_messages_include_prior_rejections_and_current_candidate():
    messages = build_evaluator_messages(
        system_prompt="review safely",
        evaluator_context_json='{"task": "classification"}',
        runtime_guidance="runtime facts",
        candidate={"learning_rate": 0.001},
        rejected_review_history=[{
            "round": 1,
            "candidate_fingerprint": "abc",
            "active_configuration": {"learning_rate": 0.01},
            "blocking_findings": [{"field": "learning_rate"}],
        }],
    )
    assert messages[0] == {"role": "system", "content": "review safely"}
    review_request = messages[1]["content"]
    assert "Previous Rejected Review Rounds" in review_request
    assert '"candidate_fingerprint": "abc"' in review_request
    assert '"learning_rate": 0.001' in review_request
    assert "explicitly explain any reversal" in review_request


def test_review_fingerprint_ignores_rationale_only_changes():
    decision = HpoDecision(accept=False, reason="Unsafe.", findings=[])
    first = rejected_review_record(
        round_idx=1,
        candidate={"batch_size": 32, "rationale": "Initial rationale."},
        decision=decision,
        previous_candidate=None,
    )
    second = rejected_review_record(
        round_idx=2,
        candidate={"batch_size": 32, "rationale": "Reworded rationale."},
        decision=decision,
        previous_candidate={"batch_size": 32, "rationale": "Initial rationale."},
    )
    assert first["candidate_fingerprint"] == second["candidate_fingerprint"]
    assert second["changes_from_previous_rejected_candidate"] == {}


def test_validation_diagnostics_include_field_input_and_error_type():
    with pytest.raises(ValidationError) as exc_info:
        ClassificationConfigModel.model_validate({"alpha": 0})

    diagnostics = format_pydantic_validation_errors(exc_info.value)
    alpha_error = next(item for item in diagnostics if item["field"] == "alpha")

    assert alpha_error == {
        "field": "alpha",
        "message": "Input should be greater than 0",
        "error_type": "greater_than",
        "input": 0,
    }


def test_validation_error_summary_is_concise_and_readable():
    with pytest.raises(ValidationError) as exc_info:
        ClassificationConfigModel.model_validate({"alpha": 0})

    diagnostics = format_pydantic_validation_errors(exc_info.value)
    summary = build_validation_error_summary(diagnostics)

    assert "alpha: Input should be greater than 0" in summary
    assert "input=0" in summary


def test_inactive_step_and_sgd_fields_are_normalized_to_schema_safe_sentinels():
    config = {
        "scheduler_name": "step",
        "min_learning_rate": 0.0001,
        "optimizer_name": "sgd",
        "eps": 0.0,
        "alpha": 0.0,
    }
    provenance = {}

    _normalize_inactive_classification_fields(config, provenance)

    assert config["min_learning_rate"] == 0.0
    assert config["eps"] == 1e-8
    assert config["alpha"] == 0.99
    assert "beta1" not in config
    assert "beta2" not in config
    assert provenance["min_learning_rate"]["source"] == "system_policy"
    assert provenance["eps"]["source_id"] == "inactive_for_sgd_optimizer"


def test_actionable_findings_override_inconsistent_accept_true():
    warning = HpoFinding(
        field="eps",
        severity="safety_warning",
        reason="Inactive field is present.",
    )
    decision = HpoDecision(
        accept=True,
        reason="Configuration is safe.",
        findings=[warning],
    )

    normalized, resolution = normalize_hpo_decision(decision, [warning])

    assert normalized.accept is False
    assert normalized.findings[0].severity == "safety_warning"
    assert resolution == "accept_set_false"


def test_preferences_override_inconsistent_reject_false():
    preference = HpoFinding(
        field="batch_size",
        severity="preference",
        reason="A larger batch could be tried.",
    )
    decision = HpoDecision(
        accept=False,
        reason="Optional improvement available.",
        findings=[preference],
    )

    normalized, resolution = normalize_hpo_decision(decision, [])

    assert normalized.accept is True
    assert normalized.reason == (
        "No hard errors or safety warnings; preference findings are advisory."
    )
    assert resolution == "accept_set_true"


@pytest.mark.parametrize(
    "field",
    (
        "memory_category",
        "max_runtime_memory_mb",
        "max_model_size_mb",
        "max_parameters_m",
        "max_cpu_latency_ms",
    ),
)
def test_planning_owned_deployment_findings_are_advisory(field):
    warning = HpoFinding(
        field=field,
        severity="safety_warning",
        reason="Deployment constraint requires verification.",
    )
    decision = HpoDecision(
        accept=False,
        reason="Latency is unverified.",
        findings=[warning],
    )

    demoted, fields = demote_external_deployment_findings(decision)
    blocking = [
        finding for finding in demoted.findings
        if finding.severity in {"hard_error", "safety_warning"}
    ]
    normalized, resolution = normalize_hpo_decision(demoted, blocking)

    assert fields == [field]
    assert demoted.findings[0].severity == "preference"
    assert normalized.accept is True
    assert resolution == "accept_set_true"


def test_soft_memory_finding_is_advisory_even_if_evaluator_calls_it_hard():
    warning = HpoFinding(
        field="max_runtime_memory_mb",
        severity="hard_error",
        reason="A conservative provisioning estimate exceeds the soft preference.",
    )
    decision = HpoDecision(accept=False, reason="Memory concern.", findings=[warning])

    normalized, fields = demote_external_deployment_findings(decision)

    assert fields == ["max_runtime_memory_mb"]
    assert normalized.findings[0].severity == "preference"


def test_hyphenated_missing_cpu_latency_finding_is_advisory():
    warning = HpoFinding(
        field="max_cpu_latency_ms",
        severity="safety_warning",
        reason="No CPU-latency benchmark exists for the exact target.",
    )
    decision = HpoDecision(accept=False, reason="Latency unverified.", findings=[warning])

    normalized, fields = demote_external_deployment_findings(decision)

    assert fields == ["max_cpu_latency_ms"]
    assert normalized.findings[0].severity == "preference"


def test_real_hpo_safety_finding_remains_blocking():
    warning = HpoFinding(
        field="batch_size",
        severity="safety_warning",
        reason="The batch does not fit training memory.",
    )
    decision = HpoDecision(accept=False, reason="Training OOM risk.", findings=[warning])

    normalized, fields = demote_external_deployment_findings(decision)

    assert fields == []
    assert normalized.findings == [warning]


def test_unrelated_unknown_safety_finding_remains_blocking():
    warning = HpoFinding(
        field="deployment_runtime",
        severity="hard_error",
        reason="The selected runtime is incompatible.",
    )
    decision = HpoDecision(accept=False, reason="Incompatible runtime.", findings=[warning])

    normalized, fields = demote_external_deployment_findings(decision)

    assert fields == []
    assert normalized.findings == [warning]


def test_detection_evaluator_sees_only_active_optimizer_fields():
    candidate = DetectionConfigModel.model_validate({
        "task_type": "detection",
        "classes": ["furniture"],
        "selected_data": [
            {"class_name": "furniture", "sources": [{"dataset_name": "demo", "count": 20}]}
        ],
        "num_epochs": 10,
        "patience": 2,
        "model_name": "yolov10_n",
        "optimizer_name": "auto",
        "amp": False,
        "rationale": "CPU-compatible YOLO configuration.",
    })

    active = evaluator_active_configuration(candidate, "detection")

    assert active["optimizer_name"] == "auto"
    assert active["amp"] is False
    assert "beta1" not in active
    assert "momentum" not in active
    assert "precision" not in active


def test_batch_rationale_cannot_use_separate_deployment_gpu_as_training_limit():
    candidate = DetectionConfigModel.model_validate({
        "task_type": "detection",
        "classes": ["traffic light"],
        "selected_data": [{
            "class_name": "traffic light",
            "sources": [{"dataset_name": "demo", "count": 20}],
        }],
        "model_name": "yolov12_x",
        "num_epochs": 100,
        "patience": 20,
        "batch_size": 4,
        "rationale": "Reduced batch size to fit the RTX 2060 used for deployment.",
        "llm_field_rationales": [{
            "field": "batch_size",
            "reason": "Batch 4 fits the 6 GB RTX 2060 VRAM limit.",
        }],
    })
    context = {
        "available_hardware": {"gpu_type": "NVIDIA GeForce RTX 2060", "vram_gb": 6},
        "training_hardware": {"gpu_type": "NVIDIA RTX 6000 Ada", "vram_gb": 48},
    }

    findings = training_hardware_role_findings(candidate, context)

    assert findings == [{
        "field": "batch_size",
        "severity": "safety_warning",
        "reason": (
            "The batch-size rationale cites deployment GPU 'NVIDIA GeForce RTX 2060', "
            "but training runs on 'NVIDIA RTX 6000 Ada'. Re-evaluate batch size using "
            "training_hardware and its hardware-safe candidates only."
        ),
        "rule_id": "hpo.training_hardware_role.v1",
    }]


def test_inactive_and_cross_task_evaluator_findings_are_discarded():
    decision = HpoDecision(
        accept=False,
        reason="Invalid precision and beta1.",
        findings=[
            HpoFinding(
                field="precision",
                severity="hard_error",
                reason="Incorrectly inferred from recipe metadata.",
            ),
            HpoFinding(
                field="beta1",
                severity="safety_warning",
                reason="Inactive for auto optimizer.",
            ),
            HpoFinding(
                field="batch_size",
                severity="hard_error",
                reason="Concrete active-field issue.",
            ),
            HpoFinding(
                field="unknown_runtime_switch",
                severity="hard_error",
                reason="Unknown fields must continue to fail closed.",
            ),
        ],
    )

    normalized, discarded = discard_inactive_schema_findings(
        decision,
        {"batch_size", "amp", "optimizer_name"},
    )

    assert discarded == ["beta1", "precision"]
    assert [finding.field for finding in normalized.findings] == [
        "batch_size",
        "unknown_runtime_switch",
    ]


def test_equivalent_model_identifier_finding_is_discarded():
    mismatch = HpoFinding(
        field="model_name",
        severity="hard_error",
        reason="yolov8_n does not match yolov8n.",
        recommended_value="yolov8n",
    )
    decision = HpoDecision(
        accept=False,
        reason="Model mismatch.",
        findings=[mismatch],
        suggestions=["Change model_name to yolov8n."],
    )

    normalized, discarded = discard_equivalent_model_findings(
        decision, "yolov8_n", "yolov8n"
    )

    assert discarded is True
    assert normalized.accept is True
    assert normalized.findings == []
    assert normalized.suggestions is None
