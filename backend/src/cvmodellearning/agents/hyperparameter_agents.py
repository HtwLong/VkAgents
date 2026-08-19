import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from typing import TypeVar, Union
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

# Import your schemas and logging utility
from cvmodellearning.schemas.classification_hpo import (
    ClassificationConfigDraft,
    ClassificationConfigModel,
    active_classification_config_fields,
)
from cvmodellearning.schemas.detection_hpo import (
    DetectionConfigDraft,
    active_detection_config_fields,
)
from cvmodellearning.schemas.detection_hpo_completion import complete_detection_config
from cvmodellearning.schemas.vqa_hpo import VQAConfigModel
from cvmodellearning.schemas.decision_schema import HpoDecision, HpoFinding
from cvmodellearning.agents.agents_utils import log_planning_step
from cvmodellearning.graphrag.hyperparameter_context import (
    llm_controlled_fields,
    validate_executable_recipe_config,
)
from cvmodellearning.models.classification_capabilities import (
    classification_prompt_constraints,
    selected_classification_model_id,
)
from cvmodellearning.models.detection_capabilities import detection_prompt_constraints
from cvmodellearning.models.registry import (
    DETECTION_HPO_MODEL_IDS,
    enabled_models,
    model_ids_equivalent,
)
from cvmodellearning.schemas.classification_hpo_completion import complete_classification_config
from cvmodellearning.schemas.dataset_assignment import planned_split_ratios
from cvmodellearning.schemas.revision import hpo_override_values
from cvmodellearning.skills import load_cv_skill
from cvmodellearning.training.resource_guard import validate_training_resource_config
from cvmodellearning.llm_config import PLANNING_MODEL
from cvmodellearning.observability.planning_usage import run_planning_completion


class HpoPhaseTimeout(TimeoutError):
    """Identifies the model phase that exceeded its bounded request timeout."""

    def __init__(self, phase: str, round_idx: int, timeout_seconds: int, attempts: int):
        self.phase = phase
        self.round_idx = round_idx
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        super().__init__(
            f"{phase} timed out after {attempts} attempts of {timeout_seconds} seconds "
            f"in round {round_idx}"
        )


_HpoCallResult = TypeVar("_HpoCallResult")


async def _hpo_model_call(
    awaitable_factory: Callable[[], Awaitable[_HpoCallResult]],
    *,
    phase: str,
    round_idx: int,
) -> _HpoCallResult:
    """Retry one isolated timed-out model request without replaying prior stages."""
    timeout_seconds = 180
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(
                awaitable_factory(), timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if attempt == attempts:
                raise HpoPhaseTimeout(
                    phase, round_idx, timeout_seconds, attempts,
                ) from exc
    raise AssertionError("unreachable")


def format_pydantic_validation_errors(error: ValidationError) -> list[dict]:
    """Return JSON-safe validation details without losing the failing field."""
    diagnostics = []
    for item in error.errors(include_url=False):
        input_value = item.get("input")
        try:
            json.dumps(input_value)
        except (TypeError, ValueError):
            input_value = repr(input_value)
        diagnostics.append({
            "field": ".".join(str(part) for part in item.get("loc", ())) or "<root>",
            "message": item.get("msg", "Validation failed."),
            "error_type": item.get("type", "value_error"),
            "input": input_value,
        })
    return diagnostics


def build_validation_error_summary(errors: list[dict]) -> str:
    """Build a concise single-line summary from structured validation diagnostics."""
    return "; ".join(
        f"{item['field']}: {item['message']} (input={item['input']!r})"
        for item in errors
    )


def normalize_hpo_decision(
    decision: HpoDecision,
    blocking_findings: list,
) -> tuple[HpoDecision, str | None]:
    """Make the decision boolean consistent with its structured findings."""
    expected_accept = not blocking_findings
    if decision.accept == expected_accept:
        return decision, None
    if expected_accept:
        return decision.model_copy(update={
            "accept": True,
            "reason": (
                "No hard errors or safety warnings; preference findings are advisory."
            ),
        }), "accept_set_true"
    return decision.model_copy(update={"accept": False}), "accept_set_false"


HPO_EXTERNAL_DEPLOYMENT_FIELDS = frozenset({
    "memory_category",
    "max_runtime_memory_mb",
    "max_model_size_mb",
    "max_parameters_m",
    "max_cpu_latency_ms",
})
HPO_QUALITY_ADVISORY_FIELDS = frozenset({
    "num_epochs", "input_size", "batch_size", "translate", "scale",
})


def demote_external_deployment_findings(
    decision: HpoDecision,
) -> tuple[HpoDecision, list[str]]:
    """Keep planning-owned deployment findings visible without blocking HPO."""
    findings = []
    demoted_fields = []
    for finding in decision.findings:
        if (
            finding.field in HPO_EXTERNAL_DEPLOYMENT_FIELDS
            and finding.severity in {"hard_error", "safety_warning"}
        ):
            finding = finding.model_copy(update={"severity": "preference"})
            demoted_fields.append(finding.field)
        findings.append(finding)
    if not demoted_fields:
        return decision, []
    return decision.model_copy(update={"findings": findings}), sorted(set(demoted_fields))


def demote_quality_heuristic_findings(decision: HpoDecision) -> tuple[HpoDecision, list[str]]:
    """Keep subjective quality advice visible after deterministic safety checks pass."""
    findings = []
    demoted = []
    for finding in decision.findings:
        if finding.field in HPO_QUALITY_ADVISORY_FIELDS and finding.severity != "preference":
            finding = finding.model_copy(update={"severity": "preference"})
            demoted.append(finding.field)
        findings.append(finding)
    if not demoted:
        return decision, []
    return decision.model_copy(update={"findings": findings}), sorted(set(demoted))


def _parse_recommended_value(value: str):
    """Parse the evaluator's text replacement without guessing at its type."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def compact_evaluator_context(context_data: dict) -> dict:
    """Keep reviewer inputs relevant and bounded instead of replaying full evidence."""
    keys = {
        "task", "classes", "selected_model_info", "selected_data",
        "performance_requirements", "robustness_requirements",
        "training_hardware", "available_hardware", "use_graphrag",
    }
    compact = {key: context_data[key] for key in keys if key in context_data}
    graph = context_data.get("hyperparameter_graph_context") or {}
    compact["hyperparameter_graph_context"] = {
        key: graph[key]
        for key in (
            "selected_model_id", "reference_configuration", "matched_adjustment_rules",
            "allowed_adjustment_fields",
        )
        if key in graph
    }
    return compact


HPO_REVIEW_HISTORY_LIMIT = 3


def rejected_review_record(
    *, round_idx: int, candidate: dict, decision: HpoDecision,
    previous_candidate: dict | None,
) -> dict:
    """Create a bounded, JSON-safe record for reviewer continuity."""
    ignored_comparison_fields = {"rationale", "llm_field_rationales"}
    comparable_candidate = {
        field: value for field, value in candidate.items()
        if field not in ignored_comparison_fields
    }
    canonical_candidate = json.dumps(
        comparable_candidate, sort_keys=True, separators=(",", ":"), default=str
    )
    changes = {}
    if previous_candidate is not None:
        comparable_previous = {
            field: value for field, value in previous_candidate.items()
            if field not in ignored_comparison_fields
        }
        for field in sorted(set(comparable_previous) | set(comparable_candidate)):
            old_value = comparable_previous.get(field)
            new_value = comparable_candidate.get(field)
            if old_value != new_value:
                changes[field] = {"from": old_value, "to": new_value}
    return {
        "round": round_idx,
        "candidate_fingerprint": hashlib.sha256(
            canonical_candidate.encode("utf-8")
        ).hexdigest()[:16],
        "active_configuration": candidate,
        "changes_from_previous_rejected_candidate": changes,
        "decision_reason": decision.reason,
        "blocking_findings": [
            finding.model_dump(mode="json") for finding in decision.findings
            if finding.severity in {"hard_error", "safety_warning"}
        ],
    }


def append_rejected_review_record(history: list[dict], record: dict) -> None:
    """Append a rejection while retaining only the most recent review rounds."""
    history.append(record)
    del history[:-HPO_REVIEW_HISTORY_LIMIT]


def build_evaluator_messages(
    *, system_prompt: str, evaluator_context_json: str, runtime_guidance: str,
    candidate: dict, rejected_review_history: list[dict],
) -> list[dict]:
    """Build a fresh review request with bounded structured rejection memory."""
    history_json = json.dumps(rejected_review_history, indent=2, default=str)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Relevant Task Context: {evaluator_context_json}\n\n"
                f"{runtime_guidance}\n\n"
                "Previous Rejected Review Rounds (oldest to newest, maximum 3):\n"
                f"{history_json}\n\n"
                "Use this history to verify that earlier blocking findings were resolved "
                "and to keep judgments consistent. Evaluate the current configuration on "
                "its own merits. Do not repeat a resolved finding, and explicitly explain "
                "any reversal of an earlier judgment. Matching fingerprints identify an "
                "unchanged active configuration.\n\n"
                "Proposed Active Runtime Configuration:\n"
                f"{json.dumps(candidate, indent=2)}"
            ),
        },
    ]


HPO_SCHEMA_FIELDS = frozenset().union(
    ClassificationConfigDraft.model_fields,
    DetectionConfigDraft.model_fields,
    VQAConfigModel.model_fields,
)

PIPELINE_OWNED_HPO_CONTEXT_FIELDS = frozenset({
    "task_type",
    "classes",
    "selected_data",
    "data_plan_constraints",
    "train_data_ratio",
    "val_data_ratio",
    "test_data_ratio",
    "model_name",
    "training_recipe_id",
})


def evaluator_active_configuration(proposal: BaseModel, task: str) -> dict:
    """Expose only fields that can affect the selected runtime to the evaluator."""
    data = proposal.model_dump(mode="json")
    if task == "classification":
        active_fields = active_classification_config_fields(data)
    elif task == "detection":
        active_fields = active_detection_config_fields(data)
    else:
        active_fields = set(data)

    active_fields -= PIPELINE_OWNED_HPO_CONTEXT_FIELDS
    active_fields |= {"rationale", "llm_field_rationales"}
    return {field: value for field, value in data.items() if field in active_fields}


def reconcile_optimizer_explanations(
    previous: BaseModel | None,
    proposal: BaseModel,
) -> BaseModel:
    """Keep explanations only for fields the optimizer owns and actually changed."""

    rationales = [
        item for item in getattr(proposal, "llm_field_rationales", [])
        if item.field not in PIPELINE_OWNED_HPO_CONTEXT_FIELDS
    ]
    if previous is None:
        rationale = " ".join(
            f"{item.field}: {item.reason}" for item in rationales
        ).strip()
        if not rationale:
            rationale = "Optimizer-owned fields were normalized against authoritative pipeline context."
        return proposal.model_copy(update={
            "llm_field_rationales": rationales,
            "rationale": rationale,
        })

    changed = changed_configuration_fields(previous, proposal)
    previous_rationales = {
        item.field: item
        for item in getattr(previous, "llm_field_rationales", [])
        if item.field not in PIPELINE_OWNED_HPO_CONTEXT_FIELDS
    }
    repaired_rationales = {item.field: item for item in rationales}
    merged_rationales = [
        repaired_rationales.get(field, item)
        for field, item in previous_rationales.items()
        if field not in changed
    ]
    merged_rationales.extend(
        repaired_rationales[field]
        for field in sorted(changed)
        if field in repaired_rationales
        and field not in PIPELINE_OWNED_HPO_CONTEXT_FIELDS
    )
    change_summary = "; ".join(
        f"{field}: {getattr(previous, field, None)!r} -> {getattr(proposal, field, None)!r}"
        for field in sorted(changed)
        if field not in PIPELINE_OWNED_HPO_CONTEXT_FIELDS
    )
    rationale = str(getattr(previous, "rationale", "")).strip()
    if change_summary:
        rationale = f"{rationale} Evaluator-authorized repair: {change_summary}.".strip()
    return proposal.model_copy(update={
        "llm_field_rationales": merged_rationales,
        "rationale": rationale,
    })


def discard_inactive_schema_findings(
    decision: HpoDecision,
    active_fields: set[str],
) -> tuple[HpoDecision, list[str]]:
    """Discard evaluator findings about known fields ignored by this runtime."""
    retained = []
    discarded = []
    for finding in decision.findings:
        if finding.field in HPO_SCHEMA_FIELDS and finding.field not in active_fields:
            discarded.append(finding.field)
        else:
            retained.append(finding)
    if not discarded:
        return decision, []
    return decision.model_copy(update={"findings": retained}), sorted(set(discarded))


def selected_model_reference(context_data: dict) -> str | None:
    """Read the concrete selected model identifier from pipeline state."""
    graph_context = context_data.get("hyperparameter_graph_context") or {}
    if graph_context.get("selected_model_id"):
        return str(graph_context["selected_model_id"])

    selected = context_data.get("selected_model_info") or {}
    models = selected.get("model") or []
    if isinstance(models, dict):
        models = [models]
    if models:
        for key in ("model_architecture", "model_name", "name", "id"):
            if models[0].get(key):
                return str(models[0][key])
    for key in ("model_id", "model_name"):
        if selected.get(key):
            return str(selected[key])
    return None


def selected_detection_model_id(context_data: dict) -> str | None:
    """Resolve the model-selection result to the detection HPO identifier."""
    references: list[str] = []
    graph_context = context_data.get("hyperparameter_graph_context") or {}
    for key in ("selected_model_id", "selected_registry_id"):
        if graph_context.get(key):
            references.append(str(graph_context[key]))

    selected = context_data.get("selected_model_info") or {}
    models = selected.get("model") or []
    if isinstance(models, dict):
        models = [models]
    if models:
        references.extend(
            str(models[0][key])
            for key in ("model_architecture", "model_name", "name", "id")
            if models[0].get(key)
        )
    references.extend(
        str(selected[key])
        for key in ("model_id", "model_name")
        if selected.get(key)
    )

    for reference in references:
        for model_id in DETECTION_HPO_MODEL_IDS:
            if model_ids_equivalent(model_id, reference):
                return model_id
        normalized = reference.strip().lower()
        for model in enabled_models("detection"):
            known_references = {model.id.lower(), *(alias.lower() for alias in model.aliases)}
            runtime_id = model.hpo_id or model.trainer_key
            if normalized in known_references and runtime_id in DETECTION_HPO_MODEL_IDS:
                return runtime_id
    return None


def discard_equivalent_model_findings(
    decision: HpoDecision,
    proposed_model: str | None,
    selected_model: str | None,
) -> tuple[HpoDecision, bool]:
    """Discard evaluator findings caused only by model-ID formatting differences."""
    if not proposed_model or not selected_model or not model_ids_equivalent(
        proposed_model, selected_model
    ):
        return decision, False

    findings = [finding for finding in decision.findings if finding.field != "model_name"]
    if len(findings) == len(decision.findings):
        return decision, False

    remaining_blockers = any(
        finding.severity in {"hard_error", "safety_warning"} for finding in findings
    )
    updates = {"findings": findings}
    if not remaining_blockers:
        updates.update({
            "accept": True,
            "reason": "The proposed model matches the selected model after canonicalization.",
            "suggestions": None,
        })
    return decision.model_copy(update=updates), True

# --- Knowledge Base Constant ---
PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE (Input Context):
You are receiving a full PipelineState JSON containing:
- `task`, `application_domain`, `classes`
- `performance_requirements`: Metrics, targets, and normalized `latency_category` / `accuracy_category` values such as VeryLow, Low, Medium, MediumHigh, or High.
- `available_hardware`: User-provided inference and deployment hardware.
- `training_hardware`: Server-selected training hardware; use it for batch size, workers, precision, and training feasibility.
- Never reduce training batch size, resolution, AMP, workers, or other training parameters because of
  `available_hardware`, deployment VRAM, or inference-memory constraints. These describe where the trained
  model will run, not where it is trained.
- `selected_data`: The authoritative train/validation/test dataset assignments from planning.
- `selected_model_info`: The architecture chosen in previous steps.
- `augmentation`, `preprocessing`: The data transformation strategies.
- `revision.active`: Confirmed user changes. Required HPO `set` operations are enforced
  deterministically; use preferred changes as advisory guidance when compatible.
"""

def changed_configuration_fields(previous: BaseModel, repaired: BaseModel) -> set[str]:
    """Return semantic top-level changes, excluding the explanatory rationale."""
    explanatory_fields = {"rationale", "llm_field_rationales"}
    previous_data = previous.model_dump(exclude=explanatory_fields)
    repaired_data = repaired.model_dump(exclude=explanatory_fields)
    return {
        field
        for field in previous_data.keys() | repaired_data.keys()
        if previous_data.get(field) != repaired_data.get(field)
    }


def merge_authorized_repair_fields(
    previous: BaseModel,
    repaired: BaseModel,
    allowed_fields: set[str],
) -> BaseModel:
    """Keep only evaluator-authorized value changes from an LLM repair."""
    explanatory_fields = {"rationale", "llm_field_rationales"}
    merged = repaired.model_dump(mode="json")
    for field, value in previous.model_dump(mode="json").items():
        if field not in allowed_fields and field not in explanatory_fields:
            merged[field] = value
    return type(repaired).model_validate(merged)


def evaluator_runtime_guidance(task: str, model_constraints: dict) -> str:
    """Describe authoritative runtime behavior used to classify findings."""
    guidance = (
        "Registered executable model constraints:\n"
        f"{json.dumps(model_constraints, indent=2)}\n"
        "Treat these registered capabilities as authoritative. A value that satisfies "
        "them is not a hard error merely because another valid value may be preferable."
    )
    if task == "classification":
        guidance += (
            " The classification input pipeline converts source images to RGB and resizes "
            "them to the configured image_size. A source dataset's native resolution or "
            "channel count therefore does not by itself make a supported image_size incompatible."
        )
    return guidance


def apply_owned_pipeline_fields(
    proposal: BaseModel,
    context: dict,
    *,
    selected_model_name: str | None = None,
) -> BaseModel:
    """Restore pipeline-owned fields and server runtime limits in every mode."""

    data = proposal.model_dump(mode="json")
    if selected_model_name is None:
        task = str(context.get("task", "")).lower()
        if task == "classification":
            selected_model_name = selected_classification_model_id(
                context.get("selected_model_info")
            )
        elif task == "detection":
            selected_model_name = selected_detection_model_id(context)
    if selected_model_name is not None:
        data["model_name"] = selected_model_name
    data["classes"] = list(context.get("classes") or [])
    data["selected_data"] = list(context.get("selected_data") or [])
    if str(context.get("task", "")).lower() == "detection":
        data["data_plan_constraints"] = dict(
            context.get("data_plan_constraints") or {}
        )
    data.update(planned_split_ratios(context) or {})

    # A recipe ID is ontology provenance, never a free-form LLM decision.
    if not bool(context.get("use_graphrag", True)):
        data["training_recipe_id"] = ""

    hardware = context.get("training_hardware") or {}
    max_batch_size = hardware.get("max_batch_size")
    batch_size = data.get("batch_size")
    if (
        isinstance(max_batch_size, int)
        and isinstance(batch_size, int)
        and batch_size != -1
        and batch_size > max_batch_size
    ):
        data["batch_size"] = max_batch_size
    if hardware and not bool(hardware.get("supports_amp", False)):
        if "amp" in data:
            data["amp"] = False
        if "precision" in data:
            data["precision"] = "fp32"
    if "workers" in data and isinstance(hardware.get("workers"), int):
        data["workers"] = hardware["workers"]

    return type(proposal).model_validate(data)


def complete_required_field_rationales(
    proposal: BaseModel,
    required_fields: set[str],
    graph_context: dict | None = None,
) -> tuple[BaseModel, list[str]]:
    """Add auditable provenance when generation omitted explanation metadata."""
    if not required_fields:
        return proposal, []

    graph_fields = set(((graph_context or {}).get("reference_configuration") or {}).keys())
    rationales = list(getattr(proposal, "llm_field_rationales", []))
    existing = {str(item.field) for item in rationales}
    rationale_type = type(proposal).model_fields["llm_field_rationales"].annotation
    item_type = rationale_type.__args__[0]
    added: list[str] = []

    for field in sorted(required_fields - existing):
        if field in graph_fields:
            reason = (
                f"The value for {field} was retained from or selected relative to "
                "the GraphRAG-grounded configuration."
            )
        else:
            reason = (
                f"The value for {field} was selected by the optimizer and passed "
                "the executable schema constraints."
            )
        rationales.append(item_type.model_validate({
            "field": field,
            "reason": reason,
        }))
        added.append(field)

    if not added:
        return proposal, []

    rationale = str(getattr(proposal, "rationale", "")).strip()
    provenance_note = (
        " Deterministic provenance was completed for: " + ", ".join(added) + "."
    )
    return proposal.model_copy(update={
        "llm_field_rationales": rationales,
        "rationale": (rationale + provenance_note).strip(),
    }), added


def validate_hpo_cross_field_configuration(
    proposal: BaseModel,
    context_data: dict,
) -> list[dict]:
    """Return deterministic schedule/override violations for an HPO proposal."""
    if context_data.get("task") != "detection":
        return []
    data = proposal.model_dump(mode="json")
    model_name = str(data.get("model_name", ""))
    if not model_name.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_")):
        return []

    epochs = int(data.get("num_epochs") or 0)
    patience = int(data.get("patience") or 0)
    warmup = float(data.get("warmup_epochs") or 0)
    mosaic = float(data.get("mosaic") or 0)
    close_mosaic = int(data.get("close_mosaic") or 0)
    violations: list[dict] = []

    if epochs >= 10 and warmup > epochs * 0.20:
        violations.append({
            "fields": ["num_epochs", "warmup_epochs"],
            "message": (
                f"warmup_epochs={warmup:g} consumes more than 20% of the {epochs}-epoch schedule."
            ),
        })

    if mosaic > 0 and close_mosaic > 0:
        normal_mosaic_epochs = epochs - close_mosaic
        minimum_normal_phase = min(20, max(5, epochs // 2))
        if normal_mosaic_epochs < minimum_normal_phase:
            violations.append({
                "fields": ["num_epochs", "close_mosaic"],
                "message": (
                    f"mosaic is active for only {normal_mosaic_epochs} epoch(s) before "
                    f"close_mosaic={close_mosaic}; require at least {minimum_normal_phase}."
                ),
            })

    if epochs >= 10 and 0 < patience < max(3, int(epochs * 0.10)):
        violations.append({
            "fields": ["num_epochs", "patience"],
            "message": (
                f"patience={patience} is too short for a {epochs}-epoch fine-tuning schedule; "
                "use 0 to disable early stopping or a coherent patience value."
            ),
        })

    return violations


def hpo_advisory_findings(proposal: BaseModel, context_data: dict) -> list[dict]:
    """Return quality recommendations that must remain visible but non-blocking."""
    if context_data.get("task") != "detection":
        return []
    data = proposal.model_dump(mode="json")
    model_name = str(data.get("model_name", ""))
    if not model_name.startswith(("yolov8_", "yolov10_", "yolov11_", "yolov12_")):
        return []
    findings: list[dict] = []
    reference = ((context_data.get("hyperparameter_graph_context") or {}).get("reference_configuration") or {})
    reference_epochs = int(reference.get("num_epochs") or 0)
    epochs = int(data.get("num_epochs") or 0)
    query = str(context_data.get("user_query") or "").lower()
    time_limited = any(phrase in query for phrase in (
        "time limit", "training time", "finish within", "maximum epochs",
        "max epochs", "only have", "budget of", "quick experiment", "smoke test",
    ))
    threshold = max(20, math.ceil(reference_epochs * 0.75)) if reference_epochs else 0
    if reference_epochs >= 40 and epochs < threshold and not time_limited:
        findings.append({
            "field": "num_epochs",
            "severity": "preference",
            "reason": (
                f"num_epochs={epochs} is substantially below the GraphRAG reference of "
                f"{reference_epochs}; consider at least {threshold} unless the reduction is justified."
            ),
            "recommended_value": str(threshold),
            "rule_id": "hpo.recipe_epoch_deviation.v1",
        })
    robustness = context_data.get("robustness_requirements") or {}
    requested_scales = {
        str(value).strip().lower()
        for value in (
            robustness.get("object_scale", [])
            if isinstance(robustness, dict)
            else getattr(robustness, "object_scale", [])
        )
    }
    query = str(context_data.get("user_query") or "").lower()
    small_objects_requested = "small" in requested_scales or any(
        phrase in query
        for phrase in ("small object", "small and far", "far away", "distant object")
    )
    hardware = context_data.get("training_hardware") or {}
    budget = float(hardware.get("training_memory_budget_gb") or 0)
    if small_objects_requested:
        translate = float(data.get("translate") or 0)
        scale = float(data.get("scale") or 0)
        input_size = int(data.get("input_size") or 0)
        batch_size = int(data.get("batch_size") or 0)
        if translate == 0 and scale == 0:
            findings.append({
                "field": "translate",
                "severity": "preference",
                "reason": (
                    "Small-object detection requires bounded geometric variation; do not "
                    "disable both translate and scale without measured dataset evidence."
                ),
                "recommended_value": "0.05",
                "rule_id": "hpo.small_object_augmentation.v1",
            })
        if 0 < budget <= 6 and input_size < 768:
            findings.append({
                "field": "input_size", "severity": "preference",
                "reason": "Consider input_size=768 for small objects on the low-memory profile; resource validation remains authoritative.",
                "recommended_value": "768", "rule_id": "hpo.small_object_resolution.v1",
            })
        if 0 < budget <= 6 and input_size >= 768 and batch_size > 2:
            findings.append({
                "field": "batch_size", "severity": "preference",
                "reason": "Consider batch_size<=2 with input_size>=768 on a <=6 GiB training budget.",
                "recommended_value": "2", "rule_id": "hpo.small_object_resolution.v1",
            })
    return findings


def training_hardware_role_findings(
    proposal: BaseModel,
    context_data: dict,
) -> list[dict]:
    """Reject training choices justified by the separate deployment GPU."""
    training = context_data.get("training_hardware") or {}
    deployment = context_data.get("available_hardware") or {}
    training_gpu = str(training.get("gpu_type") or "").lower()
    deployment_gpu = str(deployment.get("gpu_type") or "").lower()
    if not deployment_gpu or not training_gpu or deployment_gpu == training_gpu:
        return []

    def compact(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())

    deployment_marker = compact(deployment_gpu)
    # GPU names commonly include vendor/product adjectives. The stable RTX token
    # is enough to identify which device a batch rationale cites.
    deployment_tokens = [
        token for token in deployment_marker.split("nvidia") if token
    ]
    marker_candidates = {deployment_marker, *deployment_tokens}
    for prefix in ("nvidiageforce", "nvidia", "geforce"):
        if deployment_marker.startswith(prefix):
            marker_candidates.add(deployment_marker[len(prefix):])
    marker_candidates.discard("")

    batch_reasons = [
        str(item.reason)
        for item in getattr(proposal, "llm_field_rationales", []) or []
        if getattr(item, "field", None) == "batch_size"
    ]
    if not batch_reasons:
        batch_reasons = [
            sentence
            for sentence in str(getattr(proposal, "rationale", "")).split(".")
            if "batch" in sentence.lower()
        ]
    safe_phrases = (
        "inference only", "deployment only", "not used for training",
        "does not determine", "must not determine", "separate training",
    )
    for reason in batch_reasons:
        normalized = compact(reason)
        if (
            any(marker in normalized for marker in marker_candidates)
            and not any(phrase in reason.lower() for phrase in safe_phrases)
        ):
            return [{
                "field": "batch_size",
                "severity": "safety_warning",
                "reason": (
                    f"The batch-size rationale cites deployment GPU '{deployment.get('gpu_type')}', "
                    f"but training runs on '{training.get('gpu_type')}'. Re-evaluate batch size "
                    "using training_hardware and its hardware-safe candidates only."
                ),
                "rule_id": "hpo.training_hardware_role.v1",
            }]
    return []


async def generate_and_evaluate_hpo(
    json_data: str,
    job_id: str,
    max_rounds: int = 3,
    max_generation_attempts: int = 3,
) -> tuple[Union[BaseModel, None], Union[HpoDecision, None]]:
    """
    Generates one graph-grounded proposal, then only repairs fields explicitly
    identified by the evaluator. It never treats evaluator rounds as HPO trials.
    """
    client = AsyncOpenAI() 
    
    # 1. Parse the input to determine the task
    try:
        context_data = json.loads(json_data)
        task = context_data.get("task", "").lower()
    except json.JSONDecodeError:
        print("Error: Invalid JSON data provided.")
        return None, None

    use_graphrag = bool(context_data.get("use_graphrag", True))
    if use_graphrag:
        authority_guidance = (
            "Use the GraphRAG reference recipe as a starting point. Preserve or adapt each "
            "schema-configurable field based on the selected model, datasets, task, and "
            "hardware, and explain the decision. GraphRAG is evidence, not field authority."
        )
    else:
        authority_guidance = (
            "Choose any schema-valid, runtime-compatible hyperparameter values. Task, "
            "classes, selected model, dataset assignments, split ratios, hardware limits, "
            "and structural model-family invariants remain pipeline-owned."
        )

    # 2. Map the task to the correct Pydantic schema
    schema_mapping = {
        "classification": ClassificationConfigDraft,
        "detection": DetectionConfigDraft,
        "visual question answering": VQAConfigModel
    }
    
    if task not in schema_mapping:
        print(f"Error: Unsupported task '{task}'.")
        return None, None
    
    TargetSchema = schema_mapping[task]
    selected_model_name = None
    model_constraints: dict = {}
    if task == "classification":
        selected_model_name = selected_classification_model_id(
            context_data.get("selected_model_info")
        )
        if selected_model_name is None:
            print("Error: No supported selected classification model was found.")
            return None, None
        model_constraints = classification_prompt_constraints(selected_model_name)
    elif task == "detection":
        selected_model_name = selected_detection_model_id(context_data)
        if selected_model_name is None:
            print("Error: No supported selected detection model was found.")
            return None, None
        model_constraints = detection_prompt_constraints(selected_model_name)

    # 3. Setup the initial Optimizer conversation history
    optimizer_messages = [
        {
            "role": "system",
            "content": (
                f"{PIPELINE_STATE_BLUEPRINT}\n\n"
                "You are a careful Machine Learning engineer proposing an executable initial configuration. "
                "Review the `selected_model_info`, `selected_data`, and `task` from the state. "
                "Based on this context, generate a safe hyperparameter configuration. "
                f"Authority mode: {authority_guidance} "
                f"\n\n{load_cv_skill('diagnose')}\n\n{load_cv_skill('recipe-adaptation')}\n\n"
                f"{load_cv_skill('data-problems')}\n\n"
                "The state may include `hyperparameter_graph_context.reference_configuration`. "
                "Treat it as a reference recipe, not "
                "an immutable baseline. Treat graph rules listed in `matched_adjustment_rules` as evidence-backed "
                "recommendations rather than mandatory overrides. Follow or adapt them using the recipe, hardware-safe "
                "candidates, dataset, and task requirements, and explain the choice. Model-scoped rules whose conditions "
                "were not matched are informational and must not be claimed as applicable. "
                "Fields listed in `fields_requiring_llm_completion` are deliberately missing from the recipe: choose safe "
                "values for them from the model, data, hardware, schema constraints, and retrieved evidence. Do not describe "
                "an LLM-completed value as recipe-sourced. A recipe-backed Ultralytics optimizer='auto' may be changed to a "
                "supported explicit optimizer when the data, effective batch, or stability evidence specifically justifies it. "
                "Use `selected_model_info` only to identify the selected architecture; never copy epochs, patience, "
                "optimizer, losses, resolution, batch size, or augmentations from model selection. The HPO stage owns "
                "those fields. You may adapt GraphRAG defaults, but for each changed recipe-backed field state whether "
                "the basis is a user requirement, dataset profile, training hardware, matched adjustment rule, or runtime "
                "constraint. Never describe an adapted value as GraphRAG-sourced. Evaluate epochs, patience, warmup, "
                "and close_mosaic as one coherent schedule. "
                "When optimizer='auto', Ultralytics owns learning_rate and momentum, so do not claim to tune them. To control "
                "learning_rate, select an explicit optimizer and justify both optimizer_name and learning_rate. "
                "Whenever a matched GraphRAG recommendation pairs batch_size and learning_rate, treat them as a joint "
                "decision. If you choose a different hardware-safe batch with an explicit optimizer, scale or retune its "
                "learning rate from the rule's stated reference pair and explain the calculation. Do not scale learning "
                "rate merely because optimizer='auto' is using a different batch. "
                "For other fields absent from the baseline, use schema defaults unless the PipelineState provides a concrete reason. "
                "`training_recipe_id` is provenance, not a name you may invent: when `use_graphrag` is false it MUST be "
                "the empty string; when GraphRAG is enabled copy it exactly from the reference recipe. "
                "You must rely ONLY on standard, universally accepted heuristics. Do not attempt creative, novel, or experimental configurations. "
                "If a parameter is standard, use the standard value. Hallucination or guessing outside of the provided context is strictly prohibited. "
                "Pay strict attention to memory constraints and learning rates for the selected architecture. "
                "Use `training_hardware` as the sole hardware authority for batch size, input resolution, workers, "
                "AMP, precision, and training-memory feasibility. `available_hardware`, deployment GPU/VRAM, "
                "deployment_constraints, and inference-memory estimates must never be used to reduce training "
                "hyperparameters. They describe inference after training. "
                "For TorchVision Faster R-CNN, RetinaNet, and SSD, every YOLO-only augmentation field "
                "(`mosaic`, `mixup`, `cutmix`, `copy_paste`, `degrees`, `translate`, `scale`, `fliplr`, "
                "`hsv_h`, `hsv_s`, `hsv_v`, and `close_mosaic`) must be zero. Use the active "
                "`horizontal_flip_probability` field for supported horizontal flipping. "
                "Respect `performance_requirements.latency_category` and `performance_requirements.accuracy_category` "
                "when present: favor stronger training choices for MediumHigh or High accuracy requirements, "
                "efficient settings for VeryLow or Low latency requirements, and moderate defaults when both matter. "
                "Always fill the `rationale` field. Explicitly mention every LLM-completed or evaluator-adjusted field and why "
                "its value was selected."
                "\n\nOUTPUT FORMAT REQUIREMENTS: Return a single JSON object matching the executable schema for the task. "
                "The JSON MUST include a top-level llm_field_rationales array with one object per field that you set or completed. "
                "Each entry must be an object with a string `field` and a string `reason`. Do not include explanatory prose outside the JSON object. "
                "Example: {\"llm_field_rationales\": [{\"field\": \"patience\", \"reason\": \"Scaled to the small validation set and training duration\"}] }"
            )
        },
        {
            "role": "user",
            "content": (
                f"Task Context:\n{json_data}\n\n"
                "Mandatory executable model constraints:\n"
                f"{json.dumps(model_constraints, indent=2)}\n\n"
                "These constraints come from the registered runtime and must be obeyed. "
                "Choose weights and training_mode only from the supplied executable capabilities; "
                "do not infer implementation support from general ML knowledge. "
                "The selected model, classes, selected_data, and compatibility split ratios are owned by earlier pipeline steps."
            )
        }
    ]

    evaluator_system_prompt = (
        f"{PIPELINE_STATE_BLUEPRINT}\n\n"
        "You are a strict Senior Machine Learning Reviewer. Your job is to review proposed hyperparameters against the provided PipelineState. "
        "Look for catastrophic errors: Out of Memory risks based on the chosen model, exploding gradients, or logical mismatches. "
        "Check that the proposal is consistent with `performance_requirements.latency_category` and "
        "`performance_requirements.accuracy_category` when they are present. "
        "Be ruthless but constructive. If it is safe and adheres to standard practices, accept it."
        " Return structured findings. Every hard_error or safety_warning MUST name exactly one top-level configuration `field`. "
        "Evaluate only optimizer-owned fields present in Proposed Active Runtime Configuration. Earlier pipeline outputs "
        "such as classes, selected_data, split ratios, data_plan_constraints, selected model identity, and recipe provenance "
        "are authoritative context: use them to evaluate optimizer-owned fields but never return a blocking finding against them. "
        "GraphRAG recipe-backed hyperparameters that appear in the active configuration remain optimizer-owned and adjustable. "
        "Do not infer candidate fields from "
        "GraphRAG metadata or the broader PipelineState. A blocking finding must name a field present in that active "
        "configuration. `available_hardware` and deployment VRAM describe inference only. `training_hardware` is "
        "the sole authority for training batch size, resolution, workers, AMP, precision, and memory feasibility. "
        "Treat use of a different deployment GPU to justify training batch or resolution as a safety error on that field. "
        "For YOLO and RT-DETR with scheduler_name='linear', final_learning_rate_factor is an active runtime field passed "
        "to Ultralytics as lrf; it is not an inactive sentinel and must not be set to zero or a near-zero placeholder. "
        "The standard grounded default is 0.01. "
        "Use `preference` for optional alternatives; preferences must not cause rejection. Cite a recipe/rule ID when available. "
        "Reject only for concrete compatibility, safety, constraint, or ontology-grounding problems—not because another valid value might perform better."
        " The configuration schema is deliberately union-free: optimizer- and scheduler-specific fields are always present. "
        "Inactive schema sentinel fields are filtered out before runtime. A sentinel is a validation placeholder, not a "
        "tuning decision. Do not flag an inactive field merely because it exists in the broader schema, "
        "and never recommend removing a required field or setting it to null. `alpha` belongs to RMSprop, `eps` belongs to "
        "AdamW/RMSprop, and `beta1`/`beta2` belong to AdamW. For StepLR or no scheduler, min_learning_rate=0 is the required "
        "inactive sentinel. `scheduler_step_size` and `scheduler_gamma` remain schema-valid but are ignored for cosine/no scheduler. "
        "Use preference, not safety_warning, for optional cleanup or alternative valid values. "
        "Runtime-supported but potentially inefficient or suboptimal values are preferences, not blocking findings. "
        "Deployment constraints are owned and validated by model selection. Report residual uncertainty for "
        "memory_category, max_runtime_memory_mb, max_model_size_mb, max_parameters_m, or max_cpu_latency_ms "
        "as a preference; do not reject an otherwise valid HPO configuration for those fields. "
        "total_estimated_vram_gb is the estimated runtime footprint comparable to max_runtime_memory_mb; "
        "practical_min_vram_gb is conservative provisioning guidance, not measured runtime consumption. "
        "Do not infer an input incompatibility from a classification dataset's native resolution or channel count when "
        "the supplied registered constraints allow the configured size; the runtime converts images to RGB and resizes them. "
        "If accept=true, all findings must be preferences; hard_error and safety_warning require accept=false."
        " Recipe deviations, preferred small-object resolution, and augmentation-strength advice are preferences, "
        "not execution failures, unless a supplied runtime capability or resource check is actually violated."
    )

    print(f"Starting Multi-Agent Optimization for task: {task.upper()} (Job ID: {job_id})")

    last_reason = None
    last_suggestions = None
    rejection_count = 0
    previous_proposal = None
    repairable_fields: set[str] = set()
    authorized_repair_fields: set[str] = set()
    proposal = None
    decision = None
    round_diagnostics: list[dict] = []
    rejected_review_history: list[dict] = []
    previous_rejected_candidate: dict | None = None
    graph_context = context_data.get("hyperparameter_graph_context") or {}
    evaluator_context_json = json.dumps(compact_evaluator_context(context_data))


    # 4. The Evaluator-Optimizer Loop. Generation/schema retries have their
    # own budget so they cannot consume all evaluator rounds.
    evaluation_round = 0
    generation_attempt = 0
    while evaluation_round < max_rounds and generation_attempt < max_generation_attempts:
        round_idx = evaluation_round + 1
        generation_attempt += 1
        print(
            f"\n--- Evaluation Round {round_idx}/{max_rounds}, "
            f"Generation Attempt {generation_attempt}/{max_generation_attempts} ---"
        )
        
        # Phase A: The Optimizer proposes a configuration (wrapped in try/except for self-healing)
        try:
            optimizer_response = await _hpo_model_call(
                lambda: run_planning_completion(
                        job_id=job_id,
                        operation="hpo_optimizer",
                        model=PLANNING_MODEL,
                        awaitable=client.beta.chat.completions.parse(
                            model=PLANNING_MODEL,
                            messages=optimizer_messages,
                            response_format=TargetSchema,
                        ),
                    ),
                phase="optimizer",
                round_idx=round_idx,
            )

            opt_message = optimizer_response.choices[0].message
            
            if opt_message.refusal:
                print(f"❌ Optimizer refused to generate a configuration: {opt_message.refusal}")
                return None, None

            parsed_output = opt_message.parsed
            if task == "classification":
                assert selected_model_name is not None
                completed_data, _ = complete_classification_config(
                    parsed_output,
                    context_data,
                    selected_model_name,
                )
                proposal = ClassificationConfigModel.model_validate(completed_data)
            elif task == "detection":
                assert selected_model_name is not None
                proposal, completion_adjustments = complete_detection_config(
                    parsed_output,
                    context_data,
                    selected_model_name,
                )
                if completion_adjustments:
                    round_diagnostics.append({
                        "round": round_idx,
                        "generation_attempt": generation_attempt,
                        "phase": "draft_completion",
                        "reason": "applied_authoritative_detection_fields",
                        "adjustments": completion_adjustments,
                    })
            else:
                proposal = parsed_output
            proposal = apply_owned_pipeline_fields(
                proposal,
                context_data,
                selected_model_name=selected_model_name,
            )
            required_hpo_overrides = hpo_override_values(context_data)
            if required_hpo_overrides:
                proposal_data = proposal.model_dump(mode="json")
                unknown_fields = set(required_hpo_overrides) - set(proposal_data)
                if unknown_fields:
                    raise ValueError(
                        "User revision names unsupported hyperparameter fields: "
                        f"{sorted(unknown_fields)}"
                    )
                proposal_data.update(required_hpo_overrides)
                proposal = type(proposal).model_validate(proposal_data)
            if previous_proposal is not None:
                proposal = merge_authorized_repair_fields(
                    previous_proposal,
                    proposal,
                    repairable_fields,
                )
            proposal = reconcile_optimizer_explanations(
                previous_proposal,
                proposal,
            )
            proposal_changes = (
                changed_configuration_fields(previous_proposal, proposal)
                if previous_proposal is not None
                else set()
            )

            if task in {"classification", "detection"} and use_graphrag:
                proposal_data = proposal.model_dump(mode="json")
                active_fields = (
                    active_classification_config_fields(proposal_data)
                    if task == "classification"
                    else active_detection_config_fields(proposal_data)
                )
                active_fields -= PIPELINE_OWNED_HPO_CONTEXT_FIELDS
                active_explanations = [
                    item
                    for item in getattr(proposal, "llm_field_rationales", [])
                    if item.field in active_fields
                ]
                if len(active_explanations) != len(
                    getattr(proposal, "llm_field_rationales", [])
                ):
                    proposal = proposal.model_copy(
                        update={"llm_field_rationales": active_explanations}
                    )
                explained_fields = {
                    item.field for item in active_explanations
                }
                required_explanations = llm_controlled_fields(
                    proposal_data,
                    graph_context,
                    type(proposal),
                ) | (
                    proposal_changes & repairable_fields
                )
                required_explanations &= active_fields
                missing_explanations = required_explanations - explained_fields
                if missing_explanations:
                    proposal, completed_fields = complete_required_field_rationales(
                        proposal,
                        missing_explanations,
                        graph_context=graph_context if use_graphrag else None,
                    )
                    diagnostic = {
                        "round": round_idx,
                        "generation_attempt": generation_attempt,
                        "phase": "optimizer_precheck",
                        "reason": "auto_completed_field_rationales",
                        "fields": completed_fields,
                    }
                    round_diagnostics.append(diagnostic)
                    active_explanations = list(proposal.llm_field_rationales)

            if task in {"classification", "detection"}:
                cross_field_violations: list[dict] = []
                try:
                    cross_field_violations = validate_hpo_cross_field_configuration(
                        proposal,
                        context_data,
                    )
                    if cross_field_violations:
                        raise ValueError(
                            "Cross-field HPO validation failed: "
                            + "; ".join(item["message"] for item in cross_field_violations)
                        )
                    validate_executable_recipe_config(
                        proposal.model_dump(mode="json")
                    )
                    validate_training_resource_config(
                        proposal.model_dump(mode="json")
                    )
                except ValueError as exc:
                    diagnostic = {
                        "round": round_idx,
                        "generation_attempt": generation_attempt,
                        "phase": "executable_validation",
                        "reason": "executable_config_invalid",
                        "message": str(exc),
                        "cross_field_violations": cross_field_violations,
                    }
                    round_diagnostics.append(diagnostic)
                    optimizer_messages.append({
                        "role": "user",
                        "content": (
                            "Your proposal failed deterministic execution validation: "
                            f"{exc}. Correct the named interacting fields together and preserve all "
                            "pipeline-owned values. GraphRAG defaults remain context, not immutable values."
                        ),
                    })
                    continue

            if previous_proposal is not None:
                unauthorized_fields = (
                    proposal_changes - repairable_fields - set(required_hpo_overrides)
                )
                if unauthorized_fields:
                    diagnostic = {
                        "round": round_idx,
                        "generation_attempt": generation_attempt,
                        "phase": "repair_validation",
                        "reason": "unauthorized_repair_changes",
                        "fields": sorted(unauthorized_fields),
                        "allowed_fields": sorted(repairable_fields),
                    }
                    round_diagnostics.append(diagnostic)
                    print(f"⚠️ Round {round_idx} skipped before evaluation: {json.dumps(diagnostic)}")
                    optimizer_messages.append({
                        "role": "user",
                        "content": (
                            "The repair was rejected because it changed fields the evaluator did not authorize: "
                            f"{sorted(unauthorized_fields)}. Restore those fields exactly. You may change ONLY: "
                            f"{sorted(repairable_fields)}. Return the complete configuration with all other values unchanged."
                        ),
                    })
                    continue

        except ValidationError as e:
                validation_errors = format_pydantic_validation_errors(e)
                diagnostic = {
                    "round": round_idx,
                    "generation_attempt": generation_attempt,
                    "phase": "schema_validation",
                    "reason": "pydantic_validation_error",
                    "errors": validation_errors,
                }
                round_diagnostics.append(diagnostic)
                error_summary = build_validation_error_summary(validation_errors)
                print(f"⚠️ Pydantic Validation Error in Round {round_idx}: {error_summary}")

                # Feed the exact validation error back to the LLM so it can fix it
                error_feedback = (
                    "Your previous JSON output failed strict schema validation. "
                    f"Errors: {error_summary} "
                    "Correct each invalid top-level field exactly; do not remove required fields or set them to null."
                )
                optimizer_messages.append({"role": "user", "content": error_feedback})
                continue
        # Record the successful proposal in the optimizer's history
        optimizer_messages.append({
            "role": "assistant",
            # Repairs must copy the normalized configuration that the evaluator
            # actually sees, rather than the pre-normalization LLM draft.
            "content": proposal.model_dump_json(indent=2),
        })
        generation_attempt = 0
        evaluation_round += 1

        # Phase B: The Evaluator reviews the proposal
        evaluator_candidate = evaluator_active_configuration(proposal, task)
        evaluator_active_fields = set(evaluator_candidate) - {
            "rationale",
            "llm_field_rationales",
        }
        evaluator_messages = build_evaluator_messages(
            system_prompt=evaluator_system_prompt,
            evaluator_context_json=evaluator_context_json,
            runtime_guidance=evaluator_runtime_guidance(task, model_constraints),
            candidate=evaluator_candidate,
            rejected_review_history=rejected_review_history,
        )
        
        evaluator_response = await _hpo_model_call(
            lambda: run_planning_completion(
                    job_id=job_id,
                    operation="hpo_evaluator",
                    model=PLANNING_MODEL,
                    awaitable=client.beta.chat.completions.parse(
                        model=PLANNING_MODEL,
                        messages=evaluator_messages,
                        response_format=HpoDecision,
                    ),
                ),
            phase="evaluator",
            round_idx=round_idx,
        )
        
        eval_message = evaluator_response.choices[0].message
        
        if eval_message.refusal:
             print(f"❌ Evaluator refused to generate a decision: {eval_message.refusal}")
             return proposal, None
             
        decision = eval_message.parsed

        decision, discarded_model_finding = discard_equivalent_model_findings(
            decision,
            str(getattr(proposal, "model_name", "")),
            selected_model_reference(context_data),
        )
        if discarded_model_finding:
            diagnostic = {
                "round": round_idx,
                "phase": "evaluator_normalization",
                "reason": "equivalent_model_identifiers",
                "proposed_model": str(getattr(proposal, "model_name", "")),
                "selected_model": selected_model_reference(context_data),
            }
            round_diagnostics.append(diagnostic)
            print(f"ℹ️ Evaluator model finding discarded: {json.dumps(diagnostic)}")

        decision, demoted_fields = demote_external_deployment_findings(decision)
        if demoted_fields:
            diagnostic = {
                "round": round_idx,
                "phase": "evaluator_normalization",
                "reason": "planning_owned_deployment_constraint_is_advisory",
                "fields": sorted(demoted_fields),
            }
            round_diagnostics.append(diagnostic)
            print(f"ℹ️ Evaluator deployment finding demoted: {json.dumps(diagnostic)}")

        decision, demoted_quality_fields = demote_quality_heuristic_findings(decision)
        if demoted_quality_fields:
            diagnostic = {
                "round": round_idx, "phase": "evaluator_normalization",
                "reason": "quality_heuristic_is_advisory",
                "fields": demoted_quality_fields,
            }
            round_diagnostics.append(diagnostic)

        decision, discarded_inactive_fields = discard_inactive_schema_findings(
            decision,
            evaluator_active_fields,
        )
        if discarded_inactive_fields:
            diagnostic = {
                "round": round_idx,
                "phase": "evaluator_normalization",
                "reason": "inactive_or_cross_task_schema_findings_discarded",
                "fields": discarded_inactive_fields,
            }
            round_diagnostics.append(diagnostic)
            print(f"ℹ️ Inactive evaluator findings discarded: {json.dumps(diagnostic)}")

        hardware_role_findings = training_hardware_role_findings(proposal, context_data)
        if hardware_role_findings:
            existing = {(finding.field, finding.rule_id) for finding in decision.findings}
            additions = [
                HpoFinding.model_validate(item)
                for item in hardware_role_findings
                if (item["field"], item.get("rule_id")) not in existing
            ]
            decision = decision.model_copy(update={"findings": [*decision.findings, *additions]})
            round_diagnostics.append({
                "round": round_idx,
                "phase": "hardware_role_validation",
                "reason": "deployment_hardware_used_for_training_choice",
                "findings": hardware_role_findings,
            })

        advisory = hpo_advisory_findings(proposal, context_data)
        if advisory:
            existing = {(finding.field, finding.rule_id) for finding in decision.findings}
            additions = [
                HpoFinding.model_validate(item)
                for item in advisory
                if (item["field"], item.get("rule_id")) not in existing
            ]
            decision = decision.model_copy(update={"findings": [*decision.findings, *additions]})
            round_diagnostics.append({
                "round": round_idx, "phase": "advisory_validation",
                "reason": "non_blocking_quality_recommendations", "findings": advisory,
            })

        blocking_findings = [
            finding for finding in decision.findings
            if finding.severity in {"hard_error", "safety_warning"}
        ]
        actionable_findings = [
            finding for finding in blocking_findings
            if finding.field in evaluator_active_fields
        ]
        unknown_blocking_findings = [
            finding for finding in blocking_findings
            if finding.field not in evaluator_active_fields
        ]
        decision, normalization = normalize_hpo_decision(
            decision,
            blocking_findings,
        )
        if normalization:
            diagnostic = {
                "round": round_idx,
                "phase": "evaluator_normalization",
                "reason": "accept_inconsistent_with_findings",
                "fields": sorted(finding.field for finding in blocking_findings),
                "resolution": normalization,
            }
            round_diagnostics.append(diagnostic)
            print(f"⚠️ Evaluator decision normalized: {json.dumps(diagnostic)}")
        
        # Phase C: Format and execute the planning log step
        candidate_rationale = getattr(proposal, "rationale", "No rationale provided.")
        
        input_context_str = f"Constraints:\n{json_data}\n"
        if last_reason:
            input_context_str += f"\nPrior Feedback applied this round:\nReason: {last_reason}\nSuggestions: {last_suggestions}"
            
        round_log_rationale = (
            f"--- Suggester Rationale ---\n{candidate_rationale}\n\n"
            f"--- Optimizer Decision ---\nAccepted: {decision.accept}\n"
            f"Reasoning: {decision.reason}\n"
            f"Suggestions: {decision.suggestions}"
        )
        
        output_summary_dict = {
            "proposal": proposal.model_dump(),
            "decision": decision.model_dump(),
            "diagnostics": round_diagnostics,
        }

        log_planning_step(
            job_id=job_id,
            step_name="Hyperparameter Negotiation",
            input_context=input_context_str,
            rationale=round_log_rationale,
            output_summary=output_summary_dict,
            round_num=round_idx
        )

        if unknown_blocking_findings:
            diagnostic = {
                "round": round_idx,
                "phase": "evaluator_validation",
                "reason": "unknown_blocking_finding_fields",
                "fields": sorted(finding.field for finding in unknown_blocking_findings),
            }
            round_diagnostics.append(diagnostic)
            print(f"❌ Evaluator returned non-repairable findings: {json.dumps(diagnostic)}")
            decision._authorized_repair_fields = set(authorized_repair_fields)
            decision._diagnostics = list(round_diagnostics)
            return proposal, decision

        # Preferences are recorded but never trigger configuration search/repair.
        if not blocking_findings:
            print("✅ Evaluator accepted the configuration!")
            decision._authorized_repair_fields = set(authorized_repair_fields)
            decision._diagnostics = list(round_diagnostics)
            return proposal, decision
        else:
            print(f"❌ Evaluator rejected the configuration. Reason: {decision.reason}")

            review_record = rejected_review_record(
                round_idx=round_idx,
                candidate=evaluator_candidate,
                decision=decision,
                previous_candidate=previous_rejected_candidate,
            )
            append_rejected_review_record(rejected_review_history, review_record)
            previous_rejected_candidate = evaluator_candidate
            
            if decision.reason == last_reason:
                rejection_count += 1
            else:
                rejection_count = 1
                
            repairable_fields = {finding.field for finding in actionable_findings}
            authorized_repair_fields.update(repairable_fields)
            previous_proposal = proposal
            findings_json = json.dumps(
                [finding.model_dump() for finding in actionable_findings], indent=2, default=str
            )
            feedback = (
                "Repair the existing configuration; do not generate a new alternative.\n"
                f"Evaluator findings:\n{findings_json}\n"
                f"You may change ONLY these top-level fields: {sorted(repairable_fields)}.\n"
                "Copy every other field exactly from your previous configuration. Add an llm_field_rationales entry for "
                "each changed field and explain the adjustment "
                "in the main rationale. Return the complete repaired configuration."
                " Evaluator recommended values are advisory text: do not apply null/removal or any value that violates the "
                "structured-output schema. Preserve schema-valid sentinel values for inactive fields."
            )
            
            if rejection_count >= 2:
                feedback = (
                    f"The same concrete violation has occurred repeatedly: '{decision.reason}'. "
                    "Apply the evaluator's recommended value when provided.\n" + feedback
                )
                
            optimizer_messages.append({
                "role": "user",
                "content": feedback
            })
            
            last_reason = decision.reason
            last_suggestions = decision.suggestions

    if generation_attempt >= max_generation_attempts:
        print("\n⚠️ Max generation attempts reached before a valid proposal was produced.")
        if decision is None:
            decision = HpoDecision(
                accept=False,
                reason=(
                    "Generation attempts were exhausted before a proposal passed all "
                    "schema and deterministic pre-evaluation checks."
                ),
                suggestions=["Inspect the generation diagnostics and retry."],
            )
    else:
        print("\n⚠️ Max evaluator rounds reached. The agents could not agree on a safe configuration.")
    if decision is not None:
        decision._authorized_repair_fields = set(authorized_repair_fields)
        decision._diagnostics = list(round_diagnostics)
    return proposal, decision
