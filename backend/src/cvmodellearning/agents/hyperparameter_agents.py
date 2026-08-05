import json
from typing import Union
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
from cvmodellearning.schemas.decision_schema import HpoDecision
from cvmodellearning.agents.agents_utils import log_planning_step
from cvmodellearning.graphrag.hyperparameter_context import (
    llm_controlled_fields,
    validate_executable_recipe_config,
    validate_detection_graph_grounded_config,
    validate_graph_grounded_config,
)
from cvmodellearning.models.classification_capabilities import (
    classification_prompt_constraints,
    selected_classification_model_id,
)
from cvmodellearning.models.registry import (
    DETECTION_HPO_MODEL_IDS,
    enabled_models,
    model_ids_equivalent,
)
from cvmodellearning.schemas.classification_hpo_completion import complete_classification_config
from cvmodellearning.schemas.dataset_assignment import planned_split_ratios
from cvmodellearning.policies.hyperparameter_policy_registry import (
    normalize_policy_rationales,
    policy_fields,
    policy_ids_by_field,
    validate_policy_rationales,
)


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


def _ensure_llm_rationales_on_dict(
    config: dict,
    missing_fields: set[str],
    policy_context: dict | None = None,
) -> dict:
    """Insert concise fallback llm_field_rationales for missing fields.

    This is a deterministic, auditable fallback used when the LLM omitted
    required field-level rationales. It appends brief reasons and updates the
    main rationale to mention the auto-generated entries.
    """
    if not missing_fields:
        return config
    rationales = list(config.get("llm_field_rationales") or [])
    existing = {r.get("field") for r in rationales}
    policy_ids = policy_ids_by_field(policy_context or {})
    added = []
    for f in sorted(missing_fields):
        if f in existing:
            continue
        applicable_ids = policy_ids.get(f, [])
        reason = f"The generated value for {f} was documented automatically"
        if applicable_ids:
            reason += " and checked against its applicable policy guidance"
        entry = {
            "field": f,
            "reason": f"{reason}.",
            "applied_policy_ids": applicable_ids,
        }
        rationales.append(entry)
        added.append(f)
    config["llm_field_rationales"] = rationales
    if added:
        prev = config.get("rationale", "")
        addition = " Added fallback rationales for: " + ", ".join(added) + "."
        config["rationale"] = (prev + addition).strip()
    return config


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


HPO_SCHEMA_FIELDS = frozenset().union(
    ClassificationConfigDraft.model_fields,
    DetectionConfigDraft.model_fields,
    VQAConfigModel.model_fields,
)


def evaluator_active_configuration(proposal: BaseModel, task: str) -> dict:
    """Expose only fields that can affect the selected runtime to the evaluator."""
    data = proposal.model_dump(mode="json")
    if task == "classification":
        active_fields = active_classification_config_fields(data)
    elif task == "detection":
        active_fields = active_detection_config_fields(data)
    else:
        active_fields = set(data)

    active_fields |= {"rationale", "llm_field_rationales"}
    return {field: value for field, value in data.items() if field in active_fields}


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
- `selected_data`: The authoritative train/validation/test dataset assignments from planning.
- `selected_model_info`: The architecture chosen in previous steps.
- `augmentation`, `preprocessing`: The data transformation strategies.
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


def changed_grounded_fields(
    proposal: BaseModel,
    base_configuration: dict,
    allowed_adjustment_fields: set[str],
) -> set[str]:
    """Return graph-grounded fields changed without an applicable rule."""
    proposed = proposal.model_dump(mode="json")
    return {
        field
        for field, grounded_value in base_configuration.items()
        if field not in allowed_adjustment_fields and proposed.get(field) != grounded_value
    }


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
        model_constraints = {"model_name": selected_model_name}

    # 3. Setup the initial Optimizer conversation history
    optimizer_messages = [
        {
            "role": "system",
            "content": (
                f"{PIPELINE_STATE_BLUEPRINT}\n\n"
                "You are a strict, deterministic Machine Learning configuration engine. "
                "Review the `selected_model_info`, `selected_data`, and `task` from the state. "
                "Based on this context, generate a safe hyperparameter configuration. "
                "The state may include `hyperparameter_graph_context.base_configuration`; copy every field in that "
                "deterministic baseline exactly unless the field appears in `allowed_adjustment_fields`. Apply only rules "
                "listed in `matched_adjustment_rules`; model-scoped rules whose conditions were not matched are informational. "
                "Fields listed in `fields_requiring_llm_completion` are deliberately missing from the recipe: choose safe "
                "values for them from the model, data, hardware, schema constraints, and retrieved evidence. Do not describe "
                "an LLM-completed value as recipe-sourced. Use only policies in "
                "`hyperparameter_policy_context.applicable_policies`; policies are advisory and never override recipe-grounded "
                "fields or runtime constraints, except that a recipe-backed Ultralytics optimizer='auto' may be changed to a "
                "supported explicit optimizer when the data, effective batch, or stability evidence specifically justifies it. "
                "When optimizer='auto', Ultralytics owns learning_rate and momentum, so do not claim to tune them. To control "
                "learning_rate, select an explicit optimizer and justify both optimizer_name and learning_rate. For each "
                "policy-guided field, add one `llm_field_rationales` entry whose "
                "`applied_policy_ids` cites the applicable policy IDs that influenced the value. "
                "Independently assess every active field in `fields_available_for_policy_guidance`; do not retain a schema "
                "default merely because the recipe omitted the field. Keep the default only when the applicable policy and "
                "training problem profile support it. "
                "For other fields absent from the baseline, use schema defaults unless the PipelineState provides a concrete reason. "
                "`training_recipe_id` is provenance, not a name you may invent: when `use_graphrag` is false it MUST be "
                "the empty string; when GraphRAG is enabled copy it exactly from the base configuration. "
                "You must rely ONLY on standard, universally accepted heuristics. Do not attempt creative, novel, or experimental configurations. "
                "If a parameter is standard, use the standard value. Hallucination or guessing outside of the provided context is strictly prohibited. "
                "Pay strict attention to memory constraints and learning rates for the selected architecture. "
                "Respect `performance_requirements.latency_category` and `performance_requirements.accuracy_category` "
                "when present: favor stronger training choices for MediumHigh or High accuracy requirements, "
                "efficient settings for VeryLow or Low latency requirements, and moderate defaults when both matter. "
                "Always fill the `rationale` field. Explicitly mention every LLM-completed or evaluator-adjusted field and why "
                "its value was selected."
                "\n\nOUTPUT FORMAT REQUIREMENTS: Return a single JSON object matching the executable schema for the task. "
                "The JSON MUST include a top-level llm_field_rationales array with one object per field that you set or completed. "
                "Each entry must be an object with a string `field` and a string `reason`. Do not include explanatory prose outside the JSON object. "
                "Example: {\"llm_field_rationales\": [{\"field\": \"patience\", \"reason\": \"Scaled to the small validation set and training duration\", \"applied_policy_ids\": [\"hpo.common.schedule_data_size.v1\"]}] }"
            )
        },
        {
            "role": "user",
            "content": (
                f"Task Context:\n{json_data}\n\n"
                "Mandatory executable model constraints:\n"
                f"{json.dumps(model_constraints, indent=2)}\n\n"
                "These constraints come from the registered runtime and must be obeyed. "
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
        "When a field-level rationale cites a policy, verify that the ID appears in "
        "`hyperparameter_policy_context.applicable_policies` and that its guidance applies to that field. "
        "Be ruthless but constructive. If it is safe and adheres to standard practices, accept it."
        " Return structured findings. Every hard_error or safety_warning MUST name exactly one top-level configuration `field`. "
        "Evaluate only fields present in Proposed Active Runtime Configuration. Do not infer candidate fields from "
        "GraphRAG metadata or the broader PipelineState. A blocking finding must name a field present in that active "
        "configuration. `available_hardware` describes deployment, while `training_hardware` determines training AMP support. "
        "Use `preference` for optional alternatives; preferences must not cause rejection. Cite a recipe/rule ID when available. "
        "Reject only for concrete compatibility, safety, constraint, or ontology-grounding problems—not because another valid value might perform better."
        " The configuration schema is deliberately union-free: optimizer- and scheduler-specific fields are always present. "
        "Inactive schema sentinel fields are filtered out before runtime. A sentinel is a validation placeholder, not a "
        "tuning decision. Do not flag an inactive field merely because it exists in the broader schema, "
        "and never recommend removing a required field or setting it to null. `alpha` belongs to RMSprop, `eps` belongs to "
        "AdamW/RMSprop, and `beta1`/`beta2` belong to AdamW. For StepLR or no scheduler, min_learning_rate=0 is the required "
        "inactive sentinel. `scheduler_step_size` and `scheduler_gamma` remain schema-valid but are ignored for cosine/no scheduler. "
        "Use preference, not safety_warning, for optional cleanup, uncited advisory policies, or alternative valid values. "
        "Runtime-supported but potentially inefficient or suboptimal values are preferences, not blocking findings. "
        "Deployment constraints are owned and validated by model selection. Report residual uncertainty for "
        "memory_category, max_runtime_memory_mb, max_model_size_mb, max_parameters_m, or max_cpu_latency_ms "
        "as a preference; do not reject an otherwise valid HPO configuration for those fields. "
        "total_estimated_vram_gb is the estimated runtime footprint comparable to max_runtime_memory_mb; "
        "practical_min_vram_gb is conservative provisioning guidance, not measured runtime consumption. "
        "Do not infer an input incompatibility from a classification dataset's native resolution or channel count when "
        "the supplied registered constraints allow the configured size; the runtime converts images to RGB and resizes them. "
        "If accept=true, all findings must be preferences; hard_error and safety_warning require accept=false."
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
    graph_context = context_data.get("hyperparameter_graph_context") or {}
    policy_context = context_data.get("hyperparameter_policy_context") or {}
    use_graphrag = bool(context_data.get("use_graphrag", True))
    use_policy_registry = bool(context_data.get("use_policy_registry", True))
    base_configuration = graph_context.get("base_configuration") or {}
    allowed_graph_adjustments = set(graph_context.get("allowed_adjustment_fields") or [])

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
            optimizer_response = await client.beta.chat.completions.parse(
                model="gpt-5-nano",
                messages=optimizer_messages,
                response_format=TargetSchema
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
            if previous_proposal is not None:
                proposal = merge_authorized_repair_fields(
                    previous_proposal,
                    proposal,
                    repairable_fields,
                )
            proposal_changes = (
                changed_configuration_fields(previous_proposal, proposal)
                if previous_proposal is not None
                else set()
            )

            if task in {"classification", "detection"} and (use_graphrag or use_policy_registry):
                proposal_data = proposal.model_dump(mode="json")
                active_fields = (
                    active_classification_config_fields(proposal_data)
                    if task == "classification"
                    else active_detection_config_fields(proposal_data)
                )
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
                if use_policy_registry and not use_graphrag:
                    required_explanations |= policy_fields(policy_context)
                required_explanations &= active_fields
                missing_explanations = required_explanations - explained_fields
                if missing_explanations:
                    # Attempt an automatic, deterministic fallback: add short
                    # llm_field_rationales entries and continue. This reduces
                    # brittle round trips when the LLM omits only explanatory
                    # text but otherwise produced a valid configuration.
                    try:
                        proposal_dict = proposal.model_dump()
                        proposal_dict = _ensure_llm_rationales_on_dict(
                            proposal_dict,
                            missing_explanations,
                            policy_context if use_policy_registry else None,
                        )
                        # Re-validate after inserting rationales.
                        proposal = type(proposal).model_validate(proposal_dict)
                        active_explanations = [
                            item
                            for item in getattr(proposal, "llm_field_rationales", [])
                            if item.field in active_fields
                        ]
                        # Record a diagnostic that we auto-filled explanations
                        diagnostic = {
                            "round": round_idx,
                            "phase": "optimizer_precheck",
                            "reason": "auto_filled_llm_field_rationales",
                            "fields": sorted(missing_explanations),
                        }
                        round_diagnostics.append(diagnostic)
                        print(f"ℹ️ Auto-filled missing llm_field_rationales for fields: {sorted(missing_explanations)}")
                    except ValidationError as exc:
                        # If inserting fallback rationales caused validation to fail
                        # (unexpected), fall back to the previous behavior and ask
                        # the optimizer LLM to provide the missing explanations.
                        validation_errors = format_pydantic_validation_errors(exc)
                        diagnostic = {
                            "round": round_idx,
                            "phase": "optimizer_precheck",
                            "reason": "missing_llm_field_rationales",
                            "fields": sorted(missing_explanations),
                            "errors": validation_errors,
                        }
                        round_diagnostics.append(diagnostic)
                        print(f"⚠️ Round {round_idx} skipped before evaluation: {json.dumps(diagnostic)}")
                        optimizer_messages.append({
                            "role": "user",
                            "content": (
                                "Your configuration is missing field-specific explanations for LLM decisions: "
                                f"{sorted(missing_explanations)}. Add one llm_field_rationales entry per field and "
                                "also mention each field and reason in the main rationale. Do not change configuration values."
                            ),
                        })
                        continue

                if use_policy_registry:
                    active_explanations = normalize_policy_rationales(
                        active_explanations,
                        policy_context,
                    )
                    proposal = proposal.model_copy(
                        update={"llm_field_rationales": active_explanations}
                    )

                policy_guided_fields = (
                    required_explanations & policy_fields(policy_context)
                    if use_policy_registry
                    else set()
                )
                policy_errors = validate_policy_rationales(
                    active_explanations,
                    policy_guided_fields,
                    policy_context,
                )
                if policy_errors:
                    diagnostic = {
                        "round": round_idx,
                        "phase": "optimizer_precheck",
                        "reason": "invalid_policy_citations",
                        "errors": policy_errors,
                    }
                    round_diagnostics.append(diagnostic)
                    optimizer_messages.append({
                        "role": "user",
                        "content": (
                            "Policy-guided decisions require valid policy citations: "
                            f"{policy_errors}. Add applicable IDs from "
                            "hyperparameter_policy_context.applicable_policies; do not change values."
                        ),
                    })
                    continue

            unauthorized_grounded_changes = (
                changed_grounded_fields(
                    proposal,
                    base_configuration,
                    allowed_graph_adjustments | authorized_repair_fields | repairable_fields,
                )
                if use_graphrag
                else set()
            )
            if unauthorized_grounded_changes:
                diagnostic = {
                    "round": round_idx,
                    "phase": "optimizer_precheck",
                    "reason": "unauthorized_graph_grounded_changes",
                    "fields": sorted(unauthorized_grounded_changes),
                }
                round_diagnostics.append(diagnostic)
                print(f"⚠️ Round {round_idx} skipped before evaluation: {json.dumps(diagnostic)}")
                optimizer_messages.append({
                    "role": "user",
                    "content": (
                        "Your proposal changed graph-grounded base fields without a matched adjustment rule: "
                        f"{sorted(unauthorized_grounded_changes)}. Copy their exact values from "
                        "hyperparameter_graph_context.base_configuration. You may deviate only for fields in "
                        f"allowed_adjustment_fields={sorted(allowed_graph_adjustments)} or fields explicitly "
                        f"authorized by the reviewer={sorted(repairable_fields)}."
                    ),
                })
                continue

            if task in {"classification", "detection"} and use_graphrag:
                try:
                    if task == "classification":
                        validate_graph_grounded_config(
                            proposal.model_dump(mode="json"),
                            graph_context,
                            additional_allowed_fields=authorized_repair_fields | repairable_fields,
                        )
                    else:
                        validate_detection_graph_grounded_config(
                            proposal.model_dump(mode="json"),
                            graph_context,
                            additional_allowed_fields=authorized_repair_fields | repairable_fields,
                        )
                except ValueError as exc:
                    diagnostic = {
                        "round": round_idx,
                        "phase": "ontology_validation",
                        "reason": "graph_grounded_config_invalid",
                        "message": str(exc),
                    }
                    round_diagnostics.append(diagnostic)
                    print(f"⚠️ Round {round_idx} skipped before evaluation: {json.dumps(diagnostic)}")
                    optimizer_messages.append({
                        "role": "user",
                        "content": (
                            "Your proposal failed deterministic ontology recipe validation. Repair only the field named "
                            f"by this error and preserve the graph baseline: {exc}"
                        ),
                    })
                    continue

            if task in {"classification", "detection"}:
                try:
                    validate_executable_recipe_config(
                        proposal.model_dump(mode="json")
                    )
                except ValueError as exc:
                    diagnostic = {
                        "round": round_idx,
                        "phase": "executable_validation",
                        "reason": "executable_config_invalid",
                        "message": str(exc),
                    }
                    round_diagnostics.append(diagnostic)
                    optimizer_messages.append({
                        "role": "user",
                        "content": (
                            "Your proposal failed deterministic execution validation: "
                            f"{exc}. Correct only the invalid field and preserve all "
                            "pipeline-owned values."
                        ),
                    })
                    continue

            if previous_proposal is not None:
                unauthorized_fields = proposal_changes - repairable_fields
                if unauthorized_fields:
                    diagnostic = {
                        "round": round_idx,
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
        evaluator_messages = [
            {"role": "system", "content": evaluator_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task Context: {json_data}\n\n"
                    f"{evaluator_runtime_guidance(task, model_constraints)}\n\n"
                    "Proposed Active Runtime Configuration:\n"
                    f"{json.dumps(evaluator_candidate, indent=2)}"
                ),
            },
        ]
        
        evaluator_response = await client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=evaluator_messages,
            response_format=HpoDecision
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
                "each changed field, cite any applicable policy IDs in applied_policy_ids, and explain the adjustment "
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
