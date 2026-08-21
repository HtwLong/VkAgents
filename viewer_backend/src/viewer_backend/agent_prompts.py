"""Planning-only agent instructions mirrored from the original backend."""

from __future__ import annotations

from .skills import load_cv_skill


READINESS_INSTRUCTIONS = (
    "You decide if the provided description is sufficient to BEGIN structured extraction for a CV training workflow.\n"
    "ACCEPT only if following question is answered:\n"
    "Does the user want to classify, detect, segment images/videos or create a visual question answering model? It **HAS** to be one of them.\n"
    "If rejecting, return a clear 'reason'. As 'suggestions' ask the user for the missing information and nothing else.\n"
    "Return only the structured decision."
)

INTERPRETATION_INSTRUCTIONS = (
    "Extract ONLY the requested information from the user prompt. Leave fields empty if the information does not exist. "
    "Turn classes into singular form; infer classes only when directly entailed. Extract performance requirements, "
    "available hardware, deployment constraints, robustness, and model requirements. Use requirement_strength=required "
    "for explicit directives and preferred for prefer/ideally/if possible. Normalize explicit LoRA requests to "
    "training_mode='lora'. Use only VeryLow, Low, Medium, MediumHigh, or High for performance categories. Treat reliable "
    "or good accuracy as a soft MediumHigh preference and reserve High for explicit high, critical, maximum, or best. "
    "Set target_is_hard only for an explicit mandatory numeric target. Represent objective strength with hard, soft, "
    "preference, or unspecified. Extract lighting, weather, object_scale, scene_density, motion_blur, occlusion, and "
    "viewpoint robustness. Set color_semantics, horizontal_flip_safe, and text_or_symbols_present only with supporting "
    "evidence. Keep inference footprint in deployment_constraints and add hard_limits only for mandatory numeric limits. "
    "Do not treat relaxed latency as relaxed memory. Classify laptops/desktops, including Apple Silicon Macs, as "
    "ConsumerCPU; reserve EdgeDevice for mobile, embedded, Raspberry Pi, and Jetson devices. If neither hardware category "
    "nor VRAM is known, use ConsumerCPU | EdgeDevice. If performance intent has no metric, select the most relevant metric."
)

PIPELINE_STATE_BLUEPRINT = """
### PIPELINE STATE STRUCTURE
Use task, application_domain, user_query, classes, performance_requirements,
deployment_constraints, available_hardware, training_hardware, available_data,
selected_model_info, dataset_profile, and the supplied GraphRAG context.
Model selection owns architecture choice only. Never emit epochs, patience,
optimizer, learning rate, precision, losses, LoRA settings, or augmentation
fields during model selection; the HPO stage owns training configuration.
""".strip()

COMPARISON_INSTRUCTIONS = (
    "When GraphRAG candidates are present, compare at least two candidates, or every candidate if fewer than two. "
    "Use exact candidate IDs, concrete advantages and risks, and mark unsupported claims as uncertain. Candidate order is "
    "not preference. A soft numeric target is not a ceiling. Do not equate benchmarks with different datasets, protocols, "
    "input sizes, or hardware. Overall detection mAP is not evidence of small-object performance. Inference-memory estimates "
    "are deployment facts, not training feasibility facts. Mention a fallback and what evidence would change the choice."
)

MODEL_SELECTION_INSTRUCTIONS = "\n\n".join((
    PIPELINE_STATE_BLUEPRINT,
    load_cv_skill("diagnose"),
    load_cv_skill("model-selection"),
    load_cv_skill("data-problems"),
    COMPARISON_INSTRUCTIONS,
    "Write the rationale in clear English using ASCII characters only.",
))

DATASET_SELECTION_INSTRUCTIONS = "\n\n".join((
    PIPELINE_STATE_BLUEPRINT,
    load_cv_skill("diagnose"),
    load_cv_skill("dataset-and-split"),
    load_cv_skill("data-problems"),
    "Use allowed candidates as authoritative. Select only exact dataset IDs valid for the exact class. Never invent a "
    "dataset or exceed reported availability. Prefer one coherent target-domain family; mix sources only for a concrete "
    "coverage or generalization benefit and explain compatibility and leakage risks. Counts are complete source-pool "
    "counts before deterministic splitting. Local code owns final train/validation/test assignments.",
))

HYPERPARAMETER_INSTRUCTIONS = "\n\n".join((
    PIPELINE_STATE_BLUEPRINT,
    load_cv_skill("diagnose"),
    load_cv_skill("recipe-adaptation"),
    load_cv_skill("data-problems"),
    "Use the retrieved recipe as a starting point and keep every value inside supplied hard bounds. Never change the "
    "selected model or pipeline-owned classes and dataset assignments. Return a safe initial planning proposal; this "
    "viewer service will not execute it.",
))


def model_selection_instructions(task: str) -> str:
    task_rules = {
        "classification": (
            "Choose one executable classification architecture. Respect latency and accuracy categories and balance "
            "pretrained capacity against available data."
        ),
        "detection": (
            "Choose one executable detector architecture. For requested small objects compare at least three candidates "
            "when available, cover at least two architecture types, and include a feasible two-stage candidate when retrieved."
        ),
        "visual question answering": (
            "Choose one executable VQA architecture and ground the rationale in questions_list when present. Architecture "
            "selection must not emit LoRA or other HPO values."
        ),
    }
    return f"{MODEL_SELECTION_INSTRUCTIONS}\n\n{task_rules.get(task, '')}"


def dataset_selection_instructions(task: str) -> str:
    task_rules = {
        "classification": "Use only native image-classification labels and keep class source pools reasonably balanced.",
        "detection": (
            "Use bounding-box compatible sources. Counts are positive images per class and may overlap across classes. "
            "Prefer a coherent shared dataset family across requested classes and explain domain compromises."
        ),
        "visual question answering": (
            "Select visually diverse compatible images that support rich downstream question-answer pairs."
        ),
    }
    return f"{DATASET_SELECTION_INSTRUCTIONS}\n\n{task_rules.get(task, '')}"
