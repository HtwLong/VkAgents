# Adapt a retrieved CV training recipe

Treat the GraphRAG recipe as an evidence-backed starting point, not an immutable configuration.

1. Compare the recipe's source task, data scale/domain, model, image size, batch size, and training mode with the target context.
2. Preserve a reference value when its assumptions still fit. Adapt it when model, data, deployment, or training-hardware evidence provides a reason.
3. For classification, reason about pretrained preprocessing, head replacement, freezing depth, discriminative learning rates, imbalance handling, resolution, and regularization.
4. For detection, reason about resolution, effective batch size, pretrained fine-tuning, small-object needs, multi-scale/spatial augmentation, imbalance, and end-of-training augmentation behavior.
5. Choose all schema-configurable fields. Runtime-owned identifiers, class counts, paths, and structural invariants remain pipeline-owned.
6. Explain each active configurable field in `llm_field_rationales`, including whether the recipe value was preserved or adapted and why.
7. Do not claim an initial value is optimal. When evidence is weak, choose a safe executable starting point and state the uncertainty in the rationale.
8. Model selection is not a source of hyperparameters. Never copy epochs, patience, optimizer, loss, resolution, batch size, or augmentation values from `selected_model_info`; use it only to identify the selected architecture.
9. The HPO LLM may adapt GraphRAG defaults, but every changed recipe-backed field must explain the concrete basis in user requirements, dataset profile, training hardware, a matched adjustment rule, or a runtime constraint. Do not attribute an adapted value to GraphRAG.
10. Validate schedules as a system: epochs, patience, warmup, and late augmentation shutdown must leave a meaningful normal-training phase. For small-object detection, also explain resolution and spatial augmentation together.
11. Obey the supplied executable capabilities. Do not infer that weights, LoRA, or another training mode is implemented merely because it exists in general ML practice.
12. Distinguish hard executable constraints from recommendations. A recipe deviation or quality heuristic may proceed with an explicit evidence-based rationale; never present a recommendation as a runtime incompatibility.
