import csv
from pathlib import Path

import pytest

from cvmodellearning.graphrag.inference_memory import (
    calculate_inference_memory,
    calculate_kv_cache_memory,
    estimate_cnn_activation_workspace,
)


ONTOLOGY_ESTIMATES = (
    Path(__file__).parents[1]
    / "ontology_data"
    / "nodes"
    / "model_inference_memory_estimates.csv"
)


def test_mobilenet_v3_small_fp16_estimate_matches_ontology_formula():
    activation_gb = estimate_cnn_activation_workspace(
        flops_b=0.06,
        task="classification",
    )
    estimate = calculate_inference_memory(
        params_m=2.543,
        precision_mode="FP16",
        activation_workspace_gb=activation_gb,
    )

    assert estimate.weight_memory_gb == 0.005
    assert estimate.activation_workspace_gb == 0.050
    assert estimate.runtime_overhead_gb == 0.001
    assert estimate.total_estimated_vram_gb == 0.056


def test_fp32_estimate_uses_four_bytes_per_parameter():
    estimate = calculate_inference_memory(
        params_m=2.543,
        precision_mode="FP32",
        activation_workspace_gb=0.100,
    )

    assert estimate.weight_memory_gb == 0.009
    assert estimate.runtime_overhead_gb == 0.002
    assert estimate.total_estimated_vram_gb == 0.111


def test_mobile_cnn_fp32_activation_policy_matches_current_executor_rows():
    assert estimate_cnn_activation_workspace(
        flops_b=0.06,
        task="classification",
        precision_mode="FP32",
    ) == 0.100


def test_cnn_activation_policy_is_explicitly_task_specific():
    assert estimate_cnn_activation_workspace(flops_b=0.06, task="classification") == 0.050
    assert estimate_cnn_activation_workspace(flops_b=80.6, task="detection") == 0.403


def test_kv_cache_formula_includes_keys_and_values():
    assert calculate_kv_cache_memory(
        batch_size=1,
        context_tokens=2048,
        layers=32,
        kv_heads=32,
        head_dim=128,
    ) == 1.0


def test_every_ontology_row_uses_deterministic_weight_overhead_and_total_formula():
    with ONTOLOGY_ESTIMATES.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows
    for row in rows:
        estimate = calculate_inference_memory(
            params_m=float(row["params_m"]),
            precision_mode=row["precision_mode"],
            activation_workspace_gb=float(row["activation_workspace_gb"]),
            kv_cache_gb=float(row["kv_cache_gb"]),
        )
        assert estimate.weight_memory_gb == float(row["weight_memory_gb"]), row["id"]
        assert estimate.runtime_overhead_gb == float(row["runtime_overhead_gb"]), row["id"]
        assert estimate.total_estimated_vram_gb == float(
            row["total_estimated_vram_gb"]
        ), row["id"]


@pytest.mark.parametrize(
    "arguments",
    (
        {"params_m": 0, "precision_mode": "FP16", "activation_workspace_gb": 0.1},
        {"params_m": 1, "precision_mode": "unknown", "activation_workspace_gb": 0.1},
        {"params_m": 1, "precision_mode": "FP16", "activation_workspace_gb": -0.1},
    ),
)
def test_invalid_memory_inputs_fail_before_materialization(arguments):
    with pytest.raises(ValueError):
        calculate_inference_memory(**arguments)
