from viewer_backend.routers.planning import _merge_interpretation, _normalize_interpretation


def test_interpretation_normalizes_classes_constraints_and_robustness():
    result = _normalize_interpretation({
        "task": "detection",
        "classes": [" Person ", "person", "DOG"],
        "performance_requirements": {"primary_metric": "mAP", "target_value": 0.4},
        "robustness_requirements": {},
    }, "Detect small objects at night in rain when partially occluded")
    assert result["classes"] == ["person", "dog"]
    assert result["performance_requirements"]["accuracy_category"] == "MediumHigh"
    assert result["robustness_requirements"]["object_scale"] == ["small"]
    assert result["robustness_requirements"]["lighting"] == ["night"]
    assert result["robustness_requirements"]["weather"] == ["rain"]
    assert result["robustness_requirements"]["occlusion"] is True
    assert result["constraint_strengths"]["accuracy"] == "soft"
    assert result["available_hardware"]["hardware_category"] == "ConsumerCPU | EdgeDevice"


def test_interpretation_merge_preserves_existing_nested_values():
    result = _merge_interpretation(
        {"performance_requirements": {"primary_metric": "mAP", "target_value": 0.4}},
        {"performance_requirements": {"primary_metric": None, "latency_category": "Low"}},
    )
    assert result["performance_requirements"] == {
        "primary_metric": "mAP", "target_value": 0.4, "latency_category": "Low",
    }
