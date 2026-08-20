from viewer_backend.dataset_planning import build_split_assignments, dataset_role, preprocessing_plan


def test_dataset_roles_and_split_planning_use_only_reported_counts():
    context = {
        "task": "detection",
        "classes": ["person"],
        "available_data": [{"class_name": "person", "sources": [
            {"dataset_name": "coco2017_det_train", "count": 1000},
            {"dataset_name": "coco2017_det_val", "count": 80},
        ]}],
    }
    assignments, profile = build_split_assignments(context, ["coco2017_det_train"])
    allocations = [allocation for source in assignments[0]["sources"] for allocation in source["allocations"]]
    assert {item["split"] for item in allocations} == {"train", "validation", "test"}
    assert sum(item["count"] for item in allocations) <= 1080
    assert profile["verified_unique_images"] is None
    assert profile["derived_counts"]["test"] > 0
    assert dataset_role("imageNet-1K_cls_val") == "validation"


def test_preprocessing_is_robustness_aware_but_not_executed():
    plan = preprocessing_plan({
        "task": "detection",
        "robustness_requirements": {"lighting": ["night"], "object_scale": ["small"]},
    })
    assert "conservative_brightness_contrast" in plan["augmentations"]
    assert "scale_aware_crop_with_box_retention" in plan["augmentations"]
    assert plan["materialization_status"] == "planned_not_executed"
