import csv
import hashlib
import json
import re

import pytest

from cvmodellearning.download import download_data
from cvmodellearning.datasets.selection import build_dataset_assignments
from cvmodellearning.schemas.interpretation_schema import ClassDataSelection
from cvmodellearning.download.assignment_manifest import (
    DatasetContentConflict,
    DatasetManifest,
    assignment_fingerprint,
    detection_coverage_requirements,
    evaluate_detection_coverage_acceptance,
    iter_download_allocations,
    load_dataset_manifest,
    validate_content_isolation,
)
from cvmodellearning.pipelines import detection_pipe


def test_replaced_detection_transfer_failure_is_diagnostic_not_blocking():
    acceptance = evaluate_detection_coverage_acceptance(
        {
            "unique_image_shortfall": 0,
            "unique_coverage_ratio": 1.0,
            "class_split_coverage_satisfied": True,
        },
        unresolved_transfer_failures=[{"image_id": "failed-primary"}],
    )

    assert acceptance["accepted"] is True
    assert acceptance["unresolved_transfer_failures"] == []
    assert acceptance["successfully_replaced_transfer_failures"] == [
        {"image_id": "failed-primary"}
    ]


def test_llm_unique_image_objective_replaces_sum_of_multilabel_allocations():
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [
                    {"split": "train", "count": 800, "assignment_type": "official_split"},
                    {"split": "validation", "count": 100, "assignment_type": "derived_from_train"},
                    {"split": "test", "count": 100, "assignment_type": "derived_from_train"},
                ],
            }],
        }
        for class_name in ("car", "pedestrian", "bus")
    ]

    requirements = detection_coverage_requirements(requests, {
        "minimum_unique_images": 1_200,
        "preferred_unique_images": 1_500,
    })

    assert requirements["target_unique_images"] == 1_500
    assert requirements["minimum_unique_images"] == 1_200
    assert requirements["target_unique_images_by_split"] == {
        "train": 1_200,
        "validation": 150,
        "test": 150,
    }


def test_simple_detection_plan_uses_largest_class_pool_as_unique_minimum():
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [
                    {"split": "train", "count": 800, "assignment_type": "official_split"},
                    {"split": "validation", "count": 100, "assignment_type": "derived_from_train"},
                    {"split": "test", "count": 100, "assignment_type": "derived_from_train"},
                ],
            }],
        }
        for class_name in ("car", "pedestrian", "bus")
    ]

    requirements = detection_coverage_requirements(requests)

    assert requirements["target_unique_images"] == 3_000
    assert requirements["minimum_unique_images"] == 1_000


def test_simple_planner_assignments_feed_detection_download_contract():
    selected = [ClassDataSelection.model_validate({
        "class_name": class_name,
        "sources": [{"dataset_name": "bdd_100k_det_train", "count": 1_000}],
    }) for class_name in ("car", "pedestrian")]

    assignments = build_dataset_assignments(selected, selected)
    payload = [item.model_dump(mode="json") for item in assignments]
    allocations = list(iter_download_allocations(payload))
    requirements = detection_coverage_requirements(payload)

    assert {item.split for item in allocations} == {"train", "validation", "test"}
    assert all(
        item.assignment_type == "derived_from_train"
        for item in allocations
        if item.split in {"validation", "test"}
    )
    assert requirements["minimum_unique_images"] == 1_000
    assert requirements["minimum_images_per_class_by_split"] == {
        "car": {"train": 800, "validation": 100, "test": 100},
        "pedestrian": {"train": 800, "validation": 100, "test": 100},
    }


def _patch_paths(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    monkeypatch.setattr(download_data, "data_dir", lambda _job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda _job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "json_labels_path", lambda _job_id: data_root / "annotations.json")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda _job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda _job_id: artifacts_root / "download_report.json")
    # Most downloader tests mock the transfer boundary and therefore do not
    # create image files. Give each path distinct deterministic test content.
    monkeypatch.setattr(
        download_data,
        "file_sha256",
        lambda path: hashlib.sha256(str(path).encode()).hexdigest(),
    )
    return data_root, artifacts_root


def _assignments():
    return [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 1, "assignment_type": "official_split"},
                {"split": "validation", "count": 1, "assignment_type": "derived_from_train"},
                {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
            ],
        }],
    }]


def test_classification_download_reserves_disjoint_samples_per_assignment(monkeypatch, tmp_path):
    data_root, artifacts_root = _patch_paths(monkeypatch, tmp_path)
    rows = [
        {"datasetName": "bdd_100k_det_train", "imageName": f"{index}.jpg", "image": f"image:{index}"}
        for index in range(1, 4)
    ]
    monkeypatch.setattr(download_data, "query", lambda _query: rows)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )

    report = download_data.download_visionkg_mixed_datasets_classification(
        "job",
        _assignments(),
    )

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [sample["assigned_split"] for sample in manifest["samples"]] == [
        "train",
        "validation",
        "test",
    ]
    assert len({sample["sample_id"] for sample in manifest["samples"]}) == 3
    assert report["complete"] is True
    assert report["unique_downloaded"] == 3
    assert sum(item["cross_split_conflicts"] for item in report["sources"]) == 6
    with (data_root / "image_labels.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert json.loads((artifacts_root / "download_report.json").read_text()) == report


def test_classification_download_pages_past_large_prior_allocation(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [
        {"datasetName": "bdd_100k_det_train", "imageName": f"{index}.jpg", "image": f"image:{index}"}
        for index in range(8)
    ]

    def paged_query(query):
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    monkeypatch.setattr(download_data, "query", paged_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 5, "assignment_type": "official_split"},
                {"split": "test", "count": 2, "assignment_type": "derived_from_train"},
            ],
        }],
    }]

    report = download_data.download_visionkg_mixed_datasets_classification("job", requests)

    assert report["complete"] is True
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [sample["assigned_split"] for sample in manifest["samples"]] == [
        "train", "train", "train", "train", "train", "test", "test",
    ]


def _detection_row(image_index, label="car"):
    return {
        "datasetName": "bdd_100k_det_train",
        "imageName": f"{image_index}.jpg",
        "image": f"image:{image_index}",
        "imageWidth": "100",
        "imageHeight": "80",
        "labelName": label,
        "bbHeight": "20",
        "bbWidth": "30",
        "bbCentreX": "50",
        "bbCentreY": "40",
    }


def test_detection_download_keeps_images_atomic_across_splits(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [_detection_row(index) for index in range(1, 4)]
    monkeypatch.setattr(download_data, "query", lambda _query: rows)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )

    report = download_data.download_visionkg_mixed_datasets_detection("job", _assignments())

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    coco = json.loads((data_root / "annotations.json").read_text())
    assert {sample["assigned_split"] for sample in manifest["samples"]} == {
        "train",
        "validation",
        "test",
    }
    assert len(manifest["samples"]) == len(coco["images"]) == 3
    assert {image["assigned_split"] for image in coco["images"]} == {
        "train",
        "validation",
        "test",
    }
    assert report["complete"] is True
    assert report["unique_downloaded"] == 3


def test_detection_download_replaces_different_ids_with_identical_content(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [_detection_row(index, "nightstand") for index in range(1, 4)]

    def paged_query(query):
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    def materialize(images, _job_id, _progress_callback, _cancel_check):
        for image in images:
            path = data_root / image["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            # IDs 1 and 2 reproduce the Objects365 failure: distinct records
            # and names, but exactly the same downloaded bytes.
            content = b"same photograph" if path.name in {"1.jpg", "2.jpg"} else b"replacement"
            path.write_bytes(content)
        return {"successful": images, "failures": []}

    monkeypatch.setattr(download_data, "query", paged_query)
    monkeypatch.setattr(download_data, "_prepare_with_progress", materialize)
    monkeypatch.setattr(download_data, "file_sha256", lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    requests = [{
        "class_name": "nightstand",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 1, "assignment_type": "official_split"},
                {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
            ],
        }],
    }]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    coco = json.loads((data_root / "annotations.json").read_text())
    assert [(sample["image_path"], sample["assigned_split"]) for sample in manifest["samples"]] == [
        ("bdd_100k_det_train/1.jpg", "train"),
        ("bdd_100k_det_train/3.jpg", "test"),
    ]
    assert {image["image_path"] for image in coco["images"]} == {
        "bdd_100k_det_train/1.jpg",
        "bdd_100k_det_train/3.jpg",
    }
    test_source = report["sources"][1]
    assert test_source["content_duplicate_conflicts"] == 1
    assert test_source["content_duplicates"][0]["image_path"].endswith("2.jpg")
    assert report["complete"] is True


def test_detection_download_reports_shortfall_when_no_unique_replacement_exists(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    rows = [_detection_row(index, "nightstand") for index in range(1, 3)]

    def paged_query(query):
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    def materialize(images, _job_id, _progress_callback, _cancel_check):
        for image in images:
            path = data_root / image["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"same photograph")
        return {"successful": images, "failures": []}

    monkeypatch.setattr(download_data, "query", paged_query)
    monkeypatch.setattr(download_data, "_prepare_with_progress", materialize)
    monkeypatch.setattr(
        download_data,
        "file_sha256",
        lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    requests = [{
        "class_name": "nightstand",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [
                {"split": "train", "count": 1, "assignment_type": "official_split"},
                {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
            ],
        }],
    }]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [(sample["image_path"], sample["assigned_split"]) for sample in manifest["samples"]] == [
        ("bdd_100k_det_train/1.jpg", "train"),
    ]
    assert report["sources"][1]["content_duplicate_conflicts"] == 1
    assert report["coverage"]["unique_image_shortfall"] == 1
    assert report["complete"] is False


def test_detection_download_caps_large_sparql_image_pages(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    page_limits = []

    def empty_query(query):
        page_limits.append(int(re.search(r"LIMIT (\d+)", query).group(1)))
        return []

    monkeypatch.setattr(download_data, "query", empty_query)
    requests = [{
        "class_name": "traffic light",
        "sources": [{
            "dataset_name": "bdd_100k_det_train",
            "allocations": [{
                "split": "train",
                "count": 2000,
                "assignment_type": "official_split",
            }],
        }],
    }]

    download_data.download_visionkg_mixed_datasets_detection("job", requests)

    assert page_limits == [download_data.DETECTION_SPARQL_PAGE_SIZE]
    assert page_limits[0] < 2000


def test_detection_download_reserves_coco_lvis_aliases_by_source_image(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def row(dataset, image_index):
        value = _detection_row(image_index)
        value.update({
            "datasetName": dataset,
            "imageName": f"{image_index:012d}.jpg",
            "image": f"http://vision.semkg.org/dataset/{dataset}/{image_index}",
        })
        return value

    coco_rows = [row("coco2017_det_train", 1)]
    lvis_rows = [
        row("LVIS_det_train", 1),  # Same COCO photograph: must remain in train.
        row("LVIS_det_train", 2),  # Replacement candidate for validation.
    ]

    def fake_query(query):
        rows = lvis_rows if 'VALUES ?datasetName { "LVIS_det_train" }' in query else coco_rows
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [{
        "class_name": "car",
        "sources": [
            {
                "dataset_name": "coco2017_det_train",
                "allocations": [{
                    "split": "train", "count": 1, "assignment_type": "official_split",
                }],
            },
            {
                "dataset_name": "LVIS_det_train",
                "allocations": [{
                    "split": "validation", "count": 1, "assignment_type": "derived_from_train",
                }],
            },
        ],
    }]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    assert [(sample["image_path"], sample["assigned_split"]) for sample in manifest["samples"]] == [
        ("coco2017_det_train/000000000001.jpg", "train"),
        ("LVIS_det_train/000000000002.jpg", "validation"),
    ]
    assert report["complete"] is True
    assert report["sources"][1]["cross_split_conflicts"] == 1


def test_detection_same_split_reuse_merges_classes_without_redownload(monkeypatch, tmp_path):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        download_data,
        "query",
        lambda query: [_detection_row(
            1,
            "truck" if 'VALUES ?candidateLabel { "truck" }' in query else "car",
        )],
    )
    download_calls = []

    def fake_prepare(images, DATA_ROOT_PATH):
        download_calls.append(list(images))
        return {"successful": images, "failures": []}

    monkeypatch.setattr(download_data, "prepare_data", fake_prepare)
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)

    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    coco = json.loads((data_root / "annotations.json").read_text())
    assert len(download_calls) == 1
    assert manifest["samples"][0]["class_names"] == ["car", "truck"]
    assert len(coco["images"]) == 1
    assert len(coco["categories"]) == 2
    assert len(coco["annotations"]) == 2
    assert report["downloaded"] == 2
    assert report["unique_downloaded"] == 1


def test_detection_download_tops_up_unique_pool_after_multilabel_overlap(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        label = "truck" if 'VALUES ?candidateLabel { "truck" }' in query else "car"
        rows = [_detection_row(1, label), _detection_row(2, label)]
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        return rows[offset:offset + limit]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train", "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())

    assert len(manifest["samples"]) == 2
    assert report["coverage"]["verified_unique_images"] == 2
    assert report["coverage"]["verified_images_per_class"]["car"] >= 1
    assert report["coverage"]["verified_images_per_class"]["truck"] >= 1
    assert report["coverage"]["satisfied"] is True
    assert report["complete"] is True


def test_detection_unique_top_up_searches_all_labels_and_limits_imbalance(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        if 'VALUES ?candidateLabel { "person" }' in query:
            return [_detection_row(1, "person")]
        if 'VALUES ?candidateLabel { "dog" }' in query:
            return [_detection_row(1, "dog")]
        if 'VALUES ?candidateLabel { "cat" }' in query:
            return [_detection_row(1, "cat")]
        assert 'VALUES ?candidateLabel { "cat" "dog" "person" }' in query
        # The global pass can use non-cat images after the cat-only pool is
        # exhausted, and should add the least represented labels.
        return [
            _detection_row(1, "cat"),
            _detection_row(2, "person"),
            _detection_row(3, "dog"),
            _detection_row(4, "cat"),
        ]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train", "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("person", "dog", "cat")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())

    assert len(manifest["samples"]) == 3
    assert report["coverage"]["satisfied"] is True
    counts = report["coverage"]["verified_images_per_class"]
    assert counts == {"person": 2, "dog": 2, "cat": 1}
    assert max(counts.values()) - min(counts.values()) == 1


def test_detection_accepts_aspirational_unique_target_after_exhaustive_top_up(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        offset = int(re.search(r"OFFSET (\d+)", query).group(1))
        if 'VALUES ?candidateLabel { "car" }' in query:
            rows = [_detection_row(index, "car") for index in range(1, 6)]
        elif 'VALUES ?candidateLabel { "truck" }' in query:
            rows = [_detection_row(index, "truck") for index in range(5, 10)]
        else:
            assert 'VALUES ?candidateLabel { "car" "truck" }' in query
            rows = [
                _detection_row(index, "car" if index < 5 else "truck")
                for index in range(1, 10)
            ]
        return rows[offset:offset + limit]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 5,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())

    assert len(manifest["samples"]) == 9
    assert report["coverage"]["unique_coverage_ratio"] == pytest.approx(0.9)
    assert report["coverage"]["class_split_coverage_satisfied"] is True
    assert report["coverage"]["satisfied"] is False
    assert report["acceptance"]["aspirational_unique_target_accepted"] is True
    assert report["complete"] is True
    warning = next(
        item
        for item in report["warnings"]
        if item["type"] == "aspirational_unique_image_shortfall"
    )
    assert warning["target_unique_images"] == 10
    assert warning["verified_unique_images"] == 9
    assert warning["adjusted_split_sizes"] == {
        "train": 9,
        "validation": 0,
        "test": 0,
    }


def test_detection_coverage_can_substitute_for_empty_direct_class_query(
    monkeypatch, tmp_path
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        if 'VALUES ?candidateLabel { "person" }' in query:
            return []
        if 'VALUES ?candidateLabel { "dog" }' in query:
            return [_detection_row(1, "person"), _detection_row(1, "dog")]
        if 'VALUES ?candidateLabel { "cat" }' in query:
            return [_detection_row(2, "cat")]
        assert 'VALUES ?candidateLabel { "cat" "dog" "person" }' in query
        return [_detection_row(3, "dog")]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train", "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("person", "dog", "cat")
    ]

    report = download_data.download_visionkg_mixed_datasets_detection("job", requests)
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())

    assert len(manifest["samples"]) == 3
    assert report["coverage"]["satisfied"] is True
    assert report["complete"] is True
    person_source = report["sources"][0]
    assert person_source["direct_allocation_downloaded"] == 0
    assert person_source["shortfall"] == 1
    assert report["allocation_shortfalls"][0]["class_name"] == "person"
    assert report["warnings"][0]["type"] == "allocation_substitution"


def test_detection_download_keeps_all_requested_class_boxes_on_selected_image(
    monkeypatch,
    tmp_path,
):
    data_root, _ = _patch_paths(monkeypatch, tmp_path)

    def fake_query(query):
        assert 'VALUES ?labelName { "car" "truck" }' in query
        if 'VALUES ?candidateLabel { "car" }' in query:
            return [_detection_row(1, "car"), _detection_row(1, "truck")]
        return [_detection_row(2, "truck")]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    requests = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [{
                    "split": "train",
                    "count": 1,
                    "assignment_type": "official_split",
                }],
            }],
        }
        for class_name in ("car", "truck")
    ]

    download_data.download_visionkg_mixed_datasets_detection("job", requests)

    coco = json.loads((data_root / "annotations.json").read_text())
    image_one_id = next(
        image["id"] for image in coco["images"] if image["image_path"].endswith("1.jpg")
    )
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    assert {
        category_names[annotation["category_id"]]
        for annotation in coco["annotations"]
        if annotation["image_id"] == image_one_id
    } == {"car", "truck"}


def test_manifest_rejects_cross_split_reassignment():
    manifest = DatasetManifest("job", "detection", "fingerprint")
    values = {
        "sample_id": "image:1",
        "image_path": "dataset/1.jpg",
        "class_name": "car",
        "dataset_name": "dataset",
        "source_role": "train",
        "assignment_type": "derived_from_train",
    }
    manifest.add(assigned_split="validation", **values)

    with pytest.raises(ValueError, match="both"):
        manifest.add(assigned_split="test", **values)


def test_manifest_loader_rejects_invalid_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "task": "classification",
        "assignment_fingerprint": "fingerprint",
        "samples": [{
            "sample_id": "sample",
            "image_path": "dataset/image.jpg",
            "class_names": ["car"],
            "dataset_name": "bdd_100k_det_val",
            "source_role": "validation",
            "assigned_split": "train",
            "assignment_type": "official_split",
        }],
    }))

    with pytest.raises(ValueError, match="remain in their source split"):
        load_dataset_manifest(
            path,
            task="classification",
            expected_fingerprint="fingerprint",
        )


def test_downloader_contract_rejects_invalid_derived_assignment():
    requests = [{
        "class_name": "car",
        "sources": [{
            "dataset_name": "bdd_100k_det_val",
            "allocations": [{
                "split": "test",
                "count": 1,
                "assignment_type": "derived_from_train",
            }],
        }],
    }]

    with pytest.raises(ValueError, match="official training source"):
        list(iter_download_allocations(requests))


def test_detection_pipeline_rejects_required_allocation_shortfall(monkeypatch, tmp_path):
    monkeypatch.setattr(detection_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(
        detection_pipe,
        "download_visionkg_mixed_datasets_detection",
        lambda _job_id, _selected_data: {
            "complete": False,
            "sources": [{
                "class_name": "car",
                "dataset_name": "bdd_100k_det_train",
                "assigned_split": "test",
                "requested": 10,
                "downloaded": 8,
                "shortfall": 2,
            }],
        },
    )

    with pytest.raises(RuntimeError, match="test: 8/10"):
        detection_pipe.DetectionPipeline().download_data_step(
            {"selected_data": _assignments()},
            "job",
        )


def test_detection_preparation_preserves_manifest_assignments(monkeypatch, tmp_path):
    selected_data = _assignments()
    images = []
    annotations = []
    samples = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        image_path = f"bdd/{index}.jpg"
        (tmp_path / "bdd").mkdir(exist_ok=True)
        (tmp_path / image_path).write_bytes(f"image-{index}".encode())
        images.append({"id": f"image-{index}", "image_path": image_path})
        annotations.append({
            "id": f"annotation-{index}",
            "image_id": f"image-{index}",
            "category_id": 1,
            "bbox": [1, 1, 10, 10],
        })
        samples.append({
            "sample_id": f"image-{index}",
            "image_path": image_path,
            "class_names": ["car"],
            "dataset_name": "bdd_100k_det_train",
            "source_role": "train",
            "assigned_split": split,
            "assignment_type": "official_split" if split == "train" else "derived_from_train",
        })
    annotations_path = tmp_path / "annotations.json"
    manifest_path = tmp_path / "manifest.json"
    annotations_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car"}],
    }))
    manifest_path.write_text(json.dumps({
        "task": "detection",
        "assignment_fingerprint": assignment_fingerprint(selected_data),
        "samples": samples,
    }))
    monkeypatch.setattr(detection_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(detection_pipe, "json_labels_path", lambda _job_id: annotations_path)
    monkeypatch.setattr(detection_pipe, "dataset_manifest_path", lambda _job_id: manifest_path)
    monkeypatch.setattr(detection_pipe, "preparation_summary_path", lambda _job_id: tmp_path / "summary.json")
    monkeypatch.setattr(detection_pipe, "train_json_path", lambda _job_id: tmp_path / "train.json")
    monkeypatch.setattr(detection_pipe, "val_json_path", lambda _job_id: tmp_path / "validation.json")
    monkeypatch.setattr(detection_pipe, "test_json_path", lambda _job_id: tmp_path / "test.json")
    monkeypatch.setattr(detection_pipe, "_get_trainer_type", lambda _model_name: "torchvision")

    result = detection_pipe.DetectionPipeline().prepare_data_step(
        {"selected_data": selected_data, "classes": ["car"], "model_name": "detector"},
        "job",
    )

    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["unique_images"] == result["counts"]
    assert result["class_image_counts"] == {
        "car": {"train": 1, "validation": 1, "test": 1},
    }
    assert result["content_isolation"] == {
        "unique_content_hashes": 3,
        "duplicate_samples": 0,
        "cross_split_duplicates": 0,
    }
    for split in ("train", "validation", "test"):
        prepared = json.loads((tmp_path / f"{split}.json").read_text())
        assert [image["assigned_split"] for image in prepared["images"]] == [split]


def test_detection_preparation_accepts_aspirational_multilabel_pool(
    monkeypatch, tmp_path
):
    selected_data = [
        {
            "class_name": class_name,
            "sources": [{
                "dataset_name": "bdd_100k_det_train",
                "allocations": [
                    {"split": "train", "count": 4, "assignment_type": "official_split"},
                    {"split": "validation", "count": 1, "assignment_type": "derived_from_train"},
                    {"split": "test", "count": 1, "assignment_type": "derived_from_train"},
                ],
            }],
        }
        for class_name in ("car", "truck")
    ]
    assignments = [
        ("train", "car"), ("train", "car"), ("train", "car"),
        ("train", "truck"), ("train", "truck"), ("train", "truck"),
        ("train", "car+truck"),
        ("validation", "car"), ("validation", "truck"),
        ("test", "car"), ("test", "truck"),
    ]
    images = []
    annotations = []
    samples = []
    annotation_id = 0
    category_ids = {"car": 1, "truck": 2}
    for image_id, (split, labels_text) in enumerate(assignments, start=1):
        labels = labels_text.split("+")
        image_path = f"bdd/{image_id}.jpg"
        (tmp_path / "bdd").mkdir(exist_ok=True)
        (tmp_path / image_path).write_bytes(f"image-{image_id}".encode())
        images.append({"id": f"image-{image_id}", "image_path": image_path})
        for label in labels:
            annotation_id += 1
            annotations.append({
                "id": annotation_id,
                "image_id": f"image-{image_id}",
                "category_id": category_ids[label],
                "bbox": [1, 1, 10, 10],
            })
        samples.append({
            "sample_id": f"image-{image_id}",
            "image_path": image_path,
            "class_names": labels,
            "dataset_name": "bdd_100k_det_train",
            "source_role": "train",
            "assigned_split": split,
            "assignment_type": "official_split" if split == "train" else "derived_from_train",
        })

    annotations_path = tmp_path / "annotations.json"
    manifest_path = tmp_path / "manifest.json"
    annotations_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "truck"},
        ],
    }))
    manifest_path.write_text(json.dumps({
        "task": "detection",
        "assignment_fingerprint": assignment_fingerprint(selected_data),
        "samples": samples,
    }))
    monkeypatch.setattr(detection_pipe, "data_dir", lambda _job_id: tmp_path)
    monkeypatch.setattr(detection_pipe, "json_labels_path", lambda _job_id: annotations_path)
    monkeypatch.setattr(detection_pipe, "dataset_manifest_path", lambda _job_id: manifest_path)
    monkeypatch.setattr(detection_pipe, "preparation_summary_path", lambda _job_id: tmp_path / "summary.json")
    monkeypatch.setattr(detection_pipe, "train_json_path", lambda _job_id: tmp_path / "train.json")
    monkeypatch.setattr(detection_pipe, "val_json_path", lambda _job_id: tmp_path / "validation.json")
    monkeypatch.setattr(detection_pipe, "test_json_path", lambda _job_id: tmp_path / "test.json")
    monkeypatch.setattr(detection_pipe, "_get_trainer_type", lambda _model_name: "torchvision")

    result = detection_pipe.DetectionPipeline().prepare_data_step(
        {
            "selected_data": selected_data,
            "classes": ["car", "truck"],
            "model_name": "detector",
        },
        "job",
    )

    assert result["coverage"]["unique_coverage_ratio"] == pytest.approx(11 / 12)
    assert result["coverage"]["class_split_coverage_satisfied"] is True
    assert result["acceptance"]["aspirational_unique_target_accepted"] is True
    assert result["acceptance"]["accepted"] is True
    assert result["warnings"][0]["adjusted_split_sizes"] == {
        "train": 7,
        "validation": 2,
        "test": 2,
    }


def test_content_isolation_rejects_exact_duplicate_across_splits(tmp_path):
    (tmp_path / "train.jpg").write_bytes(b"same pixels")
    (tmp_path / "validation.jpg").write_bytes(b"same pixels")
    manifest = {
        "samples": [
            {"image_path": "train.jpg", "assigned_split": "train"},
            {"image_path": "validation.jpg", "assigned_split": "validation"},
        ]
    }

    with pytest.raises(DatasetContentConflict, match="Identical image content") as error:
        validate_content_isolation(manifest, tmp_path)

    assert error.value.conflicts == [{
        "first_path": "train.jpg",
        "first_split": "train",
        "duplicate_path": "validation.jpg",
        "duplicate_split": "validation",
    }]


def test_content_isolation_allows_exact_duplicate_within_one_split(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"same pixels")
    (tmp_path / "two.jpg").write_bytes(b"same pixels")
    manifest = {
        "samples": [
            {"image_path": "one.jpg", "assigned_split": "train"},
            {"image_path": "two.jpg", "assigned_split": "train"},
        ]
    }

    assert validate_content_isolation(manifest, tmp_path) == {
        "unique_content_hashes": 1,
        "duplicate_samples": 1,
        "cross_split_duplicates": 0,
    }
