import csv
import json

from cvmodellearning.download import download_data


def test_classification_download_uses_exact_matching_and_only_writes_successes(
    monkeypatch,
    tmp_path,
):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    captured = {}

    def fake_query(query_string):
        captured["query"] = query_string
        return [
            {
                "datasetName": "bdd_100k_det_train",
                "imageName": "good.jpg",
                "image": "image:good",
            },
            {
                "datasetName": "bdd_100k_det_train",
                "imageName": "missing.jpg",
                "image": "image:missing",
            },
        ]

    def fake_prepare(images, DATA_ROOT_PATH):
        assert len(images) == 1
        return {
            "successful": [images[0]],
            "failures": [],
        }

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(download_data, "prepare_data", fake_prepare)
    monkeypatch.setattr(download_data, "data_dir", lambda job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda job_id: artifacts_root / "download_report.json")

    report = download_data.download_visionkg_mixed_datasets_classification(
        "test-job",
        [{
            "class_name": "car",
            "sources": [{"dataset_name": "bdd_100k_det_train", "count": 1}],
        }],
    )

    assert 'LCASE(STR(?labelName)) = LCASE("car")' in captured["query"]
    assert 'LCASE(STR(?datasetName)) = LCASE("bdd_100k_det_train")' in captured["query"]
    assert "regex(?labelName" not in captured["query"]
    assert report["complete"] is True
    with (data_root / "image_labels.csv").open(newline="") as source:
        assert list(csv.DictReader(source)) == [{
            "image_filename": "bdd_100k_det_train/good.jpg",
            "labels": "car",
        }]
    persisted = json.loads((artifacts_root / "download_report.json").read_text())
    assert persisted["downloaded"] == 1


def test_classification_download_reports_shortfall(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    monkeypatch.setattr(download_data, "query", lambda query_string: [{
        "datasetName": "coco2017_det_train",
        "imageName": "unavailable.jpg",
        "image": "image:unavailable",
    }])
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {
            "successful": [],
            "failures": [{**images[0], "status_code": 500, "error": "HTTP 500"}],
        },
    )
    monkeypatch.setattr(download_data, "data_dir", lambda job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda job_id: artifacts_root / "download_report.json")

    report = download_data.download_visionkg_mixed_datasets_classification(
        "test-job",
        [{
            "class_name": "car",
            "sources": [{"dataset_name": "coco2017_det_train", "count": 2}],
        }],
    )

    assert report["complete"] is False
    assert report["sources"][0]["shortfall"] == 2
    assert not (data_root / "image_labels.csv").exists()


def test_classification_download_allocates_scarce_class_before_abundant_class(
    monkeypatch,
    tmp_path,
):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()

    def fake_query(query_string):
        names = ["1.jpg", "2.jpg"] if 'LCASE("rare")' in query_string else [
            "1.jpg", "2.jpg", "3.jpg", "4.jpg",
        ]
        return [
            {"datasetName": "coco2017_det_train", "imageName": name, "image": f"image:{name}"}
            for name in names
        ]

    monkeypatch.setattr(download_data, "query", fake_query)
    monkeypatch.setattr(
        download_data,
        "prepare_data",
        lambda images, DATA_ROOT_PATH: {"successful": images, "failures": []},
    )
    monkeypatch.setattr(download_data, "data_dir", lambda _job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda _job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda _job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda _job_id: artifacts_root / "download_report.json")

    report = download_data.download_visionkg_mixed_datasets_classification(
        "test-job",
        [
            {"class_name": "common", "sources": [{"dataset_name": "coco2017_det_train", "count": 2}]},
            {"class_name": "rare", "sources": [{"dataset_name": "coco2017_det_train", "count": 2}]},
        ],
    )

    assert report["complete"] is True
    with (data_root / "image_labels.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert {row["image_filename"] for row in rows if row["labels"] == "rare"} == {
        "coco2017_det_train/1.jpg",
        "coco2017_det_train/2.jpg",
    }
    assert len({row["image_filename"] for row in rows}) == 4


def test_classification_download_replaces_failed_assigned_candidate(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"
    data_root.mkdir()
    artifacts_root.mkdir()
    rows = [
        {"datasetName": "coco2017_det_train", "imageName": name, "image": f"image:{name}"}
        for name in ("bad.jpg", "good-1.jpg", "good-2.jpg")
    ]

    def fake_prepare(images, DATA_ROOT_PATH):
        successful = [image for image in images if image["file_name"] != "bad.jpg"]
        failures = [
            {**image, "status_code": 500, "error": "HTTP 500"}
            for image in images
            if image["file_name"] == "bad.jpg"
        ]
        return {"successful": successful, "failures": failures}

    monkeypatch.setattr(download_data, "query", lambda _query: rows)
    monkeypatch.setattr(download_data, "prepare_data", fake_prepare)
    monkeypatch.setattr(download_data, "data_dir", lambda _job_id: data_root)
    monkeypatch.setattr(download_data, "csv_labels_path", lambda _job_id: data_root / "image_labels.csv")
    monkeypatch.setattr(download_data, "dataset_manifest_path", lambda _job_id: data_root / "dataset_manifest.json")
    monkeypatch.setattr(download_data, "download_report_path", lambda _job_id: artifacts_root / "download_report.json")

    report = download_data.download_visionkg_mixed_datasets_classification(
        "test-job",
        [{"class_name": "car", "sources": [{"dataset_name": "coco2017_det_train", "count": 2}]}],
    )

    assert report["complete"] is True
    assert report["sources"][0]["downloaded"] == 2
    assert len(report["sources"][0]["failures"]) == 1
    with (data_root / "image_labels.csv").open(newline="") as source:
        assert {row["image_filename"] for row in csv.DictReader(source)} == {
            "coco2017_det_train/good-1.jpg",
            "coco2017_det_train/good-2.jpg",
        }
