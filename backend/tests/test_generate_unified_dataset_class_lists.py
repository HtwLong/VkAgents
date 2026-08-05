import pytest

from cvmodellearning.download import generate_unified_dataset_class_lists as generator


def test_build_class_list_query_uses_native_task_dataset_marker():
    classification = generator.build_class_list_query("classification")
    detection = generator.build_class_list_query("detection")

    assert "(^|_)(cls|classification)(_|$)" in classification
    assert "(^|_)(det|detection)(_|$)" in detection
    assert "SELECT DISTINCT ?labelName" in classification
    assert 'STRSTARTS(LCASE(STR(?labelName)), "/m/")' in detection


def test_build_class_list_query_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported VisionKG task"):
        generator.build_class_list_query("segmentation")


def test_fetch_classes_normalizes_deduplicates_and_sorts(monkeypatch):
    monkeypatch.setattr(generator, "query", lambda _: [
        {"labelName": " Zebra "},
        {"labelName": "apple"},
        {"labelName": "zebra"},
        {"labelName": "/m/012345"},
        {"other": "ignored"},
    ])

    assert generator.fetch_classes("classification") == ["apple", "zebra"]


def test_write_class_list_uses_expected_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "OUTPUT_DIRECTORY", tmp_path)

    output_path = generator.write_class_list("detection", ["apple", "zebra"])

    assert output_path == tmp_path / "unified_dataset_detection.txt"
    assert output_path.read_text(encoding="utf-8") == "apple\nzebra\n"
