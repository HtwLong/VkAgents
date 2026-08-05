"""Generate task-specific VisionKG class vocabularies.

Run from the backend directory with:

    uv run python -m cvmodellearning.download.generate_unified_dataset_class_lists
"""

from pathlib import Path

from cvmodellearning.download.visionkg_utils import query


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
TASK_MARKERS = {
    "classification": ("cls", "classification"),
    "detection": ("det", "detection"),
}


def build_class_list_query(task: str) -> str:
    """Build a label query for VisionKG datasets named for one native task."""
    try:
        markers = TASK_MARKERS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported VisionKG task: {task}") from exc

    marker_pattern = "|".join(markers)
    return f"""
    PREFIX cv: <http://vision.semkg.org/onto/v0.1/>
    PREFIX schema: <http://schema.org/>

    SELECT DISTINCT ?labelName
    WHERE {{
        ?image schema:isPartOf / schema:name ?datasetName .
        FILTER(REGEX(LCASE(STR(?datasetName)), "(^|_)({marker_pattern})(_|$)"))
        ?image cv:hasAnnotation / cv:hasLabel / cv:label ?labelName .
        FILTER(!STRSTARTS(LCASE(STR(?labelName)), "/m/"))
    }}
    ORDER BY LCASE(STR(?labelName))
    """.strip()


def fetch_classes(task: str) -> list[str]:
    """Return normalized, sorted labels for all VisionKG datasets of a task."""
    rows = query(build_class_list_query(task))
    return sorted({
        row["labelName"].strip().lower()
        for row in rows
        if row.get("labelName", "").strip()
        and not row["labelName"].strip().lower().startswith("/m/")
    })


def write_class_list(task: str, labels: list[str]) -> Path:
    """Atomically write one task vocabulary beside unified_dataset.txt."""
    output_path = OUTPUT_DIRECTORY / f"unified_dataset_{task}.txt"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text("".join(f"{label}\n" for label in labels), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    for task in TASK_MARKERS:
        labels = fetch_classes(task)
        output_path = write_class_list(task, labels)
        print(f"Wrote {len(labels)} {task} classes to {output_path}")


if __name__ == "__main__":
    main()
