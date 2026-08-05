"""Operational availability overrides for datasets.

The ontology describes what a dataset contains.  This module describes whether
the data can currently be materialized by this application.  Keeping the two
concerns separate makes temporary API/download outages easy to update without
changing ontology facts or LLM prompts.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path


DATASET_AVAILABILITY_FILE_ENV = "DATASET_AVAILABILITY_FILE"
DEFAULT_DATASET_AVAILABILITY_FILE = Path(__file__).with_name("dataset_availability.json")


class DatasetAvailabilityConfigError(ValueError):
    """Raised when the operational availability file is malformed."""


@dataclass(frozen=True)
class DatasetAvailability:
    dataset_id: str
    downloadable: bool
    reason: str = ""
    checked_at: str = ""


def availability_file() -> Path:
    override = os.getenv(DATASET_AVAILABILITY_FILE_ENV)
    return Path(override).expanduser() if override else DEFAULT_DATASET_AVAILABILITY_FILE


def load_dataset_availability() -> dict[str, DatasetAvailability]:
    """Read current overrides on every call so edits take effect immediately."""

    path = availability_file()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetAvailabilityConfigError(
            f"Could not read dataset availability configuration at {path}: {exc}"
        ) from exc

    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict):
        raise DatasetAvailabilityConfigError(
            f"Dataset availability configuration at {path} must contain a 'datasets' object."
        )

    result: dict[str, DatasetAvailability] = {}
    for dataset_id, value in datasets.items():
        if not isinstance(dataset_id, str) or not isinstance(value, dict):
            raise DatasetAvailabilityConfigError(
                f"Invalid dataset availability entry in {path}."
            )
        downloadable = value.get("downloadable")
        if not isinstance(downloadable, bool):
            raise DatasetAvailabilityConfigError(
                f"Entry '{dataset_id}' in {path} requires a boolean 'downloadable'."
            )
        result[dataset_id] = DatasetAvailability(
            dataset_id=dataset_id,
            downloadable=downloadable,
            reason=str(value.get("reason", "")),
            checked_at=str(value.get("checked_at", "")),
        )
    return result


def get_dataset_availability(dataset_id: str) -> DatasetAvailability:
    """Return an explicit override, or downloadable-by-default for new datasets."""

    return load_dataset_availability().get(
        dataset_id,
        DatasetAvailability(dataset_id=dataset_id, downloadable=True),
    )


def is_dataset_downloadable(dataset_id: str) -> bool:
    return get_dataset_availability(dataset_id).downloadable
