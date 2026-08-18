"""Editable entry point for the planning-reliability batch evaluation.

Change the values in ``BATCH SETTINGS`` and run this file from ``backend``:

    uv run python scripts/run_batch_evaluation.py
"""

from __future__ import annotations

import argparse
import asyncio

from cvmodellearning.benchmarks.batch_eval import run


# ---------------------------------------------------------------------------
# BATCH SETTINGS -- edit these values before starting an experiment.
# ---------------------------------------------------------------------------

# Repetitions of every selected prompt under each GraphRAG condition.
REPETITIONS = 10

# One of: "enabled", "disabled", or "both".
GRAPHRAG = "enabled"

# None runs all eight non-VQA prompts. To select individual prompts, use e.g.:
# CASE_IDS = ["ex-furniture-classification", "ex-handwritten-numbers"]
CASE_IDS: list[str] | None = None # ["traffic-participants", "traffic-lights-signs", "ex-furniture-classification"]

# None creates benchmark_results/<timestamp>. Set a stable directory when you
# want to resume an interrupted experiment.
OUTPUT_DIRECTORY: str | None = None

# True skips case/repetition/GraphRAG combinations already present in the
# runs.json stored in OUTPUT_DIRECTORY. Requires a stable output directory.
RESUME = False

# Run the real data preparation and one training batch after planning. The data
# downloader is cache-aware; any cache misses still use its normal download path.
RUN_TRAINING_SMOKE = True


def configured_arguments() -> argparse.Namespace:
    if REPETITIONS < 1:
        raise ValueError("REPETITIONS must be at least 1.")
    if GRAPHRAG not in {"enabled", "disabled", "both"}:
        raise ValueError('GRAPHRAG must be "enabled", "disabled", or "both".')
    if RESUME and not OUTPUT_DIRECTORY:
        raise ValueError("Set OUTPUT_DIRECTORY before enabling RESUME.")
    return argparse.Namespace(
        repetitions=REPETITIONS,
        graphrag=GRAPHRAG,
        case=CASE_IDS,
        output=OUTPUT_DIRECTORY,
        resume=RESUME,
        training_smoke=RUN_TRAINING_SMOKE,
    )


def main() -> None:
    output = asyncio.run(run(configured_arguments()))
    print(f"Reports written to {output}")


if __name__ == "__main__":
    main()
