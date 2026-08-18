from __future__ import annotations

import argparse
import asyncio
import csv
import copy
import json
import statistics
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from cvmodellearning.benchmarks.cases import CASES, BenchmarkCase
from cvmodellearning.paths import download_report_path, hpo_config_path, planning_usage_path
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline
from routers import execution, planning


STAGES = (
    ("task_interpretation", planning.task_interpret),
    ("data_check", planning.check_data),
    ("model_selection", planning.select_model),
    ("dataset_selection", planning.select_datasets),
    ("hyperparameters", planning.choose_hyperparameters),
)


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str).lower()


def _error_detail(exc: BaseException) -> Any:
    return exc.detail if isinstance(exc, HTTPException) else str(exc)


def classify_failure(stage: str, exc: BaseException) -> str:
    detail = _json_text(_error_detail(exc))
    if "rate limit" in detail or "429" in detail:
        return "external_rate_limit"
    if any(term in detail for term in ("out of memory", "cuda oom", "mps backend out of memory")):
        return "resource_limit"
    if any(term in detail for term in ("timeout", "connection", "sparql", "network")):
        return "external_service_error"
    if (
        "class ontology mapping" in detail
        or "invalid_classes" in detail
        or ("class" in detail and "not found" in detail)
    ):
        return "class_ontology_mapping_error"
    if stage == "model_selection" and any(term in detail for term in (
        "inference_memory_used_as_training_evidence",
        "comparisons_missing_advantages_or_risks",
        "missing_small_object_uncertainty",
        "selected_candidate_was_compared",
    )):
        return "model_rationale_validation_error"
    return {
        "completeness": "completeness_rejection",
        "task_interpretation": "task_interpretation_error",
        "data_check": "data_check_error",
        "model_selection": "unsupported_model",
        "dataset_selection": "infeasible_dataset_selection",
        "hyperparameters": "invalid_hyperparameters",
        "configuration_validation": "invalid_execution_configuration",
        "training_smoke": "training_smoke_error",
    }.get(stage, "unexpected_error")


def read_usage(job_id: str) -> dict[str, Any] | None:
    path = planning_usage_path(job_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


async def run_training_smoke(
    *, pipeline: Any, validated: dict[str, Any], job_id: str,
) -> dict[str, Any]:
    """Use the normal cache-aware data path and stop training after one batch."""
    smoke: dict[str, Any] = {"status": "failed", "stages": {}}
    started = perf_counter()
    try:
        smoke_config = copy.deepcopy(validated)
        batch_size = max(1, int(smoke_config.get("batch_size") or 1))
        for class_assignment in smoke_config.get("selected_data") or []:
            for source in class_assignment.get("sources") or []:
                for allocation in source.get("allocations") or []:
                    split = str(allocation.get("split") or "")
                    allocation["count"] = min(
                        int(allocation.get("count") or 1),
                        batch_size if split == "train" else 1,
                    )
        smoke["data_policy"] = {
            "source": "normal cache-aware downloader",
            "train_images_per_allocation_at_most": batch_size,
            "validation_and_test_images_per_allocation_at_most": 1,
            "saved_planning_configuration_changed": False,
        }
        await asyncio.to_thread(pipeline.download_data_step, smoke_config, job_id)
        smoke["stages"]["data_materialized"] = "passed"
        await asyncio.to_thread(pipeline.prepare_data_step, smoke_config, job_id)
        smoke["stages"]["data_prepared"] = "passed"
        readiness = execution._build_execution_readiness_report(pipeline, smoke_config, job_id)
        smoke["stages"]["execution_readiness"] = "passed"

        runtime = dict(smoke_config)
        runtime["_benchmark_max_epochs"] = 1
        runtime["_benchmark_max_batches"] = 1
        # A single smoke batch must execute an optimizer step even when the
        # planned run uses gradient accumulation across several batches.
        runtime["gradient_accumulation_steps"] = 1
        smoke["runtime_overrides"] = {
            "max_epochs": 1,
            "max_train_batches": 1,
            "gradient_accumulation_steps": 1,
        }
        if isinstance(pipeline, ClassificationPipeline):
            training_result = await asyncio.to_thread(pipeline.train_model_step, runtime, job_id)
        elif isinstance(pipeline, DetectionPipeline):
            training_result = await pipeline.train_model_step(runtime, job_id)
        else:
            raise RuntimeError(f"Unsupported benchmark pipeline: {type(pipeline).__name__}")
        smoke["stages"]["training_batch"] = "passed"
        smoke["stages"].update({
            "model_initialized": "passed",
            "forward_pass": "passed",
            "loss_computed": "passed",
            "backward_pass": "passed",
            "optimizer_step": "passed",
        })
        smoke["stages"]["validation_pass"] = (
            "passed" if isinstance(pipeline, ClassificationPipeline) else "not_verified"
        )
        smoke["status"] = "passed"
        smoke["training_result"] = training_result

        report_path = download_report_path(job_id)
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            performance = report.get("performance") or {}
            smoke["cache"] = {
                "cache_hits": performance.get("cache_hits", 0),
                "bytes_downloaded": performance.get("bytes_downloaded", 0),
            }
    except BaseException as exc:
        failed_stage = next((name for name in (
            "data_materialized", "data_prepared", "execution_readiness", "training_batch"
        ) if name not in smoke["stages"]), "training_batch")
        smoke["stages"][failed_stage] = "failed"
        smoke["error"] = _error_detail(exc)
        smoke["traceback"] = "".join(traceback.format_exception(exc))
    smoke["duration_seconds"] = round(perf_counter() - started, 3)
    return smoke


async def evaluate(
    case: BenchmarkCase, repetition: int, use_graphrag: bool, *, training_smoke: bool = False,
) -> dict[str, Any]:
    suffix = "graph" if use_graphrag else "nograph"
    job_id = f"batch-{case.id}-{suffix}-{repetition}-{uuid4().hex[:8]}"
    started = perf_counter()
    result: dict[str, Any] = {
        "job_id": job_id,
        "case_id": case.id,
        "repetition": repetition,
        "use_graphrag": use_graphrag,
        "status": "failed",
        "stages": {},
        "llm_attempts": {},
    }
    active_stage = "completeness"
    try:
        completeness = await planning.completenesscheck(planning.CompletenessCheckRequest(
            job_id=job_id, user_prompt=case.prompt, user_replies=[]
        ))
        if not completeness.accept or completeness.context is None:
            raise ValueError(completeness.reason or "Prompt was rejected as incomplete.")
        result["stages"][active_stage] = "passed"
        context: str | dict[str, Any] = completeness.context

        for active_stage, operation in STAGES:
            response = await operation(planning.StateRequest(
                context=context, job_id=job_id, use_graphrag=use_graphrag
            ))
            if response.get("llm_attempts"):
                result["llm_attempts"][active_stage] = response["llm_attempts"]
            context = response["context"]
            result["stages"][active_stage] = "passed"

        if not isinstance(context, dict):
            raise TypeError("Final planning context is not an object.")
        dataset_warnings = (
            (context.get("dataset_selection_decision_evidence") or {}).get("advisory_findings") or []
        )
        hpo_warnings = [
            finding for finding in ((context.get("hpo_decision") or {}).get("findings") or [])
            if finding.get("severity") == "preference"
        ]
        result["planning_warnings"] = {
            "dataset": dataset_warnings,
            "hyperparameters": hpo_warnings,
        }
        result["selection"] = {
            "task": context.get("task"),
            "classes": context.get("classes"),
            "model": (context.get("selected_model_info") or {}).get("model_name")
                or (context.get("selected_model_info") or {}).get("id"),
            "datasets": sorted({
                str(item.get("dataset_name") or item.get("dataset_id") or "")
                for item in context.get("selected_data") or [] if isinstance(item, dict)
            }),
            "hpo_config": context.get("hpo_config"),
        }

        active_stage = "configuration_validation"
        saved = json.loads(hpo_config_path(job_id).read_text(encoding="utf-8"))
        pipeline = execution.get_pipeline_by_task(job_id)
        validated = execution._validate_config(
            pipeline, saved, job_id=job_id, require_saved_config=True
        )
        result["configuration_validation"] = {
            "passed": True,
            "validated_fields": sorted(validated),
        }
        result["stages"][active_stage] = "passed"
        if training_smoke:
            active_stage = "training_smoke"
            result["training_smoke"] = await run_training_smoke(
                pipeline=pipeline, validated=validated, job_id=job_id
            )
            result["stages"][active_stage] = result["training_smoke"]["status"]
            if result["training_smoke"]["status"] != "passed":
                raise RuntimeError(result["training_smoke"].get("error") or "Training smoke failed.")
        result["status"] = "passed"
    except BaseException as exc:
        result["stages"][active_stage] = "failed"
        result["failure_category"] = classify_failure(active_stage, exc)
        result["error"] = _error_detail(exc)
        detail = _error_detail(exc)
        if isinstance(detail, dict):
            attempts = detail.get("interpretation_attempts") or detail.get("selection_attempts")
            if attempts:
                result["llm_attempts"][active_stage] = attempts
        result["traceback"] = "".join(traceback.format_exception(exc))
    finally:
        result["duration_seconds"] = round(perf_counter() - started, 3)
        result["llm_usage"] = read_usage(job_id)
    return result


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    usage = (result.get("llm_usage") or {}).get("totals") or {}
    selection = result.get("selection") or {}
    warnings = result.get("planning_warnings") or {}
    warning_items = [*(warnings.get("dataset") or []), *(warnings.get("hyperparameters") or [])]
    return {
        "job_id": result["job_id"], "case_id": result["case_id"],
        "repetition": result["repetition"], "use_graphrag": result["use_graphrag"],
        "status": result["status"], "failure_category": result.get("failure_category"),
        "duration_seconds": result["duration_seconds"],
        "training_smoke": (result.get("training_smoke") or {}).get("status"),
        "warning_count": len(warning_items),
        "warning_codes": json.dumps([
            item.get("code") or item.get("rule_id") for item in warning_items
        ]),
        "task": selection.get("task"), "model": selection.get("model"),
        "classes": json.dumps(selection.get("classes")),
        "datasets": json.dumps(selection.get("datasets")),
        "requests": usage.get("requests", 0), "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "calculated_cost_usd": usage.get("calculated_cost_usd"),
        "error": json.dumps(result.get("error"), default=str) if result.get("error") else None,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_flatten(item) for item in results]
    costs = [Decimal(str(row["calculated_cost_usd"])) for row in rows if row["calculated_cost_usd"] is not None]
    durations = [float(row["duration_seconds"]) for row in rows]
    return {
        "total_runs": len(rows),
        "technical_successes": sum(row["status"] == "passed" for row in rows),
        "technical_success_rate": sum(row["status"] == "passed" for row in rows) / len(rows) if rows else 0,
        "training_smoke_successes": sum(row["training_smoke"] == "passed" for row in rows),
        "total_cost_usd": format(sum(costs, Decimal(0)), "f") if costs else None,
        "mean_cost_usd": format(sum(costs, Decimal(0)) / len(costs), "f") if costs else None,
        "mean_duration_seconds": statistics.fmean(durations) if durations else None,
        "mean_input_tokens": statistics.fmean(row["input_tokens"] for row in rows) if rows else None,
        "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in rows) if rows else None,
        "failures_by_category": {
            category: sum(row["failure_category"] == category for row in rows)
            for category in sorted({row["failure_category"] for row in rows if row["failure_category"]})
        },
    }


def write_reports(output: Path, results: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    rows = [_flatten(item) for item in results]
    if rows:
        with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Batch planning evaluation", "", f"Generated: {datetime.now(UTC).isoformat()}", ""]
    lines.extend(f"- {key.replace('_', ' ').title()}: {value}" for key, value in summary.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"benchmark_results/{timestamp}").resolve()
    conditions = [True, False] if args.graphrag == "both" else [args.graphrag == "enabled"]
    selected = [case for case in CASES if not args.case or case.id in args.case]
    results: list[dict[str, Any]] = []
    saved_results = output / "runs.json"
    if args.resume and saved_results.is_file():
        loaded = json.loads(saved_results.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Existing report is not a result list: {saved_results}")
        results = loaded
    completed = {
        (item.get("case_id"), int(item.get("repetition", 0)), bool(item.get("use_graphrag")))
        for item in results
    }
    for use_graphrag in conditions:
        for case in selected:
            for repetition in range(1, args.repetitions + 1):
                if (case.id, repetition, use_graphrag) in completed:
                    continue
                print(f"[{len(results) + 1}] {case.id} repetition={repetition} graphrag={use_graphrag}", flush=True)
                result = await evaluate(
                    case, repetition, use_graphrag,
                    training_smoke=bool(getattr(args, "training_smoke", False)),
                )
                results.append(result)
                write_reports(output, results)
                print(f"    {result['status']} ({result['duration_seconds']}s)", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated evaluations of the LLM planning pipeline.")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--graphrag", choices=("enabled", "disabled", "both"), default="enabled")
    parser.add_argument("--case", action="append", help="Only run this case ID; repeat for multiple cases.")
    parser.add_argument("--output", help="Output directory (default: benchmark_results/<timestamp>).")
    parser.add_argument("--resume", action="store_true", help="Resume completed runs from --output.")
    parser.add_argument("--training-smoke", action="store_true", help="Prepare data and run one training batch.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    output = asyncio.run(run(args))
    print(f"Reports written to {output}")


if __name__ == "__main__":
    main()
