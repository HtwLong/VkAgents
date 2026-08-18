import json
import asyncio

import torch
from torch.utils.data import DataLoader, TensorDataset

from cvmodellearning.benchmarks.batch_eval import (
    classify_failure,
    run_training_smoke,
    summarize,
    write_reports,
)
from fastapi import HTTPException
from cvmodellearning.pipelines.classification_pipe import ClassificationPipeline
from cvmodellearning.pipelines.detection_pipe import DetectionPipeline
from cvmodellearning.benchmarks import batch_eval
from cvmodellearning.training.training_utils import train_one_epoch


def test_summary_and_reports_include_usage_and_failures(tmp_path):
    results = [{
        "job_id": "job-1", "case_id": "example", "repetition": 1,
        "use_graphrag": True, "status": "passed", "stages": {},
        "duration_seconds": 2.0,
        "selection": {"task": "classification", "model": "m", "classes": [], "datasets": []},
        "training_smoke": {"status": "passed"},
        "llm_usage": {"totals": {
            "requests": 2, "input_tokens": 100, "cached_input_tokens": 10,
            "output_tokens": 20, "reasoning_tokens": 5, "total_tokens": 120,
            "calculated_cost_usd": "0.01000000",
        }},
    }, {
        "job_id": "job-2", "case_id": "example", "repetition": 2,
        "use_graphrag": True, "status": "failed", "stages": {},
        "failure_category": "invalid_hyperparameters", "duration_seconds": 4.0,
        "llm_usage": None, "error": "invalid",
    }]

    summary = summarize(results)
    assert summary["technical_success_rate"] == 0.5
    assert "semantic_success_rate" not in summary
    assert summary["total_cost_usd"] == "0.01000000"
    assert summary["failures_by_category"] == {"invalid_hyperparameters": 1}

    write_reports(tmp_path, results)
    assert json.loads((tmp_path / "summary.json").read_text())["total_runs"] == 2
    assert (tmp_path / "runs.csv").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_failure_categories_distinguish_ontology_and_model_rationale_errors():
    ontology_error = HTTPException(status_code=400, detail={
        "message": "Class ontology mapping failed after one repair round.",
        "invalid_classes": ["one"],
    })
    rationale_error = HTTPException(status_code=422, detail={
        "message": "The LLM must select and compare an exact feasible GraphRAG candidate.",
        "inference_memory_used_as_training_evidence": True,
    })

    assert classify_failure("task_interpretation", ontology_error) == "class_ontology_mapping_error"
    assert classify_failure("model_selection", rationale_error) == "model_rationale_validation_error"


def test_training_smoke_limit_performs_exactly_one_optimizer_step():
    model = torch.nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.ones(6, 2), torch.zeros(6, dtype=torch.long)), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    train_one_epoch(
        model, loader, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
        max_batches=1,
    )

    assert steps == 1


def test_smoke_execution_caps_data_and_keeps_limits_out_of_saved_config(monkeypatch, tmp_path):
    observed = {}

    def download(self, config, job_id):
        observed["download"] = config

    def prepare(self, config, job_id):
        observed["prepare"] = config

    def train(self, config, job_id):
        observed["train"] = config
        return {"ok": True}

    monkeypatch.setattr(ClassificationPipeline, "download_data_step", download)
    monkeypatch.setattr(ClassificationPipeline, "prepare_data_step", prepare)
    monkeypatch.setattr(ClassificationPipeline, "train_model_step", train)
    monkeypatch.setattr(batch_eval.execution, "_build_execution_readiness_report", lambda *args: {"ready": True})
    monkeypatch.setattr(batch_eval, "download_report_path", lambda job_id: tmp_path / "missing.json")
    config = {
        "batch_size": 4,
        "gradient_accumulation_steps": 3,
        "selected_data": [{"class_name": "cat", "sources": [{
            "dataset_name": "d", "allocations": [
                {"split": "train", "count": 100},
                {"split": "validation", "count": 20},
                {"split": "test", "count": 20},
            ],
        }]}],
    }

    result = asyncio.run(run_training_smoke(
        pipeline=ClassificationPipeline(), validated=config, job_id="unit-smoke-job",
    ))

    assert result["status"] == "passed"
    assert [item["count"] for item in observed["download"]["selected_data"][0]["sources"][0]["allocations"]] == [4, 1, 1]
    assert observed["train"]["_benchmark_max_batches"] == 1
    assert observed["train"]["gradient_accumulation_steps"] == 1
    assert "_benchmark_max_batches" not in config
    assert config["selected_data"][0]["sources"][0]["allocations"][0]["count"] == 100


def test_smoke_execution_supports_torchvision_detection(monkeypatch, tmp_path):
    observed = {}

    monkeypatch.setattr(DetectionPipeline, "download_data_step", lambda self, config, job_id: None)
    monkeypatch.setattr(DetectionPipeline, "prepare_data_step", lambda self, config, job_id: None)

    async def train(self, config, job_id):
        observed.update(config)
        return {"ok": True}

    monkeypatch.setattr(DetectionPipeline, "train_model_step", train)
    monkeypatch.setattr(batch_eval.execution, "_build_execution_readiness_report", lambda *args: {"ready": True})
    monkeypatch.setattr(batch_eval, "download_report_path", lambda job_id: tmp_path / "missing.json")

    result = asyncio.run(run_training_smoke(
        pipeline=DetectionPipeline(),
        validated={"model_name": "faster_rcnn_r50", "batch_size": 1, "selected_data": []},
        job_id="torchvision-smoke-test",
    ))

    assert result["status"] == "passed"
    assert observed["_benchmark_max_epochs"] == 1
    assert observed["_benchmark_max_batches"] == 1
