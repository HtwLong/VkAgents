import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cvmodellearning.agents import agents_utils
from cvmodellearning.agents.interpretation_agents import TaskExtractionPatch
from cvmodellearning.schemas.interpretation_schema import SynonymMatch
from routers import planning


def test_load_unified_dataset_classes_uses_requested_task(tmp_path, monkeypatch):
    paths = {
        None: tmp_path / "legacy.txt",
        "classification": tmp_path / "classification.txt",
        "detection": tmp_path / "detection.txt",
    }
    paths[None].write_text("legacy\n", encoding="utf-8")
    paths["classification"].write_text("bird\nshared\n", encoding="utf-8")
    paths["detection"].write_text("car\nshared\n", encoding="utf-8")
    monkeypatch.setattr(agents_utils, "unified_dataset_path", paths.__getitem__)

    assert agents_utils.load_unified_dataset_classes() == {"legacy"}
    assert agents_utils.load_unified_dataset_classes("classification") == {"bird", "shared"}
    assert agents_utils.load_unified_dataset_classes("detection") == {"car", "shared"}
    assert agents_utils.load_unified_dataset_classes("visual question answering") == {
        "bird", "car", "shared",
    }


def test_task_interpret_loads_vocabulary_for_extracted_task(monkeypatch):
    extracted = TaskExtractionPatch(task="classification", classes=["blue jay"])
    calls = []

    async def fake_run(agent, input):
        return SimpleNamespace(final_output=extracted)

    def fake_load(task):
        calls.append(task)
        return {"blue jay"}

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "load_unified_dataset_classes", fake_load)
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)

    result = asyncio.run(planning.task_interpret(planning.StateRequest(
        job_id="task-vocabulary-test",
        context={"user_query": "Classify blue jays."},
    )))

    assert calls == ["classification"]
    assert result["context"]["classes"] == ["blue jay"]


def test_task_interpret_uses_pedestrian_for_person_in_traffic_domain(monkeypatch):
    extracted = TaskExtractionPatch(
        task="detection",
        classes=["person", "car"],
        application_domain="dense urban traffic monitoring",
    )

    async def fake_run(agent, input):
        return SimpleNamespace(final_output=extracted)

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(
        planning,
        "load_unified_dataset_classes",
        lambda _task: {"person", "pedestrian", "car"},
    )
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)

    result = asyncio.run(planning.task_interpret(planning.StateRequest(
        job_id="traffic-person-normalization-test",
        context={"user_query": "Detect people and cars in traffic."},
    )))

    assert result["context"]["classes"] == ["pedestrian", "car"]


def test_task_interpret_preserves_superclass_expansion_provenance(monkeypatch):
    extracted = TaskExtractionPatch(task="detection", classes=["furniture"])

    async def fake_run(agent, input):
        if agent is planning.task_interpretation_agent:
            return SimpleNamespace(final_output=extracted)
        return SimpleNamespace(final_output=SynonymMatch(
            original_class="furniture",
            found_match=True,
            dataset_classes=["chair", "table", "carpet", "bed"],
            reason="Expanded broader user category.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(
        planning,
        "load_unified_dataset_classes",
        lambda _task: {"chair", "table", "carpet", "bed"},
    )
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)

    result = asyncio.run(planning.task_interpret(planning.StateRequest(
        job_id="class-expansion-test",
        context={"user_query": "Detect furniture."},
    )))

    assert result["context"]["class_expansions"] == {
        "furniture": ["chair", "table", "carpet", "bed"]
    }


def test_task_interpret_repairs_all_unmapped_classes_in_one_round(monkeypatch):
    initial = TaskExtractionPatch(task="classification", classes=["one", "two"])
    repaired = TaskExtractionPatch(task="classification", classes=["1", "2"])
    repair_inputs = []

    async def fake_run(agent, input):
        if agent is planning.task_interpretation_agent:
            if not repair_inputs:
                repair_inputs.append(None)
                return SimpleNamespace(final_output=initial)
            repair_inputs.append(json.loads(input))
            return SimpleNamespace(final_output=repaired)
        user_class = input.split("User Class: '", 1)[1].split("'", 1)[0]
        return SimpleNamespace(final_output=SynonymMatch(
            original_class=user_class,
            found_match=False,
            dataset_classes=[],
            reason="No match.",
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "load_unified_dataset_classes", lambda _task: {"1", "2"})
    monkeypatch.setattr(planning, "save_checkpoint", lambda *args: None)

    result = asyncio.run(planning.task_interpret(planning.StateRequest(
        job_id="all-class-repair-test",
        context={"user_query": "Recognize one and two."},
    )))

    assert repair_inputs[1]["invalid_classes"] == ["one", "two"]
    assert result["context"]["classes"] == ["1", "2"]


def test_task_interpret_failed_repair_reports_both_attempts(monkeypatch):
    outputs = [
        TaskExtractionPatch(task="classification", classes=["one", "two"]),
        TaskExtractionPatch(task="classification", classes=["uno", "dos"]),
    ]

    async def fake_run(agent, input):
        if agent is planning.task_interpretation_agent:
            return SimpleNamespace(final_output=outputs.pop(0))
        return SimpleNamespace(final_output=SynonymMatch(
            original_class="invalid", found_match=False, dataset_classes=[], reason="No match."
        ))

    monkeypatch.setattr(planning.Runner, "run", fake_run)
    monkeypatch.setattr(planning, "load_unified_dataset_classes", lambda _task: {"1", "2"})

    with pytest.raises(HTTPException) as caught:
        asyncio.run(planning.task_interpret(planning.StateRequest(
            job_id="failed-class-repair-test",
            context={"user_query": "Recognize numbers."},
        )))

    assert caught.value.detail["invalid_classes"] == ["uno", "dos"]
    assert len(caught.value.detail["interpretation_attempts"]) == 2
