import asyncio
from types import SimpleNamespace

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
