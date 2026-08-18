import json
import asyncio
from types import SimpleNamespace

from cvmodellearning import paths
from cvmodellearning.observability import planning_usage


def _usage(*, prompt, cached, completion, reasoning=0):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def test_direct_completion_persists_api_usage_and_current_price(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    response = SimpleNamespace(
        model="gpt-5-nano",
        usage=_usage(prompt=1_000_000, cached=200_000, completion=100_000, reasoning=25_000),
    )

    returned = asyncio.run(
        planning_usage.run_planning_completion(
            job_id="priced-run",
            operation="hpo_optimizer",
            model="gpt-5-nano",
            awaitable=_return(response),
        )
    )

    assert returned is response
    saved = json.loads(paths.planning_usage_path("priced-run").read_text())
    assert saved["totals"]["input_tokens"] == 1_000_000
    assert saved["totals"]["cached_input_tokens"] == 200_000
    assert saved["totals"]["output_tokens"] == 100_000
    assert saved["totals"]["reasoning_tokens"] == 25_000
    # 800k * $0.05/M + 200k * $0.005/M + 100k * $0.40/M
    assert saved["totals"]["calculated_cost_usd"] == "0.08100000"
    assert saved["pricing"]["gpt-5-nano"]["effective_date"] == "2026-08-15"


def test_agent_wrapper_sums_all_raw_model_responses(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    result = SimpleNamespace(raw_responses=[
        SimpleNamespace(usage=SimpleNamespace(
            requests=1, input_tokens=100, output_tokens=20, total_tokens=120,
        )),
        SimpleNamespace(usage=SimpleNamespace(
            requests=1, input_tokens=50, output_tokens=10, total_tokens=60,
        )),
    ])

    async def fake_run(agent, input):
        return result

    monkeypatch.setattr(planning_usage.Runner, "run", fake_run)
    agent = SimpleNamespace(model="gpt-5-nano")

    returned = asyncio.run(
        planning_usage.run_planning_agent(
            job_id="agent-run",
            operation="model_selection",
            agent=agent,
            input="state",
        )
    )

    assert returned is result
    saved = json.loads(paths.planning_usage_path("agent-run").read_text())
    assert saved["totals"]["requests"] == 2
    assert saved["totals"]["input_tokens"] == 150
    assert saved["totals"]["output_tokens"] == 30
    assert saved["totals"]["total_tokens"] == 180
    assert saved["usage_notes"]


async def _return(value):
    return value
