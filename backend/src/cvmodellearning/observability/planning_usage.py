"""Persist token usage and calculated cost for planning-only LLM calls."""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Awaitable, TypeVar

from agents import Runner

from cvmodellearning.llm_config import MODEL_PRICING
from cvmodellearning.paths import planning_usage_path


T = TypeVar("T")
_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_MILLION = Decimal(1_000_000)


def _int_attr(value: Any, name: str) -> int:
    return int(getattr(value, name, 0) or 0)


def _cached_tokens(usage: Any) -> tuple[int, bool]:
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        details = getattr(usage, "input_tokens_details", None)
    if details is None:
        return 0, False
    return _int_attr(details, "cached_tokens"), True


def _reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        details = getattr(usage, "output_tokens_details", None)
    return _int_attr(details, "reasoning_tokens") if details is not None else 0


def _empty_bucket() -> dict[str, Any]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "calculated_cost_usd": None,
    }


def _cost(model: str, bucket: dict[str, Any]) -> Decimal | None:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    cached = Decimal(bucket["cached_input_tokens"])
    uncached = Decimal(max(bucket["input_tokens"] - bucket["cached_input_tokens"], 0))
    output = Decimal(bucket["output_tokens"])
    return (
        uncached * pricing["input_per_million"]
        + cached * pricing["cached_input_per_million"]
        + output * pricing["output_per_million"]
    ) / _MILLION


def _money(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP), "f")


def _read(path: Path, job_id: str) -> dict[str, Any]:
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "schema_version": 1,
        "job_id": job_id,
        "scope": "planning",
        "currency": "USD",
        "totals": _empty_bucket(),
        "models": {},
        "operations": {},
        "pricing": {},
        "usage_notes": [],
    }


def _add(bucket: dict[str, Any], event: dict[str, int]) -> None:
    for field in (
        "requests", "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_tokens", "total_tokens",
    ):
        bucket[field] = int(bucket.get(field, 0) or 0) + event[field]


def record_planning_usage(
    *,
    job_id: str,
    operation: str,
    model: str,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens_known: bool = True,
) -> dict[str, Any]:
    """Atomically add one completed model call to a run's planning summary."""
    path = planning_usage_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "requests": int(requests),
        "input_tokens": int(input_tokens),
        "cached_input_tokens": int(cached_input_tokens),
        "output_tokens": int(output_tokens),
        "reasoning_tokens": int(reasoning_tokens),
        "total_tokens": int(total_tokens),
    }
    with _LOCKS[str(path.resolve())]:
        document = _read(path, job_id)
        totals = document.setdefault("totals", _empty_bucket())
        model_bucket = document.setdefault("models", {}).setdefault(model, _empty_bucket())
        operation_bucket = document.setdefault("operations", {}).setdefault(operation, _empty_bucket())
        for bucket in (totals, model_bucket, operation_bucket):
            _add(bucket, event)

        pricing = MODEL_PRICING.get(model)
        if pricing:
            document.setdefault("pricing", {})[model] = {
                "input_per_million": str(pricing["input_per_million"]),
                "cached_input_per_million": str(pricing["cached_input_per_million"]),
                "output_per_million": str(pricing["output_per_million"]),
                "effective_date": pricing["effective_date"],
                "source": pricing["source"],
            }

        notes = document.setdefault("usage_notes", [])
        if not cached_tokens_known:
            note = (
                "The installed Agents SDK reports aggregate input tokens without a cached-token "
                "breakdown; those calls are calculated at the standard input rate."
            )
            if note not in notes:
                notes.append(note)

        model_costs: list[Decimal] = []
        all_models_priced = True
        for model_name, values in document["models"].items():
            value = _cost(model_name, values)
            values["calculated_cost_usd"] = _money(value)
            if value is None:
                all_models_priced = False
            else:
                model_costs.append(value)
        totals["calculated_cost_usd"] = _money(sum(model_costs, Decimal(0))) if all_models_priced else None

        # An operation can contain only the configured planning model today. Keep its
        # calculated cost useful while retaining model buckets as the source of truth.
        operation_cost = _cost(model, operation_bucket)
        operation_bucket["calculated_cost_usd"] = _money(operation_cost)

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return document


async def run_planning_agent(
    *, job_id: str, operation: str, agent: Any, input: Any,
) -> Any:
    result = await Runner.run(agent, input=input)
    requests = input_tokens = output_tokens = total_tokens = 0
    for response in getattr(result, "raw_responses", []):
        usage = response.usage
        requests += _int_attr(usage, "requests")
        input_tokens += _int_attr(usage, "input_tokens")
        output_tokens += _int_attr(usage, "output_tokens")
        total_tokens += _int_attr(usage, "total_tokens")
    if requests or total_tokens:
        record_planning_usage(
            job_id=job_id,
            operation=operation,
            model=str(agent.model),
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens_known=False,
        )
    return result


async def run_planning_completion(
    *, job_id: str, operation: str, model: str, awaitable: Awaitable[T],
) -> T:
    response = await awaitable
    usage = getattr(response, "usage", None)
    if usage is not None:
        cached, cached_known = _cached_tokens(usage)
        record_planning_usage(
            job_id=job_id,
            operation=operation,
            model=str(getattr(response, "model", None) or model),
            requests=1,
            input_tokens=_int_attr(usage, "prompt_tokens") or _int_attr(usage, "input_tokens"),
            output_tokens=_int_attr(usage, "completion_tokens") or _int_attr(usage, "output_tokens"),
            total_tokens=_int_attr(usage, "total_tokens"),
            cached_input_tokens=cached,
            reasoning_tokens=_reasoning_tokens(usage),
            cached_tokens_known=cached_known,
        )
    return response
