from __future__ import annotations

import os
from typing import TypeVar

from fastapi import HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from .settings import PLANNING_MODEL
from .store import planning_dir, read_json, write_json


T = TypeVar("T", bound=BaseModel)


async def structured_call(
    *, job_id: str, operation: str, prompt: str, response_model: type[T], model: str | None = None
) -> T:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    try:
        response = await AsyncOpenAI().beta.chat.completions.parse(
            model=model or PLANNING_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative computer-vision planning assistant. Use only the "
                        "supplied evidence, state uncertainty explicitly, and never claim that "
                        "training, downloading, evaluation, or inference was executed."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Planning model request failed: {exc}") from exc
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="Planning model returned no structured result.")
    usage = response.usage
    record_usage(
        job_id,
        operation,
        model or PLANNING_MODEL,
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        int(getattr(usage, "total_tokens", 0) or 0),
    )
    return parsed


def record_usage(job_id: str, operation: str, model: str, inputs: int, outputs: int, total: int) -> None:
    path = planning_dir(job_id) / "planning_llm_usage.json"
    document = read_json(path)
    if not isinstance(document, dict):
        document = {
            "schema_version": 1,
            "job_id": job_id,
            "scope": "planning",
            "currency": "USD",
            "totals": _bucket(),
            "models": {},
            "operations": {},
            "pricing": {},
            "usage_notes": ["Cost calculation is disabled in the lightweight backend."],
        }
    event = {"requests": 1, "input_tokens": inputs, "output_tokens": outputs, "total_tokens": total}
    for bucket in (
        document["totals"],
        document["models"].setdefault(model, _bucket()),
        document["operations"].setdefault(operation, _bucket()),
    ):
        for key, value in event.items():
            bucket[key] = int(bucket.get(key, 0)) + value
    write_json(path, document)


def _bucket() -> dict:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "calculated_cost_usd": None,
    }

