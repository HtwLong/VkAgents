"""Read-only VisionKG SPARQL access for the viewer planning service."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .settings import (
    VISIONKG_SPARQL_ATTEMPTS,
    VISIONKG_SPARQL_ENDPOINT,
    VISIONKG_SPARQL_TIMEOUT,
    VISIONKG_SPARQL_TOKEN,
)

Transport = Callable[[str], dict[str, Any]]


def build_class_availability_query(class_name: str) -> str:
    literal = json.dumps(class_name, ensure_ascii=False)
    return f"""PREFIX cv: <http://vision.semkg.org/onto/v0.1/>
PREFIX schema: <http://schema.org/>

SELECT ?datasetName (COUNT(DISTINCT ?image) AS ?count)
WHERE {{
    ?image cv:hasAnnotation/cv:hasLabel/cv:label {literal} .
    ?image schema:isPartOf / schema:name ?datasetName .
}}
GROUP BY ?datasetName
ORDER BY DESC(?count)"""


def _http_transport(query: str) -> dict[str, Any]:
    parameters = {"query": query}
    if VISIONKG_SPARQL_TOKEN:
        parameters["token"] = VISIONKG_SPARQL_TOKEN
    request = Request(
        f"{VISIONKG_SPARQL_ENDPOINT}?{urlencode(parameters)}",
        headers={"Accept": "application/sparql-results+json"},
    )
    with urlopen(request, timeout=VISIONKG_SPARQL_TIMEOUT) as response:  # noqa: S310 - configured endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VisionKG returned a non-object SPARQL response.")
    return payload


def _bindings(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("results", {}).get("bindings")
    if not isinstance(raw, list):
        raise ValueError("VisionKG returned an unexpected SPARQL result shape.")
    rows = []
    for binding in raw:
        if not isinstance(binding, dict):
            continue
        rows.append({
            key: str(value.get("value"))
            for key, value in binding.items()
            if isinstance(value, dict) and value.get("value") is not None
        })
    return rows


def query_class_availability(
    classes: list[str],
    *,
    query_output_path: Path | None = None,
    transport: Transport | None = None,
) -> list[dict[str, Any]]:
    """Return class/dataset counts without downloading any data."""

    execute = transport or _http_transport
    rendered: list[str] = []
    result = []
    for class_name in dict.fromkeys(classes):
        query = build_class_availability_query(class_name)
        rendered.append(f"# Class: {class_name}\n{query}")
        if query_output_path is not None:
            query_output_path.parent.mkdir(parents=True, exist_ok=True)
            query_output_path.write_text("\n\n".join(rendered) + "\n", encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(VISIONKG_SPARQL_ATTEMPTS):
            try:
                rows = _bindings(execute(query))
                break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < VISIONKG_SPARQL_ATTEMPTS:
                    time.sleep(min(2**attempt, 4))
        else:
            raise RuntimeError(
                f"VisionKG availability query failed for {class_name!r}: {last_error}"
            ) from last_error
        sources = []
        for row in rows:
            try:
                count = int(row.get("count", ""))
            except ValueError:
                continue
            if row.get("datasetName") and count >= 0:
                sources.append({"dataset_name": row["datasetName"], "count": count})
        sources.sort(key=lambda item: (-item["count"], item["dataset_name"]))
        result.append({"class_name": class_name, "sources": sources})
    return result
