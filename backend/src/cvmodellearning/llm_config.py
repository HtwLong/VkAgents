"""Central configuration and pricing for hosted LLM calls."""

from __future__ import annotations

import os
from decimal import Decimal


PLANNING_MODEL = os.getenv("PLANNING_LLM_MODEL", "gpt-5-nano")
ASSESSMENT_MODEL = os.getenv("ASSESSMENT_LLM_MODEL", PLANNING_MODEL)

# Standard API text-token prices per one million tokens. Keep the effective date
# with the rates so persisted run costs remain reproducible when prices change.
MODEL_PRICING = {
    "gpt-5-nano": {
        "input_per_million": Decimal("0.05"),
        "cached_input_per_million": Decimal("0.005"),
        "output_per_million": Decimal("0.40"),
        "effective_date": "2026-08-15",
        "source": "https://developers.openai.com/api/docs/models/gpt-5-nano",
    },
    "gpt-5-nano-2025-08-07": {
        "input_per_million": Decimal("0.05"),
        "cached_input_per_million": Decimal("0.005"),
        "output_per_million": Decimal("0.40"),
        "effective_date": "2026-08-15",
        "source": "https://developers.openai.com/api/docs/models/gpt-5-nano",
    },
}
