from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept: bool
    reason: str = Field(..., description="Why accept/reject")
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Human-readable tweaks if rejected"
    )