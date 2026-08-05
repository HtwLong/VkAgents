from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import List, Literal, Optional

class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept: bool
    reason: str = Field(..., description="Why accept/reject")
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Human-readable tweaks if rejected"
    )


class HpoFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="Exact top-level configuration field that needs repair")
    severity: Literal["hard_error", "safety_warning", "preference"]
    reason: str
    recommended_value: Optional[str] = Field(
        None,
        description="Concrete replacement serialized as text when one is justified",
    )
    rule_id: Optional[str] = Field(None, description="Supporting ontology recipe/rule ID")


class HpoDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept: bool
    reason: str
    findings: List[HpoFinding] = Field(default_factory=list)
    suggestions: Optional[List[str]] = None
    # Runtime-only audit metadata. It is deliberately absent from the LLM's
    # structured-output schema and from persisted decision JSON.
    _authorized_repair_fields: set[str] = PrivateAttr(default_factory=set)
    _diagnostics: List[dict] = PrivateAttr(default_factory=list)
