from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from cvmodellearning.schemas.revision import RevisionPlan


class RequirementAssessment(BaseModel):
    requirement: str = Field(min_length=1)
    status: Literal["satisfied", "not_satisfied", "unknown"]
    evidence: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class PostTrainingAssessment(BaseModel):
    assessment_id: str = Field(default_factory=lambda: uuid4().hex)
    job_id: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verdict: Literal[
        "satisfied", "partially_satisfied", "not_satisfied", "unknown"
    ]
    summary: str = Field(min_length=1)
    requirements: list[RequirementAssessment] = Field(default_factory=list)
    recommended_plan: RevisionPlan | None = None
    limitations: list[str] = Field(default_factory=list)


class AssessmentEligibility(BaseModel):
    eligible: bool
    reason: str | None = None
    can_create_revision: bool = False
    revision_reason: str | None = None

