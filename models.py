from pydantic import BaseModel, Field
from typing import List, Optional


# ── Existing models ────────────────────────────────────────────────────────────

class SimilarCaseQuery(BaseModel):
    query_text: str = Field(..., description="Facts, issue, or legal question to search for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of cases to return")
    min_score: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum similarity score")


class CaseResult(BaseModel):
    case_id: str
    case_name: str
    date: str
    court: str
    judges: str
    case_number: str
    citation: str
    appellant: str
    respondent: str
    description: str
    similarity_score: float
    best_matching_excerpt: str
    matched_chunks: int
    download_url: str
    view_url: str


# ── Argument Builder models ────────────────────────────────────────────────────

class ArgumentRequest(BaseModel):
    facts: str = Field(
        ...,
        description="The facts of your case. Be as detailed as possible.",
        min_length=20
    )
    top_k_cases: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of similar past cases to use as context (default: 5)"
    )


class ArgumentPoint(BaseModel):
    point: str
    explanation: str
    supporting_case: Optional[str] = ""


class CounterArgument(BaseModel):
    point: str
    explanation: str


class RelevantLaw(BaseModel):
    section: str
    description: str


class SimilarCaseSummary(BaseModel):
    case_name: str
    citation: str
    relevance: str
    outcome: str


class ArgumentResponse(BaseModel):
    arguments_for: List[ArgumentPoint]
    arguments_against: List[ArgumentPoint]
    counter_arguments: List[CounterArgument]
    relevant_laws: List[RelevantLaw]
    similar_cases_summary: List[SimilarCaseSummary]
    overall_assessment: str
    cases_used: int = Field(description="Number of similar cases retrieved from database")