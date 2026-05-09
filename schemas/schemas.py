"""Pydantic models for novel planning, drafting, and editing."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class StoryBrief(BaseModel):
    """Finalized planning output: enough context to draft a storyline and chapters."""

    working_title: str = Field(..., description="Working title of the novel")
    logline: str = Field(..., description="One- or two-sentence premise")
    genre: str = Field(..., description="Genre and subgenre")
    point_of_view: str = Field(
        ...,
        description="Narrative POV, e.g. first person, third limited, omniscient",
    )
    tone: str = Field(..., description="Desired narrative tone and voice")
    themes: list[str] = Field(
        default_factory=list,
        description="Core themes to reflect in the story",
    )
    characters: str = Field(..., description="Main characters, relationships, and arcs")
    setting: str = Field(..., description="Time, place, and world details")
    conflict: str = Field(..., description="Central conflict or dramatic question")
    target_chapter_count: int | None = Field(
        default=None,
        ge=1,
        description="Target number of chapters when the author specified one",
    )
    target_length: str | None = Field(
        default=None,
        description="Length target, e.g. short story, novella, novel, approximate word band",
    )
    hard_constraints: str | None = Field(
        default=None,
        description="Non-negotiable constraints from the author (content, style, facts)",
    )


class ChapterOutline(BaseModel):
    """Outline for a single chapter."""

    title: str = Field(..., description="Chapter title or working heading")
    summary: str = Field(..., description="What happens in this chapter")
    beats: list[str] | None = Field(
        default=None,
        description="Optional ordered beats or scene goals",
    )


class NovelPlan(BaseModel):
    """High-level storyline plus per-chapter outlines from the drafter."""

    overall_storyline: str = Field(
        ...,
        description="Coherent high-level arc from setup through resolution",
    )
    chapters: list[ChapterOutline] = Field(
        ...,
        description="Chapter outlines consistent with the story brief and targets",
    )


class EditorCritique(BaseModel):
    """Structured editor feedback for revising a chapter draft."""

    issues: list[str] = Field(
        default_factory=list,
        description="Problems vs. outline or brief (plot, pacing, clarity, consistency)",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete improvement ideas",
    )
    priority_fixes: list[str] = Field(
        default_factory=list,
        description="Must-address items before the draft is acceptable",
    )


class PlanningQuestion(BaseModel):
    """Planning agent asks one focused follow-up."""

    kind: Literal["question"] = "question"
    next_question: str = Field(
        ...,
        description="Exactly one focused follow-up question; no preamble",
    )


class PlanningFinalize(BaseModel):
    """Planning agent is done; emit the full brief."""

    kind: Literal["finalize"] = "finalize"
    brief: StoryBrief


PlanningTurn = Annotated[
    Union[PlanningQuestion, PlanningFinalize],
    Field(discriminator="kind"),
]


__all__ = [
    "ChapterOutline",
    "EditorCritique",
    "NovelPlan",
    "PlanningFinalize",
    "PlanningQuestion",
    "PlanningTurn",
    "StoryBrief",
]
