"""Drafter agent: StoryBrief → overall storyline + chapter outlines."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.schemas import NovelPlan, StoryBrief

DRAFTER_SYSTEM = """You are an experienced fiction story architect. Given a story brief, produce:
1) A coherent overall storyline from setup through resolution (high level, not scene-by-scene prose).
2) A list of chapter outlines that match the target chapter count when provided, otherwise a sensible default count for the length target.

Each chapter needs a title, summary, and optional beats. Outlines must align with POV, tone, themes, and hard constraints in the brief.
"""


def run_drafter(*, llm: BaseChatModel, brief: StoryBrief) -> NovelPlan:
    structured = llm.with_structured_output(NovelPlan)
    brief_text = brief.model_dump_json(indent=2)
    return structured.invoke(
        [
            SystemMessage(content=DRAFTER_SYSTEM),
            HumanMessage(
                content=f"Story brief (JSON):\n{brief_text}\n\nProduce the novel plan as specified."
            ),
        ]
    )
