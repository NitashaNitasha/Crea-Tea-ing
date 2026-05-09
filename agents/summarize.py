"""Post-chapter summarizer for rolling context (structured one-field output)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.schemas import StoryBrief

SUMMARIZER_SYSTEM = """You summarize fiction chapters for continuity in later drafting.
Write 1–2 tight paragraphs: main plot beats, character states, unresolved threads, and setting facts that must carry forward.
No critique and no suggestions; third person if the story is third person, otherwise match the narrative stance briefly."""


class ChapterRollupSummary(BaseModel):
    """Structured summary for passing to the writer on later chapters."""

    summary: str = Field(
        ...,
        description="1–2 paragraphs: plot, character state, open threads, carry-forward facts",
    )


def summarize_chapter(
    *,
    llm: BaseChatModel,
    brief: StoryBrief,
    chapter_index_one_based: int,
    chapter_title: str,
    chapter_text: str,
) -> str:
    structured = llm.with_structured_output(ChapterRollupSummary)
    brief_json = brief.model_dump_json(indent=2)
    result: ChapterRollupSummary = structured.invoke(
        [
            SystemMessage(content=SUMMARIZER_SYSTEM),
            HumanMessage(
                content=(
                    f"Story brief (JSON):\n{brief_json}\n\n"
                    f"Chapter {chapter_index_one_based}: {chapter_title}\n\n"
                    f"Chapter text:\n{chapter_text.strip()}"
                )
            ),
        ]
    )
    return result.summary.strip()
