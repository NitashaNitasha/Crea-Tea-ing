"""Editor agent: structured critique of a chapter draft."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.schemas import ChapterOutline, EditorCritique, StoryBrief

EDITOR_SYSTEM = """You are a developmental and line editor. Critique the draft against the outline and brief.
Output structured feedback only: issues, suggestions, and priority_fixes. Be specific and actionable.
Do not rewrite the chapter; do not paste long replacement prose.
"""


def run_editor(
    *,
    llm: BaseChatModel,
    brief: StoryBrief,
    outline: ChapterOutline,
    chapter_draft: str,
    chapter_index_one_based: int,
) -> EditorCritique:
    structured = llm.with_structured_output(EditorCritique)
    outline_json = outline.model_dump_json(indent=2)
    brief_json = brief.model_dump_json(indent=2)
    return structured.invoke(
        [
            SystemMessage(content=EDITOR_SYSTEM),
            HumanMessage(
                content=(
                    f"Chapter number: {chapter_index_one_based}\n\n"
                    f"Story brief (JSON):\n{brief_json}\n\n"
                    f"Chapter outline (JSON):\n{outline_json}\n\n"
                    "Chapter draft:\n"
                    f"{chapter_draft.strip()}"
                )
            ),
        ]
    )
