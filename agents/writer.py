"""Writer agent: initial chapter draft and revisions (optional human_feedback)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.schemas import ChapterOutline, EditorCritique, StoryBrief

WRITER_SYSTEM_INITIAL = """You are a skilled fiction writer. Write one chapter of prose that fulfills the outline and fits the brief (genre, POV, tone, themes, constraints).
Use the overall storyline and prior chapter summaries for continuity; do not contradict established facts.
Output only the chapter text (no title line unless the outline implies one)."""

WRITER_SYSTEM_REVISE = """You are a skilled fiction writer revising a chapter. Apply the structured editor feedback and any human author notes.
Preserve what already works; fix issues, address priority fixes, and incorporate suggestions where appropriate.
If human feedback is empty, rely only on the machine critique.
Output only the revised chapter prose (no meta-commentary)."""


def _format_prior_summaries(summaries: list[tuple[int, str]]) -> str:
    if not summaries:
        return "(No prior chapters yet.)"
    lines = []
    for idx, text in summaries:
        lines.append(f"Chapter {idx} summary:\n{text.strip()}")
    return "\n\n".join(lines)


def _format_critique(critique: EditorCritique) -> str:
    parts = []
    if critique.issues:
        parts.append("Issues:\n" + "\n".join(f"- {i}" for i in critique.issues))
    if critique.priority_fixes:
        parts.append(
            "Priority fixes:\n" + "\n".join(f"- {p}" for p in critique.priority_fixes)
        )
    if critique.suggestions:
        parts.append(
            "Suggestions:\n" + "\n".join(f"- {s}" for s in critique.suggestions)
        )
    return "\n\n".join(parts) if parts else "(No structured critique.)"


def write_chapter_initial(
    *,
    llm: BaseChatModel,
    brief: StoryBrief,
    overall_storyline: str,
    outline: ChapterOutline,
    prior_chapter_summaries: list[tuple[int, str]],
    chapter_index_one_based: int,
) -> str:
    outline_json = outline.model_dump_json(indent=2)
    brief_json = brief.model_dump_json(indent=2)
    user = (
        f"Chapter index: {chapter_index_one_based}\n\n"
        f"Story brief (JSON):\n{brief_json}\n\n"
        f"Overall storyline:\n{overall_storyline.strip()}\n\n"
        f"Prior chapter summaries:\n{_format_prior_summaries(prior_chapter_summaries)}\n\n"
        f"Current chapter outline (JSON):\n{outline_json}\n\n"
        "Write the full chapter draft."
    )
    reply = llm.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM_INITIAL),
            HumanMessage(content=user),
        ]
    )
    return (reply.content or "").strip()


def revise_chapter(
    *,
    llm: BaseChatModel,
    brief: StoryBrief,
    overall_storyline: str,
    outline: ChapterOutline,
    prior_chapter_summaries: list[tuple[int, str]],
    chapter_index_one_based: int,
    current_draft: str,
    critique: EditorCritique,
    human_feedback: str | None = None,
) -> str:
    outline_json = outline.model_dump_json(indent=2)
    brief_json = brief.model_dump_json(indent=2)
    human_notes = (human_feedback or "").strip()
    human_block = (
        human_notes
        if human_notes
        else "(No additional human notes for this round; use only the machine critique.)"
    )
    user = (
        f"Chapter index: {chapter_index_one_based}\n\n"
        f"Story brief (JSON):\n{brief_json}\n\n"
        f"Overall storyline:\n{overall_storyline.strip()}\n\n"
        f"Prior chapter summaries:\n{_format_prior_summaries(prior_chapter_summaries)}\n\n"
        f"Chapter outline (JSON):\n{outline_json}\n\n"
        "Machine editor critique:\n"
        f"{_format_critique(critique)}\n\n"
        "Human author notes (optional):\n"
        f"{human_block}\n\n"
        "Current draft to revise:\n"
        f"{current_draft.strip()}\n\n"
        "Produce the revised chapter."
    )
    reply = llm.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM_REVISE),
            HumanMessage(content=user),
        ]
    )
    return (reply.content or "").strip()
