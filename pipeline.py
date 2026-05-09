"""Novel pipeline: drafter, sequential writer–editor chapters, summaries, Markdown assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from agents.drafter import run_drafter
from agents.editor import run_editor
from agents.summarize import summarize_chapter
from agents.writer import revise_chapter, write_chapter_initial
from schemas.schemas import EditorCritique, NovelPlan, StoryBrief

__all__ = [
    "MAX_EDITOR_ROUNDS",
    "ChapterLoopResult",
    "assemble_novel",
    "run_chapter_loop",
    "run_drafter",
]

MAX_EDITOR_ROUNDS = 3

HumanFeedbackFn = Callable[[EditorCritique, int, int], str]
"""(critique, chapter_index_one_based, round_index_zero_based) -> optional human notes."""


@dataclass(frozen=True)
class ChapterLoopResult:
    """Final chapter prose and per-chapter rollup summaries used as rolling context."""

    chapter_texts: list[str]
    rolling_summaries: list[str]


def run_chapter_loop(
    *,
    llm: BaseChatModel,
    brief: StoryBrief,
    plan: NovelPlan,
    max_editor_rounds: int = MAX_EDITOR_ROUNDS,
    human_feedback_fn: HumanFeedbackFn | None = None,
) -> ChapterLoopResult:
    """
    For each chapter in order: initial draft, then ``max_editor_rounds`` editor→revision cycles.
    ``human_feedback_fn`` is invoked after each machine critique; return empty string for
    machine-only revision. Rolling summaries of finalized chapters are passed into later writers.
    """
    if max_editor_rounds < 0:
        raise ValueError("max_editor_rounds must be >= 0")

    chapter_texts: list[str] = []
    rolling_summaries: list[str] = []
    prior: list[tuple[int, str]] = []

    for idx, outline in enumerate(plan.chapters, start=1):
        draft = write_chapter_initial(
            llm=llm,
            brief=brief,
            overall_storyline=plan.overall_storyline,
            outline=outline,
            prior_chapter_summaries=prior,
            chapter_index_one_based=idx,
        )
        for round_idx in range(max_editor_rounds):
            critique = run_editor(
                llm=llm,
                brief=brief,
                outline=outline,
                chapter_draft=draft,
                chapter_index_one_based=idx,
            )
            human_notes = (
                human_feedback_fn(critique, idx, round_idx)
                if human_feedback_fn is not None
                else ""
            )
            draft = revise_chapter(
                llm=llm,
                brief=brief,
                overall_storyline=plan.overall_storyline,
                outline=outline,
                prior_chapter_summaries=prior,
                chapter_index_one_based=idx,
                current_draft=draft,
                critique=critique,
                human_feedback=human_notes,
            )

        summary = summarize_chapter(
            llm=llm,
            brief=brief,
            chapter_index_one_based=idx,
            chapter_title=outline.title,
            chapter_text=draft,
        )
        chapter_texts.append(draft)
        rolling_summaries.append(summary)
        prior.append((idx, summary))

    return ChapterLoopResult(
        chapter_texts=chapter_texts,
        rolling_summaries=rolling_summaries,
    )


def assemble_novel(
    *,
    brief: StoryBrief,
    plan: NovelPlan,
    chapter_texts: list[str],
) -> str:
    """Build Markdown: title, overall storyline, then numbered chapters with outline titles."""
    if len(chapter_texts) != len(plan.chapters):
        raise ValueError(
            f"chapter_texts has {len(chapter_texts)} entries but plan has {len(plan.chapters)} outlines"
        )

    lines: list[str] = [
        f"# {brief.working_title.strip()}",
        "",
        "## Overall storyline",
        "",
        plan.overall_storyline.strip(),
        "",
    ]
    for i, (outline, body) in enumerate(zip(plan.chapters, chapter_texts, strict=True), start=1):
        title = outline.title.strip()
        lines.extend(
            [
                f"## Chapter {i}: {title}",
                "",
                body.strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
