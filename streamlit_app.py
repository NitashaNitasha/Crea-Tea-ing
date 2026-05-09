"""Streamlit UI: planning chat, drafting, stepped writer–editor episodes, script export."""

from __future__ import annotations

import os
from typing import Literal

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from agents import (
    get_novel_llm,
    revise_chapter,
    run_drafter,
    run_editor,
    run_planning_turn,
    summarize_chapter,
    write_chapter_initial,
)
from agents._llm import DEFAULT_NOVEL_MODEL
from pipeline import MAX_EDITOR_ROUNDS, assemble_novel
from schemas.schemas import EditorCritique, PlanningFinalize, PlanningQuestion, StoryBrief

Phase = Literal["PLANNING", "DRAFTING", "WRITING", "DONE"]

MAX_PLANNING_QUESTIONS = 10


def _init_session_defaults() -> None:
    ss = st.session_state
    ss.setdefault("phase", "PLANNING")
    ss.setdefault("planning_started", False)
    ss.setdefault("initial_author_prompt", "")
    ss.setdefault("planning_messages", [])  # list[tuple[str, str]] role user|assistant
    ss.setdefault("questions_asked", 0)
    ss.setdefault("pending_brief", None)  # StoryBrief | None
    ss.setdefault("story_brief", None)
    ss.setdefault("novel_plan", None)
    ss.setdefault("status", "")
    ss.setdefault("error", None)
    # Writing
    ss.setdefault("ch_idx", 0)
    ss.setdefault("chapter_texts", [])
    ss.setdefault("prior_summaries", [])  # list[tuple[int, str]]
    ss.setdefault("current_draft", "")
    ss.setdefault("pending_critique", None)
    ss.setdefault("completed_revision_rounds", 0)


def _wipe_session() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_session_defaults()


@st.dialog("Start a new series?")
def _new_project_dialog() -> None:
    st.caption(
        "This clears planning chat, the story brief, the outline, all episode drafts, and export state. "
        "This cannot be undone."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("Reset everything", type="primary", use_container_width=True):
            _wipe_session()
            st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Drama script writer")
        st.caption(
            "Plan your show in chat, lock a brief, generate an outline, then draft each episode script "
            "with machine edits and optional notes."
        )
        st.divider()
        st.markdown("**Model** (`NOVEL_LLM`)")
        st.code(os.getenv("NOVEL_LLM", "").strip() or DEFAULT_NOVEL_MODEL, language=None)
        st.divider()
        st.markdown("**Where you are**")
        phase: Phase = st.session_state.phase
        order: list[Phase] = ["PLANNING", "DRAFTING", "WRITING", "DONE"]
        labels = {
            "PLANNING": "Planning & brief",
            "DRAFTING": "Episode outline",
            "WRITING": "Episodes",
            "DONE": "Export",
        }
        cur_i = order.index(phase)
        for i, p in enumerate(order):
            name = labels[p]
            if i < cur_i:
                st.markdown(f":white_check_mark: ~~{name}~~")
            elif i == cur_i:
                st.markdown(f":round_pushpin: **{name}**")
            else:
                st.markdown(f":white_circle: {name}")
        if st.session_state.get("status"):
            st.success(st.session_state.status)
        st.divider()
        if st.button("New project…", help="Clear everything and return to planning", use_container_width=True):
            _new_project_dialog()


@st.cache_resource
def _novel_llm() -> BaseChatModel:
    return get_novel_llm()


def _format_critique_display(c: EditorCritique) -> str:
    parts: list[str] = []
    if c.issues:
        parts.append("**Issues**\n" + "\n".join(f"- {i}" for i in c.issues))
    if c.priority_fixes:
        parts.append("**Priority fixes**\n" + "\n".join(f"- {p}" for p in c.priority_fixes))
    if c.suggestions:
        parts.append("**Suggestions**\n" + "\n".join(f"- {s}" for s in c.suggestions))
    return "\n\n".join(parts) if parts else "(No structured critique.)"


def _run_planning_and_store() -> None:
    llm = _novel_llm()
    st.session_state.error = None
    st.session_state.status = ""
    try:
        turn = run_planning_turn(
            llm=llm,
            planning_messages=st.session_state.planning_messages,
            initial_author_prompt=st.session_state.initial_author_prompt,
            questions_asked=st.session_state.questions_asked,
            max_questions=MAX_PLANNING_QUESTIONS,
        )
        if isinstance(turn, PlanningQuestion):
            st.session_state.planning_messages.append(("assistant", turn.next_question))
            st.session_state.questions_asked += 1
            st.session_state.pending_brief = None
        else:
            assert isinstance(turn, PlanningFinalize)
            st.session_state.pending_brief = turn.brief
    except Exception as e:  # noqa: BLE001 — show in UI
        st.session_state.error = str(e)


def _reset_writing_for_chapter() -> None:
    st.session_state.current_draft = ""
    st.session_state.pending_critique = None
    st.session_state.completed_revision_rounds = 0
    st.session_state["skip_human_remaining"] = False


def _finalize_current_chapter(llm: BaseChatModel) -> None:
    plan = st.session_state.novel_plan
    brief = st.session_state.story_brief
    assert plan is not None and brief is not None
    idx = st.session_state.ch_idx
    outline = plan.chapters[idx]
    draft = st.session_state.current_draft
    with st.spinner("Summarizing episode for continuity…"):
        summary = summarize_chapter(
            llm=llm,
            brief=brief,
            chapter_index_one_based=idx + 1,
            chapter_title=outline.title,
            chapter_text=draft,
        )
    st.session_state.chapter_texts.append(draft)
    st.session_state.prior_summaries.append((idx + 1, summary))
    st.session_state.ch_idx += 1
    _reset_writing_for_chapter()
    if st.session_state.ch_idx >= len(plan.chapters):
        st.session_state.phase = "DONE"
        st.session_state.status = "All episodes complete. Download below."


def _render_planning() -> None:
    st.header("Planning")
    st.caption(
        "Describe your drama series or episode arc. The assistant asks focused follow-ups or finalizes a brief "
        f"(max {MAX_PLANNING_QUESTIONS} questions)."
    )

    if st.session_state.error:
        st.error(st.session_state.error)

    if not st.session_state.planning_started:
        idea = st.text_area(
            "Initial concept (series / season / episode arc)",
            value=st.session_state.initial_author_prompt,
            height=160,
            key="planning_idea_area",
        )
        if st.button("Start planning", type="primary"):
            if not idea.strip():
                st.warning("Enter an initial idea first.")
            else:
                st.session_state.initial_author_prompt = idea.strip()
                st.session_state.planning_started = True
                with st.spinner("Thinking…"):
                    _run_planning_and_store()
                st.rerun()
        return

    st.subheader("Conversation")
    with st.chat_message("user"):
        st.markdown(st.session_state.initial_author_prompt)

    for role, content in st.session_state.planning_messages:
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content)

    brief = st.session_state.pending_brief
    if brief is not None:
        st.success("Planning produced a story brief. Review and lock it to continue.")
        with st.expander("Story brief", expanded=True):
            st.write(f"**Working title:** {brief.working_title}")
            st.write(f"**Logline:** {brief.logline}")
            st.write(f"**Genre:** {brief.genre}")
            st.write(f"**POV:** {brief.point_of_view}")
            st.write(f"**Tone:** {brief.tone}")
            if brief.themes:
                st.write("**Themes:** " + ", ".join(brief.themes))
            st.write(f"**Characters:** {brief.characters}")
            st.write(f"**Setting:** {brief.setting}")
            st.write(f"**Conflict:** {brief.conflict}")
            if brief.target_chapter_count is not None:
                st.write(f"**Target episodes:** {brief.target_chapter_count}")
            if brief.target_length:
                st.write(f"**Length target:** {brief.target_length}")
            if brief.hard_constraints:
                st.write(f"**Hard constraints:** {brief.hard_constraints}")
        if st.button("Lock brief & continue to drafting", type="primary"):
            st.session_state.story_brief = brief
            st.session_state.pending_brief = None
            st.session_state.phase = "DRAFTING"
            st.session_state.error = None
            st.rerun()
        return

    reply = st.chat_input("Your answer")
    if reply and reply.strip():
        st.session_state.planning_messages.append(("user", reply.strip()))
        with st.spinner("Thinking…"):
            _run_planning_and_store()
        st.rerun()


def _render_drafting() -> None:
    st.header("Drafting")
    brief = st.session_state.story_brief
    assert brief is not None
    st.caption("Generate a high-level storyline and per-episode outlines from your brief.")

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.button("Back to planning"):
        st.session_state.phase = "PLANNING"
        st.session_state.error = None
        st.rerun()

    if st.session_state.novel_plan is None:
        if st.button("Generate episode outline", type="primary"):
            st.session_state.error = None
            with st.spinner("Drafting outline…"):
                try:
                    st.session_state.novel_plan = run_drafter(llm=_novel_llm(), brief=brief)
                except Exception as e:  # noqa: BLE001
                    st.session_state.error = str(e)
            st.rerun()
        return

    plan = st.session_state.novel_plan
    st.subheader("Overall storyline")
    st.markdown(plan.overall_storyline)
    st.subheader(f"Episodes ({len(plan.chapters)})")
    for i, ch in enumerate(plan.chapters, start=1):
        with st.expander(f"Episode {i}: {ch.title}"):
            st.markdown(ch.summary)
            if ch.beats:
                st.markdown("**Beats**")
                for b in ch.beats:
                    st.markdown(f"- {b}")

    if st.button("Start writing episodes", type="primary"):
        st.session_state.phase = "WRITING"
        st.session_state.ch_idx = 0
        st.session_state.chapter_texts = []
        st.session_state.prior_summaries = []
        _reset_writing_for_chapter()
        st.session_state["skip_human_remaining"] = False
        st.session_state.error = None
        st.rerun()


def _render_writing() -> None:
    st.header("Episode scripts")
    plan = st.session_state.novel_plan
    brief = st.session_state.story_brief
    assert plan is not None and brief is not None

    if st.session_state.error:
        st.error(st.session_state.error)

    total = len(plan.chapters)
    done = st.session_state.ch_idx
    st.progress(min(done / total, 1.0), text=f"Episode {done + 1} of {total}" if done < total else "All episodes drafted")

    if "skip_human_remaining" not in st.session_state:
        st.session_state["skip_human_remaining"] = False
    st.checkbox(
        "Skip human notes for remaining editor rounds (this episode)",
        key="skip_human_remaining",
    )

    idx = st.session_state.ch_idx
    if idx >= total:
        st.session_state.phase = "DONE"
        st.rerun()
        return

    outline = plan.chapters[idx]
    st.subheader(f"Episode {idx + 1}: {outline.title}")
    st.caption(
        "Flow: **initial draft** → up to "
        f"**{MAX_EDITOR_ROUNDS}** editor rounds (critique → optional your notes → revise) → **finalize**."
    )

    llm = _novel_llm()

    if not st.session_state.current_draft.strip():
        if st.button("Write initial draft", type="primary", key=f"init_draft_{idx}"):
            st.session_state.error = None
            with st.spinner("Writing initial draft…"):
                try:
                    st.session_state.current_draft = write_chapter_initial(
                        llm=llm,
                        brief=brief,
                        overall_storyline=plan.overall_storyline,
                        outline=outline,
                        prior_chapter_summaries=st.session_state.prior_summaries,
                        chapter_index_one_based=idx + 1,
                    )
                except Exception as e:  # noqa: BLE001
                    st.session_state.error = str(e)
            st.rerun()
        return

    with st.expander("Current draft", expanded=False):
        st.markdown(st.session_state.current_draft)

    r = st.session_state.completed_revision_rounds
    if r < MAX_EDITOR_ROUNDS:
        st.info(
            f"**Editor round {r + 1} of {MAX_EDITOR_ROUNDS}** — run the machine editor, "
            "add optional feedback, then revise."
        )

    if st.session_state.completed_revision_rounds >= MAX_EDITOR_ROUNDS:
        if st.button("Finalize episode", type="primary", key=f"finalize_{idx}"):
            st.session_state.error = None
            try:
                _finalize_current_chapter(llm)
            except Exception as e:  # noqa: BLE001
                st.session_state.error = str(e)
            st.rerun()
        return

    if st.session_state.pending_critique is None:
        if st.button("Get machine editor critique", type="primary", key=f"editor_{idx}_{st.session_state.completed_revision_rounds}"):
            st.session_state.error = None
            with st.spinner("Editor reviewing…"):
                try:
                    st.session_state.pending_critique = run_editor(
                        llm=llm,
                        brief=brief,
                        outline=outline,
                        chapter_draft=st.session_state.current_draft,
                        chapter_index_one_based=idx + 1,
                    )
                except Exception as e:  # noqa: BLE001
                    st.session_state.error = str(e)
            st.rerun()
        return

    crit = st.session_state.pending_critique
    st.markdown("### Machine editor critique")
    st.markdown(_format_critique_display(crit))

    human_key = f"human_fb_{idx}_{st.session_state.completed_revision_rounds}"
    human_notes = st.text_area(
        "Your feedback (optional)",
        height=120,
        key=human_key,
        disabled=st.session_state.skip_human_remaining,
    )

    if st.button("Apply critique & revise", type="primary", key=f"revise_{idx}_{st.session_state.completed_revision_rounds}"):
        st.session_state.error = None
        notes = "" if st.session_state.skip_human_remaining else human_notes.strip()
        with st.spinner("Revising episode…"):
            try:
                st.session_state.current_draft = revise_chapter(
                    llm=llm,
                    brief=brief,
                    overall_storyline=plan.overall_storyline,
                    outline=outline,
                    prior_chapter_summaries=st.session_state.prior_summaries,
                    chapter_index_one_based=idx + 1,
                    current_draft=st.session_state.current_draft,
                    critique=crit,
                    human_feedback=notes,
                )
                st.session_state.pending_critique = None
                st.session_state.completed_revision_rounds += 1
            except Exception as e:  # noqa: BLE001
                st.session_state.error = str(e)
        st.rerun()


def _render_done() -> None:
    st.header("Export")
    plan = st.session_state.novel_plan
    brief = st.session_state.story_brief
    assert plan is not None and brief is not None
    md = assemble_novel(brief=brief, plan=plan, chapter_texts=st.session_state.chapter_texts)
    st.success(
        f"**{brief.working_title}** is ready — {len(plan.chapters)} episode script(s) assembled as Markdown."
    )
    st.download_button(
        "Download drama-script.md",
        data=md.encode("utf-8"),
        file_name="drama-script.md",
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )
    with st.expander("Preview full script", expanded=False):
        st.markdown(md[:50000] + ("…" if len(md) > 50000 else ""))
    st.divider()
    st.caption("Want another series?")
    if st.button("Start a new series", use_container_width=True):
        _new_project_dialog()


def main() -> None:
    st.set_page_config(page_title="Drama script writer", page_icon="📺", layout="wide")
    _init_session_defaults()
    _render_sidebar()

    st.title("Drama script writer for episodes")

    phase: Phase = st.session_state.phase
    if phase == "PLANNING":
        _render_planning()
    elif phase == "DRAFTING":
        _render_drafting()
    elif phase == "WRITING":
        _render_writing()
    else:
        _render_done()


if __name__ == "__main__":
    main()
