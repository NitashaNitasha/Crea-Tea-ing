"""Planning agent: one follow-up question or a finalized StoryBrief."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from schemas.schemas import PlanningFinalize, PlanningQuestion, StoryBrief

PLANNING_SYSTEM = """You are a fiction planning assistant. Your job is to learn enough from the author to write a strong novel brief.

Rules:
- Either ask exactly ONE focused follow-up question (no preamble, no numbered lists of multiple questions), OR finalize with a complete StoryBrief.
- If you still lack important information but the author has already answered many rounds, infer reasonable defaults in the brief and note assumptions in hard_constraints.
- When finalizing, fill every required StoryBrief field with specific, actionable content.
- Respect the author's genre, tone, and any hard constraints they stated.
"""


class _PlanningStructuredOutput(BaseModel):
    """Flat schema for LLM structured output (OpenAI response_format)."""

    kind: Literal["question", "finalize"] = Field(
        ...,
        description="question = ask one follow-up; finalize = emit full StoryBrief",
    )
    next_question: str | None = Field(
        default=None,
        description="Required when kind is question: exactly one focused question",
    )
    brief: StoryBrief | None = Field(
        default=None,
        description="Required when kind is finalize: complete story brief",
    )

    @model_validator(mode="after")
    def _consistent(self) -> _PlanningStructuredOutput:
        if self.kind == "question":
            if not (self.next_question or "").strip():
                raise ValueError("next_question is required when kind is question")
        else:
            if self.brief is None:
                raise ValueError("brief is required when kind is finalize")
        return self


def run_planning_turn(
    *,
    llm: BaseChatModel,
    planning_messages: Sequence[tuple[str, str]],
    initial_author_prompt: str,
    questions_asked: int,
    max_questions: int = 10,
) -> PlanningQuestion | PlanningFinalize:
    """
    planning_messages: (role, content) with role in {\"user\", \"assistant\"}.
    questions_asked: number of assistant questions already asked in this session.
    max_questions: after this many questions, you must finalize (enforced in prompt).
    """
    structured = llm.with_structured_output(_PlanningStructuredOutput)
    must_finalize = questions_asked >= max_questions
    cap_note = (
        "You have reached the question budget: you MUST finalize now with the best StoryBrief you can "
        "from the conversation (use sensible defaults for missing details and mention them in hard_constraints)."
        if must_finalize
        else f"You may ask at most one more question before you must finalize; question budget is {max_questions} "
        f"and {questions_asked} have been asked so far. If you already have enough to draft, finalize."
    )
    sys = f"{PLANNING_SYSTEM}\n\n{cap_note}"
    msgs: list[BaseMessage] = [
        SystemMessage(content=sys),
        HumanMessage(
            content=(
                "Initial author request:\n"
                f"{initial_author_prompt.strip()}\n\n"
                "Continue the planning conversation below."
            )
        ),
    ]
    for role, content in planning_messages:
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    raw: _PlanningStructuredOutput = structured.invoke(msgs)
    if raw.kind == "question":
        return PlanningQuestion(next_question=(raw.next_question or "").strip())
    assert raw.brief is not None
    return PlanningFinalize(brief=raw.brief)
