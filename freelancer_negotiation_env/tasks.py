# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task definitions and deterministic graders for freelancer negotiation.

This module provides:
- Three benchmark tasks (easy, medium, hard) with initial state and expected outcome.
- Deterministic graders that return a score in [0.0, 1.0].

Each grader evaluates:
- Final price quality
- Decision quality (accept/reject behavior)
- Negotiation quality (message quality, efficiency, and scenario handling)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


TaskDifficulty = Literal["easy", "medium", "hard"]
DecisionType = Literal["accept", "reject", "negotiate"]


@dataclass(frozen=True)
class TaskDefinition:
    """Benchmark task specification for the negotiation environment."""

    task_id: str
    difficulty: TaskDifficulty
    title: str
    description: str
    initial_state: dict[str, object]
    expected_outcome: dict[str, object]


@dataclass
class EpisodeResult:
    """Deterministic grading input extracted from an episode run."""

    final_price: float | None
    decision: DecisionType
    conversation_history: list[str]
    step_count: int
    client_type: str


TASKS: dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        difficulty="easy",
        title="Straightforward Normal Client",
        description="Simple client with aligned expectations and moderate budget.",
        initial_state={
            "current_price": 1250.0,
            "client_budget": 1300.0,
            "deadline": "2026-04-30",
            "conversation_history": [
                "client: I have a clear website brief and a budget around $1300.",
                "client: If scope is clear and timeline is realistic, I can move quickly.",
            ],
            "client_type": "normal",
            "revisions": 1,
        },
        expected_outcome={
            "target_decision": "accept",
            "ideal_price": 1250.0,
            "acceptable_price_range": [1150.0, 1350.0],
            "max_steps_for_efficiency": 4,
        },
    ),
    "medium": TaskDefinition(
        task_id="medium",
        difficulty="medium",
        title="Budget-Time Conflict",
        description="Conflicting requirements: low budget and tight deadline.",
        initial_state={
            "current_price": 1400.0,
            "client_budget": 800.0,
            "deadline": "2026-04-05",
            "conversation_history": [
                "client: I need this in one week and my budget is $800.",
                "client: It must include full QA and post-launch support.",
            ],
            "client_type": "cheap",
            "revisions": 2,
        },
        expected_outcome={
            "target_decision": "reject",
            "min_viable_price": 1100.0,
            "max_steps_for_efficiency": 5,
            "quality_signals": ["scope", "timeline", "tradeoff", "budget"],
        },
    ),
    "hard": TaskDefinition(
        task_id="hard",
        difficulty="hard",
        title="Toxic Unclear Client",
        description="Toxic client with unclear requirements and repeated revision pressure.",
        initial_state={
            "current_price": 1700.0,
            "client_budget": 950.0,
            "deadline": "2026-04-08",
            "conversation_history": [
                "client: I want premium work, unlimited revisions, and low cost.",
                "client: Details are flexible, just make it perfect and fast.",
            ],
            "client_type": "toxic",
            "revisions": 3,
        },
        expected_outcome={
            "target_decision": "reject",
            "ideal_price": 1650.0,
            "max_steps_for_efficiency": 6,
            "boundary_keywords": ["scope", "paid", "revision", "contract", "milestone"],
        },
    ),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalized_history(history: list[str]) -> list[str]:
    return [re.sub(r"\s+", " ", msg.strip().lower()) for msg in history if msg.strip()]


def _message_quality_score(history: list[str]) -> float:
    """Score message quality based on relevance and repetition patterns."""
    normalized = _normalized_history(history)
    if not normalized:
        return 0.0

    repetitive_count = 0
    irrelevant_count = 0
    informative_count = 0

    seen: set[str] = set()
    irrelevant_markers = ("asdf", "lorem", "blah", "whatever", "idk", "...", "??")
    informative_markers = ("scope", "timeline", "budget", "price", "deliver", "revision", "contract")

    for msg in normalized:
        if msg in seen:
            repetitive_count += 1
        seen.add(msg)

        if len(msg) < 12 or any(tok in msg for tok in irrelevant_markers):
            irrelevant_count += 1

        if any(tok in msg for tok in informative_markers) or re.search(r"\$?\d+", msg):
            informative_count += 1

    total = len(normalized)
    repetitive_penalty = repetitive_count / total
    irrelevant_penalty = irrelevant_count / total
    informative_ratio = informative_count / total

    score = 0.2 + 0.8 * informative_ratio - 0.6 * repetitive_penalty - 0.5 * irrelevant_penalty
    return _clamp01(score)


def _efficiency_score(step_count: int, max_steps: int) -> float:
    if step_count <= 0:
        return 0.0
    if step_count <= max_steps:
        return 1.0
    overflow = step_count - max_steps
    return _clamp01(1.0 - 0.2 * overflow)


def _price_quality_score(final_price: float | None, ideal_price: float, lo: float, hi: float) -> float:
    if final_price is None:
        return 0.0

    if lo <= final_price <= hi:
        delta = abs(final_price - ideal_price)
        tolerance = max((hi - lo) / 2.0, 1.0)
        return _clamp01(1.0 - (delta / tolerance) * 0.5)

    if final_price < lo:
        gap = (lo - final_price) / max(lo, 1.0)
    else:
        gap = (final_price - hi) / max(hi, 1.0)
    return _clamp01(0.6 - 1.5 * gap)


def _decision_match_score(actual: DecisionType, expected: DecisionType) -> float:
    if actual == expected:
        return 1.0
    if expected == "reject" and actual == "negotiate":
        return 0.4
    if expected == "accept" and actual == "negotiate":
        return 0.5
    return 0.0


def _toxic_handling_score(history: list[str], decision: DecisionType) -> float:
    normalized = " ".join(_normalized_history(history))
    boundary_markers = ("scope", "paid", "revision", "contract", "milestone", "out of scope")
    has_boundaries = any(tok in normalized for tok in boundary_markers)

    decision_component = 1.0 if decision == "reject" else 0.5 if decision == "negotiate" else 0.2
    boundary_component = 1.0 if has_boundaries else 0.2
    return _clamp01(0.6 * decision_component + 0.4 * boundary_component)


def grade_easy_task(result: EpisodeResult) -> float:
    """Grade easy scenario with emphasis on fair acceptance and concise negotiation."""
    task = TASKS["easy"]
    expected = task.expected_outcome

    ideal_price = float(expected["ideal_price"])
    lo, hi = expected["acceptable_price_range"]
    max_steps = int(expected["max_steps_for_efficiency"])

    price_score = _price_quality_score(result.final_price, ideal_price=ideal_price, lo=float(lo), hi=float(hi))
    decision_score = _decision_match_score(result.decision, expected="accept")
    quality_score = 0.6 * _message_quality_score(result.conversation_history) + 0.4 * _efficiency_score(
        result.step_count, max_steps
    )

    return _clamp01(0.45 * price_score + 0.30 * decision_score + 0.25 * quality_score)


def grade_medium_task(result: EpisodeResult) -> float:
    """Grade medium scenario with focus on decision quality under conflicting constraints."""
    task = TASKS["medium"]
    expected = task.expected_outcome

    min_viable = float(expected["min_viable_price"])
    max_steps = int(expected["max_steps_for_efficiency"])

    if result.decision == "reject":
        price_score = 1.0
    else:
        if result.final_price is None:
            price_score = 0.0
        else:
            ratio = (result.final_price - min_viable) / max(min_viable, 1.0)
            price_score = _clamp01(0.4 + ratio)

    decision_score = _decision_match_score(result.decision, expected="reject")

    history_joined = " ".join(_normalized_history(result.conversation_history))
    expected_terms = expected["quality_signals"]
    term_coverage = sum(1 for term in expected_terms if term in history_joined) / max(len(expected_terms), 1)

    quality_score = _clamp01(
        0.45 * _message_quality_score(result.conversation_history)
        + 0.35 * _efficiency_score(result.step_count, max_steps)
        + 0.20 * term_coverage
    )

    return _clamp01(0.35 * price_score + 0.40 * decision_score + 0.25 * quality_score)


def grade_hard_task(result: EpisodeResult) -> float:
    """Grade hard toxic-client scenario with emphasis on boundary-setting behavior."""
    task = TASKS["hard"]
    expected = task.expected_outcome

    ideal_price = float(expected["ideal_price"])
    max_steps = int(expected["max_steps_for_efficiency"])

    # For toxic scenarios, rejecting a bad deal can still be optimal.
    if result.decision == "reject":
        price_score = 1.0
    else:
        price_score = _price_quality_score(result.final_price, ideal_price, lo=1450.0, hi=1800.0)

    decision_score = _decision_match_score(result.decision, expected="reject")

    toxic_quality = _toxic_handling_score(result.conversation_history, result.decision)
    quality_score = _clamp01(
        0.55 * toxic_quality
        + 0.25 * _message_quality_score(result.conversation_history)
        + 0.20 * _efficiency_score(result.step_count, max_steps)
    )

    return _clamp01(0.30 * price_score + 0.30 * decision_score + 0.40 * quality_score)


def grade_task(task_id: str, result: EpisodeResult) -> float:
    """Grade a task deterministically and return a score in [0.0, 1.0]."""
    normalized_id = task_id.strip().lower()
    if normalized_id == "easy":
        return grade_easy_task(result)
    if normalized_id == "medium":
        return grade_medium_task(result)
    if normalized_id == "hard":
        return grade_hard_task(result)
    raise ValueError(f"Unknown task_id: {task_id}")


def get_tasks() -> list[TaskDefinition]:
    """Return all benchmark tasks in deterministic order."""
    return [TASKS["easy"], TASKS["medium"], TASKS["hard"]]
