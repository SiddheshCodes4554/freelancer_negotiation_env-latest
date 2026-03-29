# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Freelancer-client negotiation environment implementation.

This environment simulates price/deadline negotiation with deterministic,
rule-based client behavior for reliable grading and evaluation.
"""

import re
import os
import random
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FreelancerNegotiationAction, FreelancerNegotiationObservation
except ImportError:
    from models import FreelancerNegotiationAction, FreelancerNegotiationObservation


_COMMUNICATION_SCORE_CACHE: dict[str, float] = {}
_EVAL_CLIENT = None


def _build_eval_client():
    """Build OpenAI client for communication scoring when env vars are available."""
    global _EVAL_CLIENT
    if _EVAL_CLIENT is not None:
        return _EVAL_CLIENT

    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")
    if not api_base_url or not model_name or not hf_token:
        return None

    try:
        from openai import OpenAI

        _EVAL_CLIENT = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=5.0)
        return _EVAL_CLIENT
    except Exception:
        return None


def evaluate_communication(message: str) -> float:
    """Score communication quality in [0.0, 1.0] using an OpenAI-compatible endpoint.

    Criteria: politeness, professionalism, clarity, and negotiation logic.
    Falls back to 0.5 if parsing or API calls fail.
    """
    normalized = re.sub(r"\s+", " ", message.strip())
    if not normalized:
        return 0.5

    cache_key = normalized.lower()
    if cache_key in _COMMUNICATION_SCORE_CACHE:
        return _COMMUNICATION_SCORE_CACHE[cache_key]

    client = _build_eval_client()
    model_name = os.getenv("MODEL_NAME")
    if client is None or not model_name:
        score = 0.5
        _COMMUNICATION_SCORE_CACHE[cache_key] = score
        return score

    try:
        prompt = (
            "Rate this negotiation message from 0 to 1 based on professionalism, clarity, and politeness. "
            "Only return a number.\n\n"
            f"Message: {normalized[:500]}"
        )

        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            top_p=1,
            max_tokens=6,
            messages=[
                {"role": "system", "content": "Return only a number between 0 and 1."},
                {"role": "user", "content": prompt},
            ],
            timeout=5.0,
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", content)
        if not match:
            raise ValueError("No numeric score found")

        score = float(match.group(0))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    # Keep cache bounded for long-running episodes.
    if len(_COMMUNICATION_SCORE_CACHE) >= 256:
        _COMMUNICATION_SCORE_CACHE.clear()
    _COMMUNICATION_SCORE_CACHE[cache_key] = score
    return score


class FreelancerNegotiationEnvironment(Environment):
    """
    A rule-based negotiation simulator between freelancer (agent) and client.

    State maintained by the environment:
    - current_price
    - client_budget
    - deadline
    - conversation_history
    - client_type (cheap, normal, premium, toxic)
    - strategy_type (aggressive, balanced, cooperative)
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    @dataclass(frozen=True)
    class Scenario:
        """Static scenario specification for deterministic reset behavior."""

        client_type: str
        strategy_type: str
        client_budget: float
        current_price: float
        ideal_price: float
        minimum_price: float
        deadline: str
        opening_message: str

    @dataclass(frozen=True)
    class DealRecord:
        """Compact memory record for completed episodes."""

        client_type: str
        final_price: float
        success: bool
        number_of_steps: int

    MAX_STEPS: int = 8

    def __init__(self):
        """Initialize negotiation state and deterministic scenario rotation."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count: int = 0

        # Mutable episode state
        self.current_price: float = 0.0
        self.client_budget: float = 0.0
        self.deadline: str = ""
        self.ideal_price: float = 0.0
        self.minimum_price: float = 0.0
        self.current_offer: float = 0.0
        self.client_type: str = "normal"
        self.strategy_type: str = "balanced"
        self.revisions: int = 1
        self.done: bool = False
        self.last_client_message: str = ""
        self.conversation_history: list[str] = []
        self.deal_memory: list[FreelancerNegotiationEnvironment.DealRecord] = []
        self.memory_summary: list[dict[str, object]] = []
        self._episode_memory_recorded: bool = False

        # Deterministic scenario bank for stable grading.
        self._scenarios: list[FreelancerNegotiationEnvironment.Scenario] = [
            self.Scenario(
                client_type="cheap",
                strategy_type="aggressive",
                client_budget=700.0,
                current_price=950.0,
                ideal_price=900.0,
                minimum_price=740.0,
                deadline="2026-04-18",
                opening_message="Your quote is too high for me. Can you do this for $700?",
            ),
            self.Scenario(
                client_type="normal",
                strategy_type="balanced",
                client_budget=1200.0,
                current_price=1300.0,
                ideal_price=1350.0,
                minimum_price=980.0,
                deadline="2026-04-25",
                opening_message="I like your portfolio. Can we align on price and timeline?",
            ),
            self.Scenario(
                client_type="premium",
                strategy_type="cooperative",
                client_budget=2600.0,
                current_price=2100.0,
                ideal_price=2450.0,
                minimum_price=1780.0,
                deadline="2026-05-03",
                opening_message="Quality matters most. Share your best proposal.",
            ),
            self.Scenario(
                client_type="toxic",
                strategy_type="balanced",
                client_budget=900.0,
                current_price=1400.0,
                ideal_price=1500.0,
                minimum_price=1080.0,
                deadline="2026-04-15",
                opening_message="I need premium quality fast, unlimited revisions, and low cost.",
            ),
        ]

    def _load_scenario(self) -> None:
        """Load a deterministic scenario based on reset count."""
        scenario = self._scenarios[self._reset_count % len(self._scenarios)]

        self.client_type = scenario.client_type
        self.strategy_type = scenario.strategy_type
        self.client_budget = scenario.client_budget
        self.current_price = scenario.current_price
        self.ideal_price = scenario.ideal_price
        self.minimum_price = scenario.minimum_price
        self.current_offer = scenario.current_price
        self.deadline = scenario.deadline
        self.revisions = 1
        self.done = False
        self.last_client_message = scenario.opening_message
        self.conversation_history = [f"client: {scenario.opening_message}"]
        self._episode_memory_recorded = False
        self._refresh_memory_summary()

    def _refresh_memory_summary(self) -> None:
        """Keep memory summary as the last 3 completed deals."""
        self.memory_summary = [
            {
                "client_type": item.client_type,
                "final_price": item.final_price,
                "success": item.success,
                "number_of_steps": item.number_of_steps,
            }
            for item in self.deal_memory[-3:]
        ]

    def _memory_guidance_for_client(self) -> str:
        """Derive deterministic guidance from similar historical deals.

        Returns one of: neutral, cautious, repeat.
        """
        similar = [item for item in self.deal_memory if item.client_type == self.client_type]
        if not similar:
            return "neutral"

        recent_similar = similar[-3:]
        success_rate = sum(1 for item in recent_similar if item.success) / len(recent_similar)
        if success_rate >= 0.6:
            return "repeat"
        return "cautious"

    def _record_episode_memory(self, success: bool) -> None:
        """Append current episode result to bounded memory."""
        if self._episode_memory_recorded:
            return

        record = self.DealRecord(
            client_type=self.client_type,
            final_price=round(self.current_price, 2),
            success=success,
            number_of_steps=self._state.step_count,
        )
        self.deal_memory.append(record)
        self.deal_memory = self.deal_memory[-5:]
        self._episode_memory_recorded = True
        self._refresh_memory_summary()

    def _build_observation(self, reward: float, info: dict[str, object]) -> FreelancerNegotiationObservation:
        """Create a structured observation from current environment state."""
        return FreelancerNegotiationObservation(
            client_message=self.last_client_message,
            negotiation_state={
                "current_price": round(self.current_price, 2),
                "deadline": self.deadline,
                "revisions": self.revisions,
            },
            conversation_history=list(self.conversation_history),
            memory_summary=list(self.memory_summary),
            done=self.done,
            reward=round(reward, 3),
            metadata={
                "info": info,
                "client_type": self.client_type,
                "client_budget": self.client_budget,
                "ideal_price": self.ideal_price,
                "minimum_price": self.minimum_price,
                "current_offer": round(self.current_offer, 2),
                "strategy_type": self.strategy_type,
                "memory_summary": self.memory_summary,
                "step": self._state.step_count,
            },
        )

    def _interpret_action_by_strategy(
        self,
        action_type: str,
        proposed_price: float,
    ) -> tuple[str, float, dict[str, object]]:
        """Translate raw action into a strategy-aware effective action and price."""
        memory_guidance = self._memory_guidance_for_client()
        details: dict[str, object] = {
            "strategy_type": self.strategy_type,
            "memory_guidance": memory_guidance,
            "input_action_type": action_type,
            "input_price": round(proposed_price, 2),
        }

        effective_action = action_type
        adjusted_price = proposed_price

        if self.strategy_type == "aggressive":
            floor = max(self.ideal_price * 0.92, self.current_price)
            adjusted_price = max(proposed_price * 1.08, floor)
            if action_type == "accept" and adjusted_price < self.ideal_price * 0.9:
                effective_action = "reject"
                details["strategy_override"] = "reject_low_offer"

        elif self.strategy_type == "cooperative":
            target_close_price = max(self.client_budget * 0.9, self.ideal_price * 0.82)
            adjusted_price = min(proposed_price * 0.95, self.current_price)
            adjusted_price = max(adjusted_price, target_close_price)

            # Cooperative strategy prioritizes closure after a few turns.
            if action_type == "negotiate" and self._state.step_count >= 3 and self.current_price >= target_close_price:
                effective_action = "accept"
                details["strategy_override"] = "accelerate_closure"

        else:  # balanced
            adjusted_price = (proposed_price * 0.7) + (self.current_price * 0.3)

        if memory_guidance == "cautious":
            adjusted_price = max(adjusted_price, self.ideal_price * 0.9)
            if effective_action == "accept" and adjusted_price < self.ideal_price * 0.95:
                effective_action = "negotiate"
                details["memory_override"] = "avoid_early_accept"

        elif memory_guidance == "repeat":
            if self.strategy_type == "aggressive":
                adjusted_price *= 1.03
            elif self.strategy_type == "cooperative" and effective_action == "negotiate" and self._state.step_count >= 2:
                effective_action = "accept"
                details["memory_override"] = "repeat_successful_closure"

        details["effective_action_type"] = effective_action
        details["effective_price"] = round(adjusted_price, 2)
        return effective_action, adjusted_price, details

    @staticmethod
    def _validate_action(action: FreelancerNegotiationAction) -> tuple[bool, float, dict[str, object]]:
        """Validate incoming action and return (is_valid, penalty, details)."""
        details: dict[str, object] = {}

        if action is None:
            return False, -4.0, {"error": "missing_action"}

        action_type = getattr(getattr(action, "action_type", None), "value", None)
        if action_type not in {"negotiate", "accept", "reject"}:
            return False, -4.0, {"error": "invalid_action_type", "received": str(action_type)}

        message = str(getattr(action, "message", "")).strip()
        if not message:
            return False, -2.0, {"error": "empty_message"}

        details["normalized_action_type"] = action_type
        details["normalized_message"] = message
        return True, 0.0, details

    @staticmethod
    def _detect_negotiation_intent(message: str) -> str:
        """Infer coarse negotiation intent from free-form text."""
        text = message.lower()
        if any(token in text for token in ("accept", "agreed", "deal", "let's proceed")):
            return "accept"
        if any(token in text for token in ("reject", "decline", "cannot proceed", "walk away")):
            return "reject"
        if any(token in text for token in ("discount", "lower", "reduce", "budget")):
            return "concession"
        if any(token in text for token in ("scope", "milestone", "timeline", "deadline", "revision")):
            return "structured"
        return "neutral"

    def _shift_deadline(self, days: int) -> None:
        """Deterministically push deadline by N days when delays occur."""
        try:
            parsed = datetime.strptime(self.deadline, "%Y-%m-%d")
            self.deadline = (parsed + timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            # Keep deadline unchanged if format is unexpected.
            return

    @staticmethod
    def _extract_price_from_text(text: str) -> float | None:
        """Extract the first price from text for INR-style formats.

        Supported formats include:
        - ₹5000
        - Rs 3,000
        - INR 12,500.50
        - Rs.4500
        """
        pattern = re.compile(
            r"(?:\bINR\b|\bRs\.?\b|₹)\s*([0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]+)?)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None

        numeric_str = match.group(1).replace(",", "")
        try:
            return float(numeric_str)
        except ValueError:
            return None

    def _deterministic_rng(self, channel: str) -> random.Random:
        """Create deterministic RNG scoped to episode, step, and channel."""
        seed_text = f"{self._state.episode_id}:{self._state.step_count}:{self.client_type}:{channel}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed)

    @staticmethod
    def _inr(price: float) -> str:
        """Format integer-ish value as INR text for realistic client messages."""
        return f"₹{int(round(price))}"

    def _price_band(self, proposed_price: float) -> str:
        """Classify offer as low/reasonable/high relative to client profile."""
        if self.client_type == "cheap":
            reasonable_cap = self.client_budget * 1.00
        elif self.client_type == "normal":
            reasonable_cap = self.client_budget * 1.10
        elif self.client_type == "premium":
            reasonable_cap = self.client_budget * 1.35
        else:  # toxic
            reasonable_cap = self.client_budget * 0.95

        low_cutoff = self.minimum_price * 0.85

        if proposed_price > reasonable_cap:
            return "high"
        if proposed_price < low_cutoff:
            return "low"
        return "reasonable"

    def _simulate_client_message(self, band: str, counter_offer: float, accepted: bool) -> str:
        """Generate realistic client messages with deterministic bounded variation."""
        rng = self._deterministic_rng("message")

        if self.client_type == "cheap":
            if accepted and band == "low":
                choices = [
                    f"This feels low for your quality, but I can approve {self._inr(counter_offer)}.",
                    f"It is cheaper than expected, still fine. Let us close at {self._inr(counter_offer)}.",
                ]
            elif accepted:
                choices = [
                    f"Okay, this works. Let us finalize at {self._inr(counter_offer)}.",
                    f"Deal. I can proceed at {self._inr(counter_offer)}.",
                ]
            else:
                choices = [
                    f"That is too high, my budget is {self._inr(self.client_budget)}. Can you do {self._inr(counter_offer)}?",
                    f"I cannot stretch beyond {self._inr(counter_offer)}. Please reduce your price.",
                ]

        elif self.client_type == "premium":
            if accepted and band == "low":
                choices = [
                    f"This is lower than expected for premium delivery, but I accept {self._inr(counter_offer)}.",
                    f"Price is unusually low, still acceptable. We can proceed at {self._inr(counter_offer)}.",
                ]
            elif accepted:
                choices = [
                    f"Quality-first approach looks good. I accept {self._inr(counter_offer)}.",
                    f"This is fair for quality. Let us confirm at {self._inr(counter_offer)}.",
                ]
            else:
                choices = [
                    f"I value quality, but can we settle around {self._inr(counter_offer)}?",
                    f"I can move quickly if we agree at {self._inr(counter_offer)}.",
                ]

        elif self.client_type == "toxic":
            if accepted and band == "low":
                choices = [
                    f"Fine, we can proceed at {self._inr(counter_offer)}. I still expect extra polish.",
                    f"Okay, accepted at {self._inr(counter_offer)}. Include quick support if possible.",
                ]
            elif accepted:
                choices = [
                    f"Alright, accepted at {self._inr(counter_offer)}. I need 2 revisions included.",
                    f"Let us close this at {self._inr(counter_offer)}, but I expect more flexibility.",
                ]
            else:
                choices = [
                    f"I need 2 revisions included and my budget is still {self._inr(self.client_budget)}.",
                    f"Price is high for me. Also include extra work; I can do only {self._inr(counter_offer)}.",
                ]

        else:  # normal
            if accepted and band == "low":
                choices = [
                    f"This is lower than expected, but acceptable. Let us proceed at {self._inr(counter_offer)}.",
                    f"Your offer is quite low. I can still confirm at {self._inr(counter_offer)}.",
                ]
            elif accepted:
                choices = [
                    f"That looks fair. I am happy to proceed at {self._inr(counter_offer)}.",
                    f"Reasonable offer. Let us finalize at {self._inr(counter_offer)}.",
                ]
            else:
                choices = [
                    f"That is a bit high. Can you do it for {self._inr(counter_offer)}?",
                    f"I understand your quote, but my budget points to {self._inr(counter_offer)}.",
                ]

        return choices[rng.randrange(len(choices))]

    def _client_counter_offer(self, proposed_price: float) -> tuple[str, float, bool]:
        """Generate realistic deterministic client response by client type.

        Behavior:
        - High offer: counter lower.
        - Reasonable offer: accept.
        - Very low offer: suspicious but accepts.
        - Toxic client adds revisions and delays.
        """
        rng = self._deterministic_rng("counter")
        band = self._price_band(proposed_price)

        # Bounded deterministic variation.
        variation = 1.0 + rng.uniform(-0.03, 0.03)

        if self.client_type == "cheap":
            if band == "high":
                counter = min(self.client_budget * 0.98, proposed_price * (0.80 * variation))
                accepted = False
            elif band == "reasonable":
                counter = proposed_price
                accepted = True
            else:  # low
                counter = proposed_price
                accepted = True

        elif self.client_type == "premium":
            if band == "high":
                counter = min(self.client_budget * 1.30, proposed_price * (0.94 * variation))
                accepted = False
            elif band == "reasonable":
                counter = proposed_price
                accepted = True
            else:  # low
                counter = proposed_price
                accepted = True

        elif self.client_type == "toxic":
            self.revisions += 1
            self._shift_deadline(days=1)
            if band == "high":
                counter = min(self.client_budget * 0.92, proposed_price * (0.76 * variation))
                accepted = False
            elif band == "reasonable":
                # Toxic clients delay decisions even for reasonable prices.
                counter = min(self.client_budget * 0.90, proposed_price * (0.88 * variation))
                accepted = False
            else:  # low
                counter = proposed_price
                accepted = True

        else:  # normal
            if band == "high":
                counter = min(self.client_budget * 1.05, proposed_price * (0.90 * variation))
                accepted = False
            elif band == "reasonable":
                counter = proposed_price
                accepted = True
            else:  # low
                counter = proposed_price
                accepted = True

        self.current_price = round(counter, 2)
        message = self._simulate_client_message(band=band, counter_offer=self.current_price, accepted=accepted)
        return message, self.current_price, accepted

    def _is_repeated_message(self, message: str) -> bool:
        """Detect repeated freelancer messages."""
        normalized = re.sub(r"\s+", " ", message.strip().lower())

        freelancer_messages = [
            entry.replace("freelancer: ", "", 1).strip().lower()
            for entry in self.conversation_history
            if entry.startswith("freelancer: ")
        ]
        # step() appends the current message before reward computation, so compare
        # against prior freelancer turns only.
        if freelancer_messages and normalized in freelancer_messages[:-1]:
            return True

        return False

    def _is_irrelevant_message(self, message: str) -> bool:
        """Detect negotiation-irrelevant freelancer messages."""
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        if len(normalized) < 8:
            return True

        noise_markers = ("lorem", "asdf", "blah", "idk", "random", "irrelevant")
        if any(marker in normalized for marker in noise_markers):
            return True

        negotiation_markers = (
            "price",
            "budget",
            "scope",
            "deadline",
            "deliver",
            "revision",
            "milestone",
            "contract",
            "$",
        )
        return not any(marker in normalized for marker in negotiation_markers) and self._extract_price_from_text(normalized) is None

    def _handled_toxic_client_well(self, action_message: str, action_type: str) -> bool:
        """Detect constructive boundary-setting behavior in toxic negotiations."""
        if self.client_type != "toxic" or action_type != "negotiate":
            return False

        normalized = action_message.lower()
        boundary_markers = ("scope", "additional", "extra", "paid", "revision", "rate", "contract")
        asked_free_work = "free" in normalized and "not" not in normalized
        return any(token in normalized for token in boundary_markers) and not asked_free_work

    def _deal_close_to_ideal(self) -> bool:
        """Check if final agreed price is close to target ideal price."""
        delta_ratio = abs(self.current_price - self.ideal_price) / max(self.ideal_price, 1.0)
        return delta_ratio <= 0.08

    def _deal_too_cheap(self) -> bool:
        """Check if accepted deal undervalues the freelancer significantly."""
        return self.current_price < self.ideal_price * 0.82

    def _lost_client_unnecessarily(self, action_type: str) -> bool:
        """Penalize avoidable walk-aways when deal zone still exists."""
        if action_type != "reject":
            return False
        return self.current_price >= self.ideal_price * 0.9 and self._state.step_count < self.MAX_STEPS

    def _compute_reward(
        self,
        accepted: bool,
        action_type: str,
        action_message: str,
        previous_offer: float,
    ) -> tuple[float, dict[str, object]]:
        """Compute dense reward as the sum of all configured components."""
        reward = 0.0
        components: dict[str, object] = {}

        # Track required variables explicitly.
        components["ideal_price"] = round(self.ideal_price, 2)
        components["minimum_price"] = round(self.minimum_price, 2)
        components["client_budget"] = round(self.client_budget, 2)
        components["current_offer"] = round(self.current_offer, 2)
        components["number_of_steps"] = self._state.step_count

        # 2) Progress reward (per step)
        prev_dist = abs(previous_offer - self.ideal_price)
        new_dist = abs(self.current_offer - self.ideal_price)
        if new_dist < prev_dist - 1e-6:
            reward += 2.0
            components["progress_reward"] = 2.0
        elif new_dist > prev_dist + 1e-6:
            reward -= 2.0
            components["progress_reward"] = -2.0
        else:
            components["progress_reward"] = 0.0

        # 5) Communication quality (optional)
        communication_score = evaluate_communication(action_message)
        communication_bonus = communication_score * 3.0
        reward += communication_bonus
        components["communication_score"] = round(communication_score, 3)
        components["communication_bonus"] = round(communication_bonus, 3)

        # 6) Penalties
        if self._is_repeated_message(action_message):
            reward -= 3.0
            components["repeated_message_penalty"] = -3.0
        if self._is_irrelevant_message(action_message):
            reward -= 2.0
            components["irrelevant_message_penalty"] = -2.0

        # Optional existing bonus retained for toxic-client boundary handling.
        if self._handled_toxic_client_well(action_message=action_message, action_type=action_type):
            reward += 5.0
            components["toxic_handling_bonus"] = 5.0

        # 1) Deal quality (final step)
        if self.done and accepted:
            final_ratio = self.current_offer / max(self.ideal_price, 1.0)
            if self.current_offer < self.minimum_price:
                reward -= 10.0
                components["deal_quality"] = -10.0
            elif final_ratio >= 0.9:
                reward += 10.0
                components["deal_quality"] = 10.0
            elif final_ratio >= 0.7:
                reward += 5.0
                components["deal_quality"] = 5.0
            else:
                components["deal_quality"] = 0.0

        # 3) Efficiency (terminal)
        if self.done:
            if self._state.step_count <= 3:
                reward += 3.0
                components["efficiency_bonus"] = 3.0
            elif self._state.step_count <= 5:
                reward += 1.0
                components["efficiency_bonus"] = 1.0

        # 4) Decision quality (terminal)
        if self.done:
            bad_deal = self.current_offer < self.minimum_price
            if action_type == "reject":
                if bad_deal:
                    reward += 5.0
                    components["decision_quality"] = 5.0
                else:
                    reward -= 6.0
                    components["decision_quality"] = -6.0
            elif accepted and bad_deal:
                reward -= 8.0
                components["decision_quality"] = -8.0

        components["total"] = round(reward, 3)
        return reward, components

    def reset(self) -> FreelancerNegotiationObservation:
        """
        Reset the environment.

        Returns:
            Initial FreelancerNegotiationObservation for a deterministic client scenario
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._load_scenario()

        info = {
            "event": "reset",
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
        }
        return self._build_observation(reward=0.0, info=info)

    def step(self, action: FreelancerNegotiationAction) -> FreelancerNegotiationObservation:  # type: ignore[override]
        """
        Execute one negotiation step using a deterministic multi-stage pipeline.

        Args:
            action: Agent action containing utterance and high-level intent

        Returns:
            Updated FreelancerNegotiationObservation with reward/done and info metadata
        """
        if self.done:
            return self._build_observation(
                reward=0.0,
                info={"event": "already_done", "message": "Episode already finished. Call reset()."},
            )

        self._state.step_count += 1

        is_valid, validation_penalty, validation_details = self._validate_action(action)
        if not is_valid:
            self.last_client_message = "I could not process that response. Please send a valid negotiation action."
            self.conversation_history.append(f"freelancer: {getattr(action, 'message', '')}")
            self.conversation_history.append(f"client: {self.last_client_message}")

            if self._state.step_count >= self.MAX_STEPS:
                self.done = True

            info = {
                "event": "step",
                "error": "invalid_action",
                "validation": validation_details,
                "action_type": "invalid",
                "strategy_type": self.strategy_type,
                "current_price": round(self.current_price, 2),
                "current_offer": round(self.current_offer, 2),
                "client_budget": self.client_budget,
                "ideal_price": self.ideal_price,
                "minimum_price": self.minimum_price,
                "deadline": self.deadline,
                "revisions": self.revisions,
                "client_type": self.client_type,
                "memory_summary": self.memory_summary,
                "reward_components": {"invalid_action_penalty": validation_penalty, "total": validation_penalty},
            }
            return self._build_observation(reward=validation_penalty, info=info)

        self.conversation_history.append(f"freelancer: {action.message}")
        previous_offer = self.current_offer

        # Parse message content.
        negotiation_intent = self._detect_negotiation_intent(action.message)
        proposed_price = self._extract_price_from_text(action.message)
        if proposed_price is None:
            proposed_price = self.current_price

        # Update negotiation state with the new offer.
        if action.action_type.value == "negotiate" and proposed_price is not None:
            self.current_offer = proposed_price
            self.current_price = proposed_price

        effective_action, effective_price, strategy_details = self._interpret_action_by_strategy(
            action_type=action.action_type.value,
            proposed_price=proposed_price,
        )

        # Intent can refine interpretation deterministically.
        if effective_action == "negotiate" and negotiation_intent == "accept" and self.current_offer >= self.minimum_price:
            effective_action = "accept"
            strategy_details["intent_override"] = "accept_signal"
        elif effective_action == "negotiate" and negotiation_intent == "reject":
            effective_action = "reject"
            strategy_details["intent_override"] = "reject_signal"

        accepted = False

        if effective_action == "accept":
            accepted = True
            self.done = True
            self.last_client_message = "Confirmed. We have a deal."
            self.current_price = effective_price
        elif effective_action == "reject":
            self.done = True
            self.last_client_message = "Understood. I will look for another freelancer."
        else:
            client_message, counter_offer, accepted = self._client_counter_offer(effective_price)
            self.last_client_message = client_message
            self.current_price = counter_offer
            self.done = accepted

        if self._state.step_count >= self.MAX_STEPS and not self.done:
            self.done = True
            self.last_client_message = "We are out of time. Let us pause negotiations here."

        self.current_offer = self.current_price

        if self.done:
            self._record_episode_memory(success=accepted)

        self.conversation_history.append(f"client: {self.last_client_message}")

        reward, reward_components = self._compute_reward(
            accepted=accepted,
            action_type=effective_action,
            action_message=action.message,
            previous_offer=previous_offer,
        )
        if validation_penalty:
            reward += validation_penalty
            reward_components["validation_penalty"] = validation_penalty
            reward_components["total"] = round(reward, 3)

        info = {
            "event": "step",
            "accepted": accepted,
            "action_type": effective_action,
            "intent": negotiation_intent,
            "validation": validation_details,
            "strategy_type": self.strategy_type,
            "current_price": round(self.current_price, 2),
            "current_offer": round(self.current_offer, 2),
            "client_budget": self.client_budget,
            "ideal_price": self.ideal_price,
            "minimum_price": self.minimum_price,
            "deadline": self.deadline,
            "revisions": self.revisions,
            "client_type": self.client_type,
            "memory_summary": self.memory_summary,
            "strategy_details": strategy_details,
            "reward_components": reward_components,
        }

        return self._build_observation(reward=reward, info=info)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
