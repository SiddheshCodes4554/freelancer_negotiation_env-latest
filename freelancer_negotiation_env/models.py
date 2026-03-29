# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the freelancer negotiation reinforcement learning environment.

These schemas define the interaction contract between agent and environment:
- The action sent by the agent on each step.
- The observation returned by the environment after each step.
"""

from enum import Enum

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class NegotiationActionType(str, Enum):
    """Supported high-level decision types for each negotiation turn."""

    NEGOTIATE = "negotiate"
    ACCEPT = "accept"
    REJECT = "reject"


class NegotiationState(BaseModel):
    """Current structured state of the deal being negotiated."""

    current_price: float = Field(..., description="Current proposed price for the project.")
    deadline: str = Field(..., description="Current proposed deadline (for example, '2026-04-15').")
    revisions: int = Field(..., ge=0, description="Number of revisions included in the current offer.")


class FreelancerNegotiationAction(Action):
    """Action emitted by the agent for a single negotiation step."""

    message: str = Field(..., min_length=1, description="Natural-language message the agent sends to the client.")
    action_type: NegotiationActionType = Field(
        ..., description="High-level intent of the action: negotiate, accept, or reject."
    )


class FreelancerNegotiationObservation(Observation):
    """Observation returned after each step in the freelancer negotiation environment."""

    client_message: str = Field(..., description="Latest message received from the client.")
    negotiation_state: NegotiationState = Field(..., description="Current structured negotiation state.")
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Chronological conversation transcript as plain text messages.",
    )
    memory_summary: list[dict[str, object]] = Field(
        default_factory=list,
        description="Summary of the last 3 completed deals for memory-aware policies.",
    )
    done: bool = Field(..., description="Whether the negotiation episode has terminated.")
