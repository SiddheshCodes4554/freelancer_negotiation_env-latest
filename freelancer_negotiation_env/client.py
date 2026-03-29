# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Freelancer Negotiation Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FreelancerNegotiationAction, FreelancerNegotiationObservation


class FreelancerNegotiationEnv(
    EnvClient[FreelancerNegotiationAction, FreelancerNegotiationObservation, State]
):
    """
    Client for the Freelancer Negotiation Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with FreelancerNegotiationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(FreelancerNegotiationAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FreelancerNegotiationEnv.from_docker_image("freelancer_negotiation_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FreelancerNegotiationAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FreelancerNegotiationAction) -> Dict:
        """
        Convert FreelancerNegotiationAction to JSON payload for step message.

        Args:
            action: FreelancerNegotiationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
            "action_type": action.action_type.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FreelancerNegotiationObservation]:
        """
        Parse server response into StepResult[FreelancerNegotiationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FreelancerNegotiationObservation
        """
        obs_data = payload.get("observation", {})
        observation = FreelancerNegotiationObservation(
            client_message=obs_data.get("client_message", ""),
            negotiation_state=obs_data.get(
                "negotiation_state",
                {"current_price": 0.0, "deadline": "", "revisions": 0},
            ),
            conversation_history=obs_data.get("conversation_history", []),
            memory_summary=obs_data.get("memory_summary", []),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
