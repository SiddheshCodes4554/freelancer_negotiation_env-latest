"""Run deterministic LLM inference across all freelancer negotiation benchmark tasks.

Environment variables:
- API_BASE_URL: OpenAI-compatible endpoint URL for the LLM API.
- MODEL_NAME: Model identifier to use for chat completions.
- HF_TOKEN: API token used as OpenAI client api_key.

Optional:
- OPENENV_BASE_URL: Base URL of a running OpenEnv server (default: http://localhost:8000).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from openai import OpenAI

from freelancer_negotiation_env import FreelancerNegotiationAction, FreelancerNegotiationEnv
from freelancer_negotiation_env.models import FreelancerNegotiationAction as ActionModel
from freelancer_negotiation_env.models import NegotiationActionType
from freelancer_negotiation_env.tasks import EpisodeResult, TaskDefinition, get_tasks, grade_task

MAX_STEPS_PER_TASK = 8
RANDOM_SEED = 7


@dataclass
class TaskRunSummary:
    task_id: str
    total_reward: float
    grader_score: float
    steps: int
    decision: str


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_policy_prompt(task: TaskDefinition, observation: dict[str, object], step_index: int) -> str:
    return (
        "You are a freelancer negotiating with a client. "
        "Return ONLY compact JSON with keys message and action_type. "
        "action_type must be one of: negotiate, accept, reject.\n\n"
        f"Task: {task.title} ({task.difficulty})\n"
        f"Task description: {task.description}\n"
        f"Expected outcome: {json.dumps(task.expected_outcome, ensure_ascii=True)}\n"
        f"Step index: {step_index}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=True)}\n"
        "Guidelines:\n"
        "- Keep messages concrete and relevant to price/scope/deadline.\n"
        "- Prefer early resolution when reasonable.\n"
        "- For toxic behavior, set boundaries and avoid agreeing to free unlimited work.\n"
    )


def _extract_action_json(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model response is not valid JSON: {raw_text}")
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")

    action_type = str(payload.get("action_type", "negotiate")).strip().lower()
    if action_type not in {"negotiate", "accept", "reject"}:
        action_type = "negotiate"

    message = str(payload.get("message", "Let's discuss scope, budget, and timeline.")).strip()
    if not message:
        message = "Let's discuss scope, budget, and timeline."

    return {"message": message, "action_type": action_type}


def _llm_action(
    client: OpenAI,
    model_name: str,
    task: TaskDefinition,
    observation: dict[str, object],
    step_index: int,
) -> ActionModel:
    prompt = _build_policy_prompt(task=task, observation=observation, step_index=step_index)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a deterministic negotiation policy. "
                    "Always return only JSON with keys: message, action_type."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = _extract_action_json(content)
    return FreelancerNegotiationAction(
        message=parsed["message"],
        action_type=NegotiationActionType(parsed["action_type"]),
    )


def _observation_to_dict(obs: object) -> dict[str, object]:
    # Pydantic model_dump is available for Pydantic v2 models.
    if hasattr(obs, "model_dump"):
        return obs.model_dump()  # type: ignore[no-any-return]
    return {
        "client_message": getattr(obs, "client_message", ""),
        "negotiation_state": getattr(obs, "negotiation_state", {}),
        "conversation_history": getattr(obs, "conversation_history", []),
        "done": getattr(obs, "done", False),
        "reward": getattr(obs, "reward", 0.0),
    }


def run_task(
    env: FreelancerNegotiationEnv,
    llm_client: OpenAI,
    model_name: str,
    task: TaskDefinition,
) -> TaskRunSummary:
    reset_result = env.reset()
    obs = reset_result.observation

    total_reward = 0.0
    last_decision: str = "negotiate"
    final_price: float | None = None
    step_count = 0

    for step_index in range(1, MAX_STEPS_PER_TASK + 1):
        obs_dict = _observation_to_dict(obs)
        action = _llm_action(
            client=llm_client,
            model_name=model_name,
            task=task,
            observation=obs_dict,
            step_index=step_index,
        )

        step_result = env.step(action)
        obs = step_result.observation
        step_count = step_index
        last_decision = action.action_type.value
        total_reward += float(step_result.reward or 0.0)

        state_obj = getattr(obs, "negotiation_state", None)
        if hasattr(state_obj, "current_price"):
            final_price = float(state_obj.current_price)

        done = bool(getattr(obs, "done", False) or step_result.done)
        if done:
            break

    history = list(getattr(obs, "conversation_history", []))

    episode_result = EpisodeResult(
        final_price=final_price,
        decision=last_decision,  # type: ignore[arg-type]
        conversation_history=history,
        step_count=step_count,
        client_type=str(getattr(obs, "metadata", {}).get("client_type", "normal")),
    )
    grader_score = grade_task(task.task_id, episode_result)

    return TaskRunSummary(
        task_id=task.task_id,
        total_reward=round(total_reward, 3),
        grader_score=round(grader_score, 4),
        steps=step_count,
        decision=last_decision,
    )


def main() -> None:
    random.seed(RANDOM_SEED)

    api_base_url = _required_env("API_BASE_URL")
    model_name = _required_env("MODEL_NAME")
    hf_token = _required_env("HF_TOKEN")

    openenv_base_url = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")

    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token)

    summaries: list[TaskRunSummary] = []
    with FreelancerNegotiationEnv(base_url=openenv_base_url) as env:
        for task in get_tasks():
            summary = run_task(env=env, llm_client=llm_client, model_name=model_name, task=task)
            summaries.append(summary)
            print(
                f"task={summary.task_id} steps={summary.steps} decision={summary.decision} "
                f"reward={summary.total_reward:.3f} score={summary.grader_score:.4f}"
            )

    final_score = sum(s.grader_score for s in summaries) / max(len(summaries), 1)
    total_reward = sum(s.total_reward for s in summaries)

    print(f"final_score={final_score:.4f}")
    print(f"total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
