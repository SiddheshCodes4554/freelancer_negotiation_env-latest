"""Submission inference runner for freelancer negotiation benchmark.

This script is intentionally formatted to satisfy submission logging constraints:
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import OpenAI

from freelancer_negotiation_env import FreelancerNegotiationAction, FreelancerNegotiationEnv
from freelancer_negotiation_env.models import FreelancerNegotiationAction as ActionModel
from freelancer_negotiation_env.models import NegotiationActionType
from freelancer_negotiation_env.tasks import TaskDefinition, get_tasks, grade_task

# Required by submission guidance when from_docker_image() mode is used.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
BENCHMARK_NAME = "freelancer_negotiation_env"
MAX_STEPS_PER_TASK = 8
RANDOM_SEED = 7
SUCCESS_SCORE_THRESHOLD = 0.6
LOG_SCORE_EPSILON = 0.01


@dataclass
class TaskRunSummary:
    task_id: str
    total_reward: float
    grader_score: float
    steps: int
    success: bool


def _required_token() -> str:
    token = HF_TOKEN or os.getenv("API_KEY")
    if not token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return token


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _escape_field(value: str) -> str:
    """Keep single-line output for strict parser compatibility."""
    return value.replace("\n", " ").replace("\r", " ").strip()


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = _escape_field(error) if error else "null"
    action_value = _escape_field(action)
    print(
        f"[STEP] step={step} action={action_value} reward={reward:.2f} done={_bool_text(done)} error={error_value}",
        flush=True,
    )


def _clamp_open01(value: float) -> float:
    return max(LOG_SCORE_EPSILON, min(1.0 - LOG_SCORE_EPSILON, value))


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
    def _fallback_action() -> ActionModel:
        state = observation.get("negotiation_state", {}) if isinstance(observation, dict) else {}
        current_price = state.get("current_price", 1200) if isinstance(state, dict) else 1200
        return FreelancerNegotiationAction(
            message=(
                f"I can deliver this professionally for Rs {int(float(current_price))} "
                "with clear scope, timeline, and revision terms."
            ),
            action_type=NegotiationActionType.NEGOTIATE,
        )

    prompt = _build_policy_prompt(task=task, observation=observation, step_index=step_index)

    try:
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
    except Exception:
        return _fallback_action()


def _observation_to_dict(obs: object) -> dict[str, object]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()  # type: ignore[no-any-return]
    return {
        "client_message": getattr(obs, "client_message", ""),
        "negotiation_state": getattr(obs, "negotiation_state", {}),
        "conversation_history": getattr(obs, "conversation_history", []),
        "done": getattr(obs, "done", False),
        "reward": getattr(obs, "reward", 0.0),
    }


def _extract_step_error(obs: object) -> str | None:
    metadata = getattr(obs, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    info = metadata.get("info")
    if not isinstance(info, dict):
        return None
    error = info.get("error")
    if error is None:
        return None
    return str(error)


def run_task(env: Any, llm_client: OpenAI, model_name: str, task: TaskDefinition) -> TaskRunSummary:
    rewards: list[float] = []
    steps_taken = 0

    _log_start(task=task.task_id, env=BENCHMARK_NAME, model=model_name)

    success = False
    total_reward = 0.0
    grader_score = 0.0

    try:
        reset_result = env.reset()
        obs = getattr(reset_result, "observation", None)
        if obs is None:
            raise RuntimeError("reset() did not return an observation")

        final_price: float | None = None
        decision = "negotiate"

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
            obs = getattr(step_result, "observation", None)
            if obs is None:
                raise RuntimeError("step() did not return an observation")
            steps_taken = step_index
            decision = action.action_type.value

            step_reward = float(getattr(step_result, "reward", 0.0) or 0.0)
            total_reward += step_reward
            rewards.append(step_reward)

            state_obj = getattr(obs, "negotiation_state", None)
            current_price = getattr(state_obj, "current_price", None)
            if current_price is not None:
                final_price = float(current_price)

            done = bool(getattr(obs, "done", False) or getattr(step_result, "done", False))
            error = _extract_step_error(obs)
            action_str = f"{decision}:{action.message}"
            _log_step(step=step_index, action=action_str, reward=step_reward, done=done, error=error)

            if done:
                break

        from freelancer_negotiation_env.tasks import EpisodeResult

        episode_result = EpisodeResult(
            final_price=final_price,
            decision=decision,  # type: ignore[arg-type]
            conversation_history=list(getattr(obs, "conversation_history", [])),
            step_count=steps_taken,
            client_type=str(getattr(obs, "metadata", {}).get("client_type", "normal")),
        )
        grader_score = float(grade_task(task.task_id, episode_result))
        success = grader_score >= SUCCESS_SCORE_THRESHOLD

        return TaskRunSummary(
            task_id=task.task_id,
            total_reward=round(total_reward, 3),
            grader_score=round(grader_score, 4),
            steps=steps_taken,
            success=success,
        )
    finally:
        _log_end(success=success, steps=steps_taken, score=_clamp_open01(grader_score), rewards=rewards)


def main() -> None:
    random.seed(RANDOM_SEED)
    _ = perf_counter()  # Keep deterministic seed behavior while avoiding noisy stdout.

    token = _required_token()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=token)

    summaries: list[TaskRunSummary] = []
    client = FreelancerNegotiationEnv(base_url=OPENENV_BASE_URL)
    with client.sync() as env:
        for task in get_tasks():
            summaries.append(run_task(env=env, llm_client=llm_client, model_name=MODEL_NAME, task=task))


if __name__ == "__main__":
    main()
