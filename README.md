# Freelancer Negotiation Environment

A deterministic, OpenEnv-compatible benchmark for evaluating negotiation agents in realistic freelancer-client conversations.

## Why This Environment Is Competitive

This submission is built around the criteria that matter in both research and production:

- Real-world utility: models margin, scope control, timeline pressure, and toxic client handling.
- Deterministic grading: reproducible scenarios and bounded behavior for fair ranking.
- Explainability: reward components and strategy metadata are surfaced in observations.
- Robustness: supports cooperative, normal, aggressive, and adversarial client profiles.
- Submission readiness: Dockerized, OpenEnv validated, and baseline inference included.

## Problem Framing

Freelancer negotiation is a multi-objective control problem. Strong policies must optimize across competing goals:

- Revenue quality: avoid underpricing and low-signal concessions.
- Conversion quality: close viable deals and reject bad ones.
- Communication quality: stay relevant, concise, and professional.
- Operational quality: constrain revisions, scope creep, and deadline risk.

This environment captures those tradeoffs in a deterministic simulator that is suitable for RL policy development, offline evaluation, and benchmark comparison.

## Environment Overview

### Core State

- current_price
- client_budget
- ideal_price
- minimum_price
- deadline
- revisions
- conversation_history
- client_type (cheap, normal, premium, toxic)
- strategy_type (aggressive, balanced, cooperative)
- memory_summary (recent deal outcomes)

### Episode Flow

1. reset() rotates deterministic scenarios.
2. step(action) validates action, parses offer/intent, updates negotiation state.
3. Client simulator generates counter behavior based on client profile.
4. Dense reward is emitted each turn.
5. Episode ends on accept, reject, or max-step timeout.

## Action and Observation Contracts

### Action

FreelancerNegotiationAction

- message: str
- action_type: one of negotiate | accept | reject

### Observation

FreelancerNegotiationObservation

- client_message: str
- negotiation_state:
  - current_price: float
  - deadline: str
  - revisions: int
- conversation_history: list[str]
- memory_summary: list[dict[str, object]]
- done: bool
- reward: float
- metadata: dict with strategy and reward diagnostics

## Task Suite and Graders

The benchmark includes 3 deterministic tasks in freelancer_negotiation_env/tasks.py:

- Easy: straightforward normal client, fair closure expected.
- Medium: budget-deadline conflict, disciplined reject-or-negotiate behavior expected.
- Hard: toxic client with revision pressure, boundary-setting expected.

Grading characteristics:

- Returns normalized scores in [0.0, 1.0].
- Mixes decision quality, price quality, efficiency, and communication signals.
- Explicitly avoids constant-score behavior.

## Reward Design

Dense reward combines:

- Deal quality at terminal step.
- Negotiation progress toward ideal terms.
- Communication quality bonus.
- Repetition/irrelevance penalties.
- Toxic-client boundary handling bonus.
- Efficiency and decision-quality components.

The design encourages realistic behavior, not exploitative short-horizon tricks.

## Project Structure

```text
.
├── inference.py
├── README.md
└── freelancer_negotiation_env/
    ├── __init__.py
    ├── client.py
    ├── models.py
    ├── tasks.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── tests/
    │   └── test_environment_and_grading.py
    └── server/
        ├── app.py
        ├── freelancer_negotiation_env_environment.py
        └── Dockerfile
```

## Quick Start

### 1. Install Dependencies

```bash
cd freelancer_negotiation_env
python -m pip install -U pip
python -m pip install .[dev]
```

### 2. Run the Environment Server

```bash
python -m uvicorn freelancer_negotiation_env.server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run Unit Tests

```bash
python -m pytest freelancer_negotiation_env/tests -q
```

### 4. Build and Run Docker

```bash
docker build -t freelancer-negotiation-env:latest -f freelancer_negotiation_env/server/Dockerfile freelancer_negotiation_env
docker run --rm -p 8000:8000 freelancer-negotiation-env:latest
```

## Inference Baseline (Submission Mandatory)

The root-level inference.py is compliant with OpenAI-client requirements and emits structured logs:

- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

### Required Environment Variables

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional:

- OPENENV_BASE_URL (default: http://localhost:8000)
- LOCAL_IMAGE_NAME (for docker-image client mode)

### Run Baseline

```bash
python inference.py
```

## Validation Checklist

This project is designed to pass standard submission gates:

- Environment deploys and responds to /reset.
- Dockerfile builds successfully.
- openenv validate passes.
- Baseline inference script exists at repo root.
- At least 3 tasks with deterministic, non-constant graders.

## Engineering Quality Notes

- Deterministic scenario rotation for reproducibility.
- Memory-aware strategy adaptation across episodes.
- Strict action validation and robust price parsing (including INR formats).
- Lightweight Docker context via .dockerignore for faster builds.
- Test coverage for parser behavior, grader bounds, and memory behavior.

## Intended Use

- Benchmarking negotiation policies.
- RL training and regression testing.
- Offline policy comparison with repeatable outcomes.

## License

This project uses the same licensing and attribution structure as the surrounding repository context.
