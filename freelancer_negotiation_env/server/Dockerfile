# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.11-slim AS builder

WORKDIR /app

# Install only what is needed for building Python wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Use a dedicated virtual environment for clean dependency transfer.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy full project context from freelancer_negotiation_env root.
COPY . /app

# Install project and dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Copy the ready-to-run virtual environment and app sources.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

EXPOSE 8000

# Lightweight health check without installing curl.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Run FastAPI app through uvicorn.
CMD ["python", "-m", "uvicorn", "freelancer_negotiation_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
