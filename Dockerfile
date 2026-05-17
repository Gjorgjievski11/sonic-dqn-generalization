# ── CPU image ─────────────────────────────────────────────────
# docker compose build train-cpu
FROM python:3.11-slim-bookworm AS cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cmake \
    libsdl2-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv sync --extra cpu --no-dev

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY train.py evaluate.py ./

VOLUME ["/app/roms", "/app/runs"]
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "train.py", "--config", "configs/dqn_sonic.yaml"]


# ── GPU image (CUDA 12.1) ─────────────────────────────────────
# docker compose build train-gpu
# Requires NVIDIA Container Toolkit on the host.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    libsdl2-dev \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN uv sync --extra gpu --no-dev

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY train.py evaluate.py ./

VOLUME ["/app/roms", "/app/runs"]
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "train.py", "--config", "configs/dqn_sonic.yaml"]