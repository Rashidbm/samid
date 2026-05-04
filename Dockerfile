# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Saamid Early Warning System — portable backend + dashboard image.
#
# Two layers:
#   builder  → installs uv-managed Python deps + system audio libs and
#              pre-downloads the HuggingFace model so the container boots
#              instantly even without internet
#   runtime  → slim runtime image, copies built venv + repo
#
# Quick run, no hardware (replays a real drone phone recording):
#   docker run --rm -p 8000:8000 ghcr.io/<you>/saamid:latest
#
# Live mode with USB audio interface plugged into the host:
#   docker run --rm -p 8000:8000 \
#       --device /dev/snd \
#       -e SAAMID_SIMULATE= \
#       ghcr.io/<you>/saamid:latest
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.5.11

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# System audio libs (sounddevice → portaudio) + ffmpeg for non-WAV inputs.
# build-essential pulls in gcc for any wheels that lack a manylinux build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        portaudio19-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager — same one used in dev).
ARG UV_VERSION
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Layer the dependency install separately from the source so that source-only
# edits don't bust the heavy torch/transformers cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Bring in the rest of the repo and install it as a package context (no
# project install needed; we run via `python -m`).
COPY . .

# Pre-download the model weights into the image so first boot is offline-safe.
# This adds ~430 MB but means the container starts instantly and works on
# air-gapped networks.
ARG SAAMID_HUB_ID=Rashidbm/samid-drone-detector
ENV HF_HOME=/app/.cache/huggingface
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run python -c "import os; from transformers import AutoFeatureExtractor, AutoModelForAudioClassification; \
        hub=os.environ['SAAMID_HUB_ID']; \
        AutoFeatureExtractor.from_pretrained(hub); \
        AutoModelForAudioClassification.from_pretrained(hub); \
        print('[builder] model cached')" \
    && rm -rf /app/.cache/huggingface/hub/.locks

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Same audio runtime libs (no -dev / -build needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring the prebuilt venv + the application code + the warmed model cache.
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAAMID_HUB_ID=Rashidbm/samid-drone-detector \
    SAAMID_MICS=/app/configs/mics_4_square.json \
    SAAMID_THRESHOLD=0.25 \
    SAAMID_SITE_ID=RUH-14 \
    SAAMID_SIMULATE=/app/data_demo/test_4mic.wav

EXPOSE 7860

# Tini is PID 1 so Ctrl+C / docker stop reach uvicorn cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--log-level", "info", \
     "--ws-ping-interval", "20", "--ws-ping-timeout", "20"]
