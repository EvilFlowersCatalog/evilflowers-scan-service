# syntax=docker/dockerfile:1.7
#
# evilflowers-scan-service: the Celery worker of evilflowers_scan.clean / .erase.
#
#   * scantailor-deviant-cli, built from source (no distro package), as in evilflowers-books-digitalizer's Dockerfile;
#   * Python 3.12 + the locked venv (uv.lock): the worker kit, OpenCV (headless), Pillow;
#   * DocRes is NOT baked in: its venv and weights live on the /models PVC (DOCRES_ENABLED, DOCRES_REPO, DOCRES_PYTHON).
#
# The scantailor stage and the runtime share one Debian release (trixie), so the copied binary finds its libraries.

# The -dev packages double as runtime dependencies: the runtime soname package names churn across Debian releases
# (the t64 transition), the -dev names do not.
ARG SCANTAILOR_DEPS="qtbase5-dev libqt5svg5-dev libtiff-dev libjpeg62-turbo-dev \
    libpng-dev zlib1g-dev libopenjp2-7-dev libexiv2-dev libcanberra-dev \
    libmupdf-dev libharfbuzz-dev libjbig2dec0-dev libmujs-dev libgumbo-dev \
    libfreetype-dev libleptonica-dev"

# -- scantailor-deviant-cli -----------------------------------------------------------------------------------------
FROM debian:trixie-slim AS scantailor
ARG SCANTAILOR_DEPS
# A commit of scantailor-deviant (its default branch head on 2026-11-16: "PDF import added with MuPDF integration").
ARG SCANTAILOR_REF=031878a8ce8fdb49da39ed1ede221790d0d4eb97

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential cmake pkg-config qttools5-dev libboost-dev libeigen3-dev \
        $SCANTAILOR_DEPS \
    && rm -rf /var/lib/apt/lists/*

# CXX 17: the distro's Eigen needs >= C++14.
RUN git clone https://github.com/ImageProcessing-ElectronicPublications/scantailor-deviant.git /src/st \
 && git -C /src/st checkout "${SCANTAILOR_REF}" \
 && sed -i 's/SET(CMAKE_CXX_STANDARD 11)/SET(CMAKE_CXX_STANDARD 17)/' /src/st/CMakeLists.txt \
 && cmake -S /src/st -B /build/st -DCMAKE_BUILD_TYPE=Release \
 && make -C /build/st -j"$(nproc)" scantailor-deviant-cli \
 && install -m 755 /build/st/src/app_cli/scantailor-deviant-cli /usr/local/bin/

# -- build: the virtual environment, from uv.lock -------------------------------------------------------------------
FROM python:3.12-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

# evilflowers-worker is a git dependency (pinned commit in uv.lock); uv clones it with the git CLI.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached while only the code changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --all-extras --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-editable

# -- runtime -----------------------------------------------------------------------------------------------------------
FROM python:3.12-slim-trixie
ARG SCANTAILOR_DEPS

# libgomp1: ScanTailor is built with OpenMP (build-essential brings it into the build stage only).
RUN apt-get update && apt-get install -y --no-install-recommends $SCANTAILOR_DEPS libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=scantailor /usr/local/bin/scantailor-deviant-cli /usr/local/bin/
RUN scantailor-deviant-cli --help >/dev/null

WORKDIR /app
COPY --from=build /app/.venv /app/.venv

ARG IMAGE_TAG=dev
# No REMAP_SIGTERM: SIGTERM (scale-in, rollout) is evilflowers-worker's shutdown. The job runs on for
# WORKER_SOFT_SHUTDOWN_TIMEOUT, then it is cancelled and requeued at once (WORKER_CANCEL_TIMEOUT at most).
# HOME=/tmp: Qt (ScanTailor) writes its settings under $HOME; the pod's root filesystem is read-only, /tmp an emptyDir.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    QT_QPA_PLATFORM=offscreen \
    IMAGE_TAG=${IMAGE_TAG}

USER 65534:65534

# Prefork with one child: Flower can inspect the worker while a batch is being cleaned, and telemetry starts in the
# child (worker_process_init). The bulk Deployment keeps this; the interactive one passes
# --queues=evilflowers_scan.standalone.interactive.
CMD ["celery", "-A", "evilflowers_scan.worker", "worker", "--loglevel=INFO", "-E", "--concurrency=1", \
     "--queues=evilflowers_scan.standalone"]
