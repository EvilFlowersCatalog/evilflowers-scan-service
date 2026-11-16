.PHONY: install install-all lint format typecheck test check worker worker-interactive image diagrams

# Core + dev tools: enough for the tests.
install:
	uv sync

# Everything the image has (adds the OpenTelemetry and Pyroscope SDKs).
install-all:
	uv sync --all-extras

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

# A local worker (needs a reachable REDIS_URL, STORAGE_PATH and scantailor-deviant-cli on PATH).
# macOS: --pool=solo (evilflowers-worker README).
worker:
	uv run celery -A evilflowers_scan.worker worker --loglevel=INFO -E --concurrency=1 \
		--queues=evilflowers_scan.standalone

worker-interactive:
	uv run celery -A evilflowers_scan.worker worker --loglevel=INFO -E --concurrency=1 \
		--queues=evilflowers_scan.standalone.interactive

image:
	docker build --build-arg IMAGE_TAG=local@$$(git rev-parse --short HEAD) -t evilflowers-scan-service:local .

diagrams:
	PLANTUML_LIMIT_SIZE=8192 plantuml -tsvg -charset UTF-8 docs/diagrams/*.puml
	PLANTUML_LIMIT_SIZE=8192 plantuml -tpng -charset UTF-8 docs/diagrams/*.puml
