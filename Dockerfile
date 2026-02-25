FROM python:3.11-slim

# Install git and venv tooling; clean up apt caches to keep the image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash git python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Create an isolated virtualenv and install Spec Agent with dev extras (includes CLI and lint/test deps).
RUN python3 -m venv /app/.venv \
 && . /app/.venv/bin/activate \
 && pip install --upgrade pip \
 && pip install -e ".[dev]"

# Default state directory inside the container; override with -e SPEC_AGENT_STATE_DIR if desired.
ENV PATH="/app/.venv/bin:${PATH}"
ENV SPEC_AGENT_STATE_DIR=/app/.spec_agent

# Default entrypoint shows CLI help; override in docker run as needed.
CMD ["./spec-agent", "--help"]
