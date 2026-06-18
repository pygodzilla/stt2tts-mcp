# STT2TTS MCP server — temporary test image
# Builds a slim container that boots the server and stays alive for stdio probes.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OS deps: ffmpeg is needed for STT audio conversion (server.py imports it).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better layer caching.
COPY pyproject.toml README.md config.yaml ./

# Copy source.
COPY stt2tts_mcp ./stt2tts_mcp
COPY tests ./tests

# Install the package itself (entry point `stt2tts-mcp` from pyproject).
# We deliberately skip heavy engine deps — a smoke test only needs the
# server to boot and respond to MCP protocol calls (health_check,
# list_tts_voices, list_stt_models, reload_config). Those calls work
# without any engine installed.
RUN pip install --no-cache-dir .

# Run regression test as part of image build to catch breakage early.
RUN python tests/test_piper_no_json.py

# Container entrypoint: the MCP server speaks JSON-RPC over stdio.
# We launch it in a loop so probes can be sent, with logs tee'd to a file.
CMD ["sh", "-c", "python -m stt2tts_mcp.server 2>&1 | tee /tmp/stt2tts.log"]
