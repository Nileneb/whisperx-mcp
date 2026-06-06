FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# WHY: CUDA-Torch deterministisch aus dem cu124-Wheel-Index (sonst zieht whisperx CPU-Torch).
RUN pip install --upgrade pip \
    && pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
        "whisperx>=3.3.0" \
        "mcp>=1.2.0" \
        "starlette>=0.37" \
        "uvicorn[standard]>=0.30" \
        "anyio>=4.4"

WORKDIR /srv
COPY app/ /srv/app/
COPY vocab/ /srv/vocab/

EXPOSE 8000
# --proxy-headers: hinter einem TLS-terminierenden Reverse-Proxy bleibt das Schema (https) im
# /mcp -> /mcp/ Redirect erhalten (sonst Downgrade auf http -> MCP-Session bricht).
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
