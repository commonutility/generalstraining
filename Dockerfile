FROM nvidia/cuda:13.0.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "jax[cuda13]" \
    && python -m pip install ./vendor/generals-bots \
    && python -m pip install .

CMD ["python", "jobs/gpu_smoke.py"]
