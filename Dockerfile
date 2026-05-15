FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

COPY pyproject.toml .
COPY src/ src/

RUN uv sync --no-dev

COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

ENV PYTHONPATH=/workspace/src
ENV DESTROR_DATA_DIR=/workspace/data
ENV DESTROR_RESULTS_DIR=/workspace/results

CMD ["uv", "run", "python", "scripts/run_attack.py"]
