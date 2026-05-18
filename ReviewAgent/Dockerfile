# ReviewAgent reproducible environment
# Build:  docker build -t reviewagent .
# Run:    docker run --rm --gpus all -v "$(pwd)/data:/app/data" reviewagent
#
# For CPU-only hosts, omit --gpus all. GPU acceleration requires a CUDA-enabled
# host (nvidia-container-toolkit) and CUDA 12.x-compatible drivers.
#
# The image pins Python 3.13 and installs the project from pyproject.toml so
# every numerical claim in the paper reproduces against the released anchor,
# classification gold, paired ratings, and A1b ablation output.

FROM python:3.13-slim

WORKDIR /app

# System deps for sentence-transformers, scikit-learn, hdbscan, scipy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        gcc \
        g++ \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata first so dependency layer is cached.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source, scripts, configs.
COPY src/      src/
COPY scripts/  scripts/
COPY configs/  configs/

# Default entry: print reproducibility quickstart. Replace with a specific
# pipeline script (e.g. python -m scripts.run_full_pipeline) to actually run.
CMD ["python", "-c", "print('ReviewAgent reproducible env.\\nMount data/ as a volume and invoke pipeline scripts under /app/scripts/.')"]
