# Reproducible CPU/Linux image for training + batch inference.
#
# Apple-Silicon MPS is a *development* accelerator only, torch MPS wheels are
# macOS-only, so the portable, cloud-deployable path is this CPU image. On GCP
# swap the base for an NVIDIA CUDA image and the cuda torch wheel (see
# docs/cloud_gcp.md); nothing else changes.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

WORKDIR /app

# rasterio / geopandas / shapely ship self-contained manylinux wheels (bundled
# GDAL/GEOS/PROJ), so only a couple of runtime libs are needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU torch first so the project install resolves against it.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY sql ./sql
COPY tests ./tests
RUN pip install -e ".[dl,stats,geo,mlops]"

# Default: run the offline test suite (minus the heavy deep forward-pass tests)
# to prove the image is wired correctly.
CMD ["pytest", "-q", "--ignore=tests/test_deep_models.py"]
