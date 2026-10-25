# ML training image. Default = rocm/pytorch. CUDA = override base.
# Build with:   docker build -t bff-ml:rocm .

ARG BASE=rocm/pytorch:latest
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# bff-core is private: override with a token URL when building locally.
#   docker build --build-arg \
#     BFF_CORE_REF=git+https://x-access-token:<TOKEN>@github.com/mahardhika/bff-core.git@v0.1.0 .
ARG BFF_CORE_REF=git+https://github.com/mahardhika/bff-core.git@v0.1.0

WORKDIR /app

# bff-core first: the `ml` package depends on it and it is not on PyPI.
RUN pip install "$BFF_CORE_REF"

COPY pyproject.toml /app/ml/pyproject.toml
COPY ml/__init__.py /app/ml/ml/__init__.py
RUN pip install -e /app/ml[cuda,notebook,dev]

COPY ml/ /app/ml/

WORKDIR /app/ml

CMD ["python", "-m", "ml", "train", "--help"]
