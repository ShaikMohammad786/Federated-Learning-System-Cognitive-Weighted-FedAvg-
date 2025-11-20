FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to allow better layer caching
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt && pip cache purge

# Copy only necessary sources to keep image small (data and caches are volume-mounted)
COPY config.py /workspace/config.py
COPY utils.py /workspace/utils.py
COPY main.py /workspace/main.py
COPY orchestrator.py /workspace/orchestrator.py
COPY client.py /workspace/client.py
COPY evaluate.py /workspace/evaluate.py
COPY data_prep.py /workspace/data_prep.py

# Default command can be overridden in docker-compose
CMD ["python", "orchestrator.py"]


