#  CUDA base image with Python 3.11 
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent .pyc files, enable unbuffered stdout/stderr, and hint CUDA
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo

#  Install Python 3.11 + system dependencies 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 python3.11-venv python3.11-dev python3-pip \
       gcc libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

#  Install PyTorch with CUDA 12.4 support 
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

#  Install remaining Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (volume mount overrides this in dev)
COPY . .

# Keep the container alive so the user can exec into it manually.
# Run:  docker exec -it <container_id> bash
# Then: python run_training.py   OR   python run_validation.py
CMD ["sleep", "infinity"]
