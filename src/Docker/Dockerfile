FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /workspace

# Copy the entire repository
COPY . /workspace/

# Mirror requirements file for environments that expect /environment/requirements.txt
RUN mkdir -p /environment && \
        if [ -f /workspace/src/requirements.txt ]; then cp /workspace/src/requirements.txt /environment/requirements.txt; \
        elif [ -f /workspace/requirements.txt ]; then cp /workspace/requirements.txt /environment/requirements.txt; \
        fi

# Install Python dependencies
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
        if [ -f /workspace/src/requirements.txt ]; then \
            python -m pip install --no-cache-dir -r /workspace/src/requirements.txt; \
        elif [ -f /workspace/requirements.txt ]; then \
            python -m pip install --no-cache-dir -r /workspace/requirements.txt; \
        elif [ -f /environment/requirements.txt ]; then \
            python -m pip install --no-cache-dir -r /environment/requirements.txt; \
        else \
            echo "requirements.txt not found in expected locations" && exit 1; \
        fi

# Install additional recommended packages for research
RUN python -m pip install --no-cache-dir \
    huggingface-hub \
    tensorboard \
    matplotlib \
    pandas

# Create directories for data and outputs if they don't exist
RUN mkdir -p src/data/driving/train && \
    mkdir -p src/data/driving/val && \
    mkdir -p src/outputs

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Default command: python with main.py
ENTRYPOINT ["python"]
CMD ["src/main.py", "--help"]
