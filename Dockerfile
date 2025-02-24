FROM python:3.10-slim

WORKDIR /app

# Environment variables for CPU-only operation
ENV CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    SDL_AUDIODRIVER=disk \
    VLLM_TARGET_DEVICE=cpu

# Install system dependencies (probably more than needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libasound-dev \
    libsndfile1-dev \
    kmod \
    libomp5 \
    libssl3 \
    ca-certificates \
    openssl \
    libopenblas0-pthread \
    libgl1 \
    libglib2.0-0 \
    libnuma-dev \
    gcc-12 \
    g++-12 \
    espeak-ng \
    libaio-dev \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
sh -s -- -y --default-toolchain stable --profile minimal && \
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /etc/profile.d/rust.sh

ENV PATH="/root/.cargo/bin:${PATH}"

# Verify installation
RUN rustc --version && cargo --version

# Update pip and install base requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy \
    "cmake>=3.26" \
    packaging \
    ninja \
    "setuptools-scm>=8"

# Install Base PyTorch System - For CPU-only
RUN pip install \
  torch \
  torchvision \
  torchaudio \
  torchdatasets \
  torchtext \
  datasets \
  transformers \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Install vLLM CPU version from source
RUN git clone https://github.com/vllm-project/vllm.git \
    && cd vllm \
    && pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python setup.py install \
    && pip install --no-cache-dir -e . \
    && cd .. \
    && rm -rf vllm

# Fix networkx compatibility
RUN pip install --force-reinstall networkx==3.2.1

# Install Python requirements with CPU-only constraints
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps

# Install unidic for processing Japanese texts
RUN python -m unidic download

# Copy application files
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
