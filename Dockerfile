FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

# Environment variables for CPU-only operation
# ENV CUDA_VISIBLE_DEVICES=-1 \
#     TF_CPP_MIN_LOG_LEVEL=3 \
#     SDL_AUDIODRIVER=disk \
#     VLLM_TARGET_DEVICE=cpu

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

ADD https://astral.sh/uv/install.sh /uv-installer.sh  
RUN sh /uv-installer.sh && rm /uv-installer.sh  
ENV PATH="/root/.local/bin/:$PATH"  

COPY pyproject.toml .
COPY uv.lock .
RUN uv venv

# Set the virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Update pip and install base requirements (CPU-only)
# RUN uv pip install --upgrade pip setuptools wheel && \
#     uv pip install numpy \
#     "cmake>=3.26" \
#     packaging \
#     ninja \
#     "setuptools-scm>=8" --no-cache-dir

# Update pip and install base requirements
RUN uv pip install --upgrade pip setuptools wheel && \
    uv pip install numpy \
    "cmake>=3.26" \
    packaging \
    ninja \
    vllm \
    "setuptools-scm>=8" --no-cache-dir

# Install Base PyTorch System - For CPU-only
# RUN pip install \
#   torch \
#   torchvision \
#   torchaudio \
#   torchdatasets \
#   torchtext \
#   datasets \
#   transformers \
#   --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

  RUN uv pip install \
  torch \
  torchaudio \
  torchdatasets \
  torchtext \
  datasets \
  transformers \
  --extra-index-url https://download.pytorch.org/whl --no-cache-dir

# Install vLLM CPU version from source
# RUN git clone https://github.com/vllm-project/vllm.git \
#     && cd vllm \
#     && pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu \
#     && python setup.py install \
#     && pip install --no-cache-dir -e . \
#     && cd .. \
#     && rm -rf vllm

# Fix networkx compatibility
RUN uv pip install --force-reinstall --no-cache-dir networkx==3.2.1

# Install Python requirements with CPU-only constraints
# COPY requirements.txt .

RUN uv sync --frozen

# Install unidic for processing Japanese texts
RUN python -m unidic download

# Copy application files
COPY . .

EXPOSE 7860
CMD ["uv", "run", "app.py"]
