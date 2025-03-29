# Single-stage build with CUDA support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Accept build argument for CUDA architectures
ARG CUDA_ARCHITECTURES="89"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libcurl4-openssl-dev \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    python3 \
    python3-pip \
    python3.12-venv \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}
ENV FORCE_CUDA="1"
ENV CUDA_MODULE_LOADING=LAZY
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# Create and activate Python virtual environment
RUN python3 -m venv /root/.venv
ENV PATH="/root/.venv/bin:$PATH"

# Upgrade pip first
RUN /root/.venv/bin/pip install -U pip setuptools wheel

# Upgrade pip
RUN pip3 install -U pip setuptools wheel

# Install PyTorch with CUDA 12.6 support (closest to our CUDA 12.8)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install paddle with GPU support for Python 3.12
RUN pip3 install --no-cache-dir \
    paddlepaddle-gpu==2.6.1.post120 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install just onnx, leave onnxruntime-gpu for entrypoint
RUN pip3 install --no-cache-dir onnx

# Install other dependencies
RUN pip3 install --no-cache-dir \
    openai \
    pillow \
    rapidocr_onnxruntime \
    rapidocr_paddle \
    pypdf2 && \
    pip3 install --no-cache-dir --no-binary :all: tesserocr && \
    pip3 install --no-cache-dir docling

# Clone and build llama.cpp
WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp.git .
# Optionally, you can specify a version/tag/commit with:
# RUN git checkout master # or specific tag/commit hash

RUN mkdir -p build && \
    cd build && \
    cmake .. \
        -DLLAMA_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    cmake --build . --config Release -j $(nproc) && \
    find . -name "*.so*" -exec cp {} /usr/local/lib \; && \
    cp bin/* /usr/local/bin/

# Ensure LD_LIBRARY_PATH includes CUDA libraries
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/root/.venv/lib/python3.12/site-packages/onnxruntime/capi:/usr/local/lib:${LD_LIBRARY_PATH}"

# Add gpu-check.sh file directly to the image to avoid needing to copy it
RUN echo '#!/bin/bash\n\
# gpu-check.sh - Verify GPU setup at container runtime\n\
\n\
echo "====================== NVIDIA DEVICES ======================"\n\
ls -la /dev | grep nvidia\n\
echo ""\n\
\n\
echo "====================== NVIDIA SMI ======================"\n\
if command -v nvidia-smi &> /dev/null; then\n\
    nvidia-smi\n\
else\n\
    echo "nvidia-smi not found - GPU may not be accessible"\n\
fi\n\
echo ""\n\
\n\
echo "====================== PYTORCH CUDA ======================"\n\
python3 -c "\n\
import torch\n\
print(\"PyTorch version:\", torch.__version__)\n\
print(\"CUDA available:\", torch.cuda.is_available())\n\
if torch.cuda.is_available():\n\
    print(\"CUDA version:\", torch.version.cuda)\n\
    print(\"GPU count:\", torch.cuda.device_count())\n\
    print(\"GPU name:\", torch.cuda.get_device_name(0))\n\
"\n\
echo ""\n\
\n\
echo "====================== PADDLE CUDA ======================"\n\
python3 -c "\n\
import paddle\n\
print(\"Paddle version:\", paddle.__version__)\n\
print(\"Paddle compiled with CUDA:\", paddle.device.is_compiled_with_cuda())\n\
if paddle.device.is_compiled_with_cuda():\n\
    print(\"GPU count:\", paddle.device.cuda.device_count())\n\
"\n\
echo ""\n\
\n\
echo "====================== ONNX RUNTIME PROVIDERS ======================"\n\
python3 -c "\n\
import onnxruntime as ort\n\
print(\"ONNX Runtime version:\", ort.__version__)\n\
print(\"Available providers:\", ort.get_available_providers())\n\
"\n\
echo ""\n\
\n\
echo "====================== RAPIDOCR TEST ======================"\n\
python3 -c "\n\
try:\n\
    from rapidocr_paddle import RapidOCR\n\
    print(\"RapidOCR Paddle loaded successfully\")\n\
    # Initialize but do not run (no image provided)\n\
    rapid_ocr = RapidOCR()\n\
    print(\"RapidOCR Paddle initialized successfully\")\n\
except Exception as e:\n\
    print(\"Error loading RapidOCR:\", str(e))\n\
"\n\
' > /usr/local/bin/gpu-check.sh && chmod +x /usr/local/bin/gpu-check.sh

# Create a custom entrypoint script that installs onnxruntime-gpu at startup
RUN echo '#!/bin/bash\n\
\n\
# First time marker\n\
MARKER_FILE="/root/.onnxruntime_installed"\n\
\n\
# Only install onnxruntime-gpu on first container start\n\
if [ ! -f "$MARKER_FILE" ]; then\n\
    echo "First container start - installing onnxruntime-gpu with CUDA support..."\n\
    export CUDA_HOME=/usr/local/cuda\n\
    export CUDA_PATH=/usr/local/cuda\n\
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"\n\
    pip3 uninstall -y onnxruntime onnxruntime-gpu\n\
    pip3 install --no-cache-dir onnxruntime-gpu\n\
    \n\
    # Create marker file to avoid reinstalling on subsequent starts\n\
    touch "$MARKER_FILE"\n\
    \n\
    # Verify installation\n\
    echo "ONNX Runtime providers:"\n\
    python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"\n\
else\n\
    echo "ONNX Runtime with CUDA already installed."\n\
fi\n\
\n\
echo "Run \"gpu-check.sh\" to verify all GPU components"\n\
\n\
# Start interactive shell or execute command if specified\n\
if [ $# -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi\n\
' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]