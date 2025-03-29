#!/bin/bash
# gpu-check.sh - Verify GPU setup at container runtime

echo "====================== NVIDIA DEVICES ======================"
ls -la /dev | grep nvidia
echo ""

echo "====================== NVIDIA SMI ======================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found - GPU may not be accessible"
fi
echo ""

echo "====================== PYTORCH CUDA ======================"
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
"
echo ""

echo "====================== PADDLE CUDA ======================"
python3 -c "
import paddle
print('Paddle version:', paddle.__version__)
print('Paddle compiled with CUDA:', paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print('GPU count:', paddle.device.cuda.device_count())
"
echo ""

echo "====================== ONNX RUNTIME PROVIDERS ======================"
python3 -c "
import onnxruntime as ort
print('ONNX Runtime version:', ort.__version__)
print('Available providers:', ort.get_available_providers())
"
echo ""

echo "====================== RAPIDOCR TEST ======================"
python3 -c "
try:
    from rapidocr_paddle import RapidOCR
    print('RapidOCR Paddle loaded successfully')
    # Initialize but don't run (no image provided)
    rapid_ocr = RapidOCR()
    print('RapidOCR Paddle initialized successfully')
except Exception as e:
    print('Error loading RapidOCR:', str(e))
"