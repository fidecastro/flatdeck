#!/bin/bash

# Set default image name and tag
IMAGE_NAME="flatdeck"
IMAGE_TAG="v1.0"

# Process any command line parameters
while [[ $# -gt 0 ]]; do
  case $1 in
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --image-name NAME   Set the Docker image name (default: llama-cpp-connector)"
      echo "  --tag TAG           Set the Docker image tag (default: latest)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Detect CUDA compute capability using nvidia-smi
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)

# Check if detection was successful
if [ -z "$COMPUTE_CAP" ]; then
    echo "Error: Could not detect CUDA compute capability."
    echo "Using default value of 89 (RTX 4090)."
    CUDA_ARCH="89"
else
    # Remove the dot from the compute capability (e.g., 8.9 -> 89)
    CUDA_ARCH=${COMPUTE_CAP//./}
    echo "Detected CUDA architecture: $COMPUTE_CAP (using $CUDA_ARCH for build)"
fi

# Display build information
echo "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
echo "CUDA Architecture: $CUDA_ARCH"
echo "Starting build process..."

# Build the Docker image with the detected architecture
docker build \
    --build-arg CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -f Dockerfile .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build complete!"
    echo "You can run the container with: docker run --gpus all -it $IMAGE_NAME:$IMAGE_TAG"
else
    echo "Build failed. Please check the error messages above."
    exit 1
fi