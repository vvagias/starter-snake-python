# Start with the official Python 3.8 image
FROM python:3.7-slim


# Install the necessary system packages for CUDA
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA’s package repositories for CUDA
RUN curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu20.04/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update && apt-get install -y nvidia-container-toolkit

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["python", "inference.py"]