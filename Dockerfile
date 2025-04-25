# Dockerfile

FROM python:3.10.14-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    python3-dev \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the app
COPY . .

# Expose gRPC port
EXPOSE 50051

# Default command
CMD ["python", "xai_server.py"]
