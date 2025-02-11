# Use Python 3.10.14 slim image
FROM python:3.10.14-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for building Python packages
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
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt requirements.txt

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install aix360

# Copy the source code
COPY . .

# Expose the gRPC port
EXPOSE 50051

# Command to run the gRPC server
CMD ["python", "xai_server.py"]
