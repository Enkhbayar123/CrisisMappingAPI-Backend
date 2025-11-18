# Use the official PyTorch image with CUDA support
# Check your server's CUDA version (nvidia-smi), but 12.1 is a safe modern bet.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /code

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for OpenCV and Postgres)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Run the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]