# Vehicle Damage Intelligence — Production Container Image
FROM python:3.11-slim

# System dependencies for OpenCV + ONNX
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker cache layer)
COPY requirements.txt .

# Install Python dependencies (no cache to keep image lean)
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Ensure outputs directories exist
RUN mkdir -p outputs/api data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Default mode: Streamlit UI
# Override with: docker run -e MODE=api ...
ENV MODE=ui

EXPOSE 8501 8000

# Entrypoint — routes to correct mode via run.py
CMD ["sh", "-c", "python run.py --mode ${MODE:-ui} --host 0.0.0.0"]
