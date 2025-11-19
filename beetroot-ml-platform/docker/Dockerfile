# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system deps (optional, but useful for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model, preserving folder structure
COPY scripts ./scripts
COPY models ./models

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI app
# scripts.api -> folder "scripts", file "api.py", variable "app"
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
