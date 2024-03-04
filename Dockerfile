# Use the official fastapi uvicorn image
FROM python:3.11-slim

# Install system dependencies including Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application
COPY . .

# If your main file is named app.py, use the following line to start the FastAPI application
CMD ["uvicorn", "scripts.get_chunks_api:app", "--host", "0.0.0.0", "--port", "80"]