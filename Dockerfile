# Use the official fastapi uvicorn image
FROM python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application
COPY . .

# If your main file is named app.py, use the following line to start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]