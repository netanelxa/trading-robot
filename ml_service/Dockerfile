FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies and clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ml_service.py .

EXPOSE 5002

CMD ["python", "ml_service.py"]