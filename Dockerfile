FROM python:3.11-slim

# System deps for opencv & mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole package
COPY . /app/Workout-tracking

WORKDIR /app

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "Workout-tracking.api:app", "--host", "0.0.0.0", "--port", "8001"]
