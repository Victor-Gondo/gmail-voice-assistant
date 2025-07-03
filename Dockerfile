# Voice assistant container
FROM python:3.12-slim

# System deps for audio (Linux example; tweak if Windows)
RUN apt-get update && apt-get install -y \
      libasound2 libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + config
COPY . .

# Ensure dotenv is loaded
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "agent.py"]
