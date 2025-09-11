# Use a specific Python base image (e.g., 3.10-slim-buster or 3.11-slim-bullseye)
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Install ffmpeg for audio/video processing
# This uses apt-get because the base image is Debian-based (buster)
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements.txt (now empty, but good practice to keep)
COPY requirements.txt .

# Install Python dependencies explicitly
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv openai-whisper torch==2.1.0 numpy==1.24.4 python-multipart

# Copy the rest of your application code
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the application, using shell form for environment variable expansion
CMD ["/bin/bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]