# ==============================================================================
# frontend/openai_service/Dockerfile (UPDATED for Deepgram, OpenAI, and robust pip)
# Dockerfile for the dedicated OpenAI Whisper, GPT, and Deepgram service
# ==============================================================================

# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for pydub, Deepgram, and general Python compilation
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* # Corrected typo here

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first to ensure a robust installation environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW DIAGNOSTIC STEP: Verify Deepgram imports ---
# This step will fail the build if DeepgramClient or PrerecordedOptions cannot be imported.
# CORRECTED: PrerecordedOptions is imported from deepgram.models
RUN python -c "import deepgram; from deepgram import DeepgramClient; from deepgram.models import PrerecordedOptions; print('Deepgram SDK components imported successfully in Docker build.')" || \
    (echo '!!! ERROR: Deepgram SDK components import FAILED in Docker build. Check deepgram-sdk version and import path. !!!' && exit 1)

# Copy the rest of your application code
COPY . .

# Expose the fixed port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application using python -m uvicorn
# MODIFIED: Simplified CMD for better compatibility with Railway's Python detection
CMD ["python", "-m", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]
