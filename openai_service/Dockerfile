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

# --- DIAGNOSTIC STEP 1: Verify Deepgram base import (Optional, for debugging) ---
# Uncomment this if you want to explicitly check Deepgram import during build
# RUN echo "--- Verifying Deepgram SDK (base module) after full requirements.txt install ---" && \
#     python -c "import deepgram; print('Deepgram SDK (base module) imported successfully.')" || \
#     (echo "!!! ERROR: Deepgram SDK (base module) failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 2: Verify Uvicorn import (Optional, for debugging) ---
# Uncomment this if you want to explicitly check Uvicorn import during build
# RUN echo "--- Verifying Uvicorn module after full requirements.txt install ---" && \
#     python -c "import uvicorn; print('Uvicorn module imported successfully.')" || \
#     (echo "!!! ERROR: Uvicorn module failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 3: Verify FastAPI import (Optional, for debugging) ---
# Uncomment this if you want to explicitly check FastAPI import during build
# RUN echo "--- Verifying FastAPI module after full requirements.txt install ---" && \
#     python -c "import fastapi; print('FastAPI module imported successfully.')" || \
#     (echo "!!! ERROR: FastAPI module failed to import. Check above logs for details. !!!" && exit 1)

# Copy the rest of your application code
COPY . .

# Expose the fixed port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application using python -m uvicorn
# MODIFIED: Simplified CMD for better compatibility with Railway's Python detection
CMD ["python", "-m", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]
