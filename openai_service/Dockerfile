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

# Force reinstall deepgram-sdk to bust cache and ensure latest version/clean install
# This is a crucial step to eliminate any caching issues with deepgram-sdk itself.
RUN pip install --no-cache-dir --force-reinstall deepgram-sdk

# Install any other needed Python packages specified in requirements.txt (excluding deepgram-sdk as it's forced above)
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW COMPREHENSIVE DIAGNOSTIC STEP: Find PrerecordedOptions ---
# This step will try to list the contents of the 'deepgram' package
# and explicitly attempt to import PrerecordedOptions from various common locations.
# Dockerfile snippet (starting at line 37)

RUN python -c "import os, sys;\
try: import deepgram;\
print(f'--- Contents of deepgram package ({deepgram.__file__}): ---');\
for root, dirs, files in os.walk(os.path.dirname(deepgram.__file__)): \
for f in files + dirs: \
print(os.path.join(root, f)); \
print('--- End Deepgram package contents ---');\
\
try: from deepgram.models import PrerecordedOptions; print('SUCCESS: PrerecordedOptions found at deepgram.models'); sys.exit(0)\
except ImportError: pass;\
try: from deepgram import PrerecordedOptions; print('SUCCESS: PrerecordedOptions found at deepgram (top level)'); sys.exit(0)\
except ImportError: pass;\
\
print('!!! ERROR: PrerecordedOptions NOT FOUND after inspecting deepgram package. !!!'); sys.exit(1);\
except ImportError as e:\
print(f'!!! ERROR: Deepgram SDK (base module) failed to import during diagnostic: {e}. !!!'); sys.exit(1);\
" || (echo '!!! FATAL ERROR: Docker build diagnostic for Deepgram failed. !!!' && exit 1)

# Copy the rest of your application code
COPY . .

# Expose the fixed port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application using python -m uvicorn
# MODIFIED: Simplified CMD for better compatibility with Railway's Python detection
CMD ["python", "-m", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]
