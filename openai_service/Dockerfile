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
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first to ensure a robust installation environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Force reinstall deepgram-sdk to bust cache and ensure latest version/clean install
RUN pip install --no-cache-dir --force-reinstall deepgram-sdk==5.0.0

# Install other needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Diagnostic step to verify PrerecordedOptions import
RUN python -c "import os, sys; \
    try: \
        import deepgram; \
        from deepgram.models import PrerecordedOptions; \
        print('SUCCESS: PrerecordedOptions found at deepgram.models'); \
        sys.exit(0); \
    except ImportError as e: \
        print(f'ERROR: Failed to import PrerecordedOptions: {e}'); \
        sys.exit(1);" || \
    (echo '!!! FATAL ERROR: Docker build diagnostic for Deepgram failed. !!!' && exit 1)

# Copy the rest of your application code
COPY . .

# Expose the fixed port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application using python -m uvicorn
CMD ["python", "-m", "uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]