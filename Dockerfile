# Use a specific Python base image (e.g., 3.10-slim-buster or 3.11-slim-bullseye)
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip uninstall -y numpy torch
RUN pip install --no-cache-dir numpy==1.24.4 torch==2.1.0 openai-whisper python-dotenv fastapi uvicorn
# You might also need to explicitly install other direct dependencies here if they are not picked up by whisper/fastapi

# Copy the rest of your application code
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the application
CMD ["/bin/bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]