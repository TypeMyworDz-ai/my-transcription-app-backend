# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# NEW: Install PyTorch for CPU first, explicitly from the CPU index
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install any other needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Set environment variables for CPU optimization
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV WHISPER_MODEL_SIZE=base

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the whisper_service.py script when the container launches
CMD ["python", "whisper_service.py"]