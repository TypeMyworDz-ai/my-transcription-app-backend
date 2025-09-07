from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import uuid
from datetime import datetime

# Create the FastAPI app
app = FastAPI(title="Transcription Service")

# Allow frontend to connect to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI model for transcription
print("Loading Whisper model... This might take a moment.")
model = whisper.load_model("base")
print("Model loaded successfully!")

# Store transcription jobs in memory
jobs = {}

@app.get("/")
async def root():
    return {"message": "Transcription Service is running!"}

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    # Check if file is audio or video
    if not file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe the audio/video
        print(f"Starting transcription for {file.filename}")
        result = model.transcribe(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Update job with results
        jobs[job_id].update({
            "status": "completed",
            "transcription": result["text"],
            "language": result["language"],
            "completed_at": datetime.now().isoformat()
        })
        
        print(f"Transcription completed for {file.filename}")
        
    except Exception as e:
        # If something goes wrong
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        print(f"Transcription failed: {str(e)}")
    
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

# Run this if the file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
