import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form # NEW: Import Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
from typing import Optional # NEW: Import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Whisper Transcription Service")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model on startup
logger.info("Loading Whisper model...")
# Consider loading a larger model if you have enough RAM (e.g., "medium", "large")
# model = whisper.load_model("medium") 
model = whisper.load_model("base")  
logger.info("Whisper model loaded successfully")

@app.get("/")
async def root():
    return {
        "message": "Whisper Transcription Service is running!",
        "model": "base",
        "status": "ready"
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None) # NEW: Optional language parameter
):
    logger.info(f"Received file: {file.filename}, requested language: {language}")
    
    # Check if file is audio/video
    if not file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Transcribe with Whisper, passing language if provided
            logger.info("Starting transcription...")
            if language:
                result = model.transcribe(tmp_file.name, language=language)
                logger.info(f"Transcribed with language: {language}")
            else:
                result = model.transcribe(tmp_file.name)
                logger.info("Transcribed with auto-detected language")
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            logger.info("Transcription completed successfully")
            return {
                "transcript": result["text"],
                "language": result.get("language", "unknown"),
                "status": "completed"
            }
            
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)