import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
from typing import Optional
import torch

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

# Global model variable
model = None

def load_whisper_model():
    """Load Whisper model with CPU optimizations"""
    global model
    if model is None:
        logger.info("Loading Whisper model with CPU optimizations...")
        
        # Use 'base' model for optimal speed/accuracy balance on CPU
        MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
        model = whisper.load_model(MODEL_SIZE)
        
        # CPU optimizations
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(4)
        
        logger.info(f"Whisper {MODEL_SIZE} model loaded successfully")
    
    return model

# Load model on startup
load_whisper_model()

@app.get("/")
async def root():
    return {
        "message": "Whisper Transcription Service is running", 
        "model": os.getenv("WHISPER_MODEL_SIZE", "base"),
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model": os.getenv("WHISPER_MODEL_SIZE", "base")
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en")
):
    """
    Transcribe audio file using Whisper with CPU optimizations
    """
    try:
        logger.info(f"Starting transcription for: {file.filename}, language: {language_code}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        # File size limit for performance (100MB max)
        if file_size_mb > 100:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 100MB."
            )
        
        logger.info(f"Processing file: {file_size_mb:.2f} MB")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Get model
            whisper_model = load_whisper_model()
            
            # Prepare language parameter
            language_param = None if language_code in ["auto", ""] else language_code
            
            logger.info("Starting Whisper transcription with CPU optimizations...")
            
            # OPTIMIZED TRANSCRIPTION - KEY PERFORMANCE IMPROVEMENTS
            result = whisper_model.transcribe(
                temp_path,
                language=language_param,
                task="transcribe",
                fp16=False,  # CRITICAL: Disable FP16 for CPU (removes warnings)
                verbose=False,  # Reduce logging overhead
                condition_on_previous_text=False,  # Faster processing
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                initial_prompt=None,  # Remove for speed
                word_timestamps=False,  # MAJOR SPEED BOOST: Disable word timestamps
            )
            
            # Extract results
            transcript_text = result['text'].strip()
            detected_language = result.get('language', language_code)
            
            # Log completion
            char_count = len(transcript_text)
            logger.info(f"Transcription completed! {char_count} characters generated")
            
            return {
                "status": "completed",
                "transcript": transcript_text,
                "language": detected_language,
                "model_used": os.getenv("WHISPER_MODEL_SIZE", "base"),
                "file_size_mb": round(file_size_mb, 2),
                "character_count": char_count
            }
            
        except Exception as transcription_error:
            logger.error(f"Transcription processing error: {str(transcription_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Transcription processing failed: {str(transcription_error)}"
            )
            
        finally:
            # Always clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info("Temporary file cleaned up successfully")
            except Exception as cleanup_error:
                logger.warning(f"Temp file cleanup warning: {cleanup_error}")
                
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription service error: {str(e)}")

# Health check endpoint for monitoring
@app.get("/status")
async def status():
    return {
        "service": "whisper-transcription",
        "status": "running",
        "model": os.getenv("WHISPER_MODEL_SIZE", "base"),
        "cpu_optimized": True,
        "max_file_size_mb": 100
    }

# For deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Whisper service on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        access_log=True
    )
