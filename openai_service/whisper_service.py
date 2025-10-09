import logging
import sys
import subprocess
import os
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from typing import Optional
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING OPENAI WHISPER & GPT SERVICE (ON RENDER) ===")

# Load environment variables (for local testing, Render injects them)
load_dotenv()

# Access OpenAI API Key directly from os.environ
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

logger.info(f"DEBUG: Environment variable 'OPENAI_API_KEY' found: {bool(OPENAI_API_KEY)}")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not configured. OpenAI Whisper and GPT services will not function.")

# Initialize OpenAI clients (one for Whisper, one for GPT - both use the same API key)
openai_whisper_client = None
openai_gpt_client = None

if OPENAI_API_KEY:
    try:
        # Forcefully remove common proxy environment variables before initializing httpx.Client.
        _original_http_proxy = os.environ.pop('HTTP_PROXY', None)
        _original_https_proxy = os.environ.pop('HTTPS_PROXY', None)
        _original_no_proxy = os.environ.pop('NO_PROXY', None)
        
        custom_http_client = httpx.Client(trust_env=False)
        openai_whisper_client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=custom_http_client)
        logger.info("OpenAI Whisper client initialized successfully.")

        openai_gpt_client = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=custom_http_client)
        logger.info("OpenAI GPT client initialized successfully.")

        # Restore original proxy environment variables if they existed
        if _original_http_proxy is not None:
            os.environ['HTTP_PROXY'] = _original_http_proxy
        if _original_https_proxy is not None:
            os.environ['HTTPS_PROXY'] = _original_https_proxy
        if _original_no_proxy is not None:
            os.environ['NO_PROXY'] = _original_no_proxy

    except Exception as e:
        logger.error(f"Error initializing OpenAI clients: {e}")
        openai_whisper_client = None
        openai_gpt_client = None
else:
    logger.warning("OpenAI API key is missing, OpenAI clients will not be initialized.")

app = FastAPI(title="OpenAI Whisper & GPT Transcription/Formatting Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compress_audio_for_transcription(input_path: str, output_path: str = None) -> str:
    """Compress audio file optimally for transcription."""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"
    
    try:
        logger.info(f"Compressing {input_path} for transcription service...")
        audio = AudioSegment.from_file(input_path)
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio for transcription service")
        
        target_sample_rate = 16000
        audio = audio.set_frame_rate(target_sample_rate)
        logger.info(f"Reduced sample rate to {target_sample_rate} Hz for transcription service")
        
        audio.export(
            output_path, 
            format="mp3",
            bitrate="64k",
            parameters=[
                "-q:a", "9",
                "-ac", "1",
                "-ar", str(target_sample_rate)
            ]
        )
        logger.info(f"Audio compression complete for transcription service: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing audio for transcription service: {e}")
        logger.warning(f"Compression failed for {input_path}, returning original path. Service might still handle it.")
        return input_path

@app.post("/transcribe")
async def transcribe_audio_openai(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en")
):
    logger.info(f"OpenAI Whisper transcription endpoint called for file: {file.filename}, language: {language_code}")

    if not openai_whisper_client:
        raise HTTPException(status_code=503, detail="OpenAI Whisper service is not initialized (API key missing).")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    compressed_path = tmp_path

    try:
        compressed_path = compress_audio_for_transcription(tmp_path)

        with open(compressed_path, "rb") as audio_file:
            transcript = openai_whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language_code,
                response_format="json"
            )
        
        transcription_text = transcript.text
        logger.info(f"OpenAI Whisper transcription completed for {file.filename}")
        return {
            "status": "completed",
            "transcription": transcription_text,
            "language": language_code,
            "service_used": "openai_whisper"
        }

    except openai.APIError as e:
        logger.error(f"OpenAI API Error during transcription: {e.response}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e.response}")
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI Whisper transcription: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up original temp file: {tmp_path}")
        if compressed_path != tmp_path and os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed temp file: {compressed_path}")

@app.post("/ai/admin-format-gpt")
async def ai_admin_format_gpt(
    transcript: str = Form(...),
    formatting_instructions: str = Form("Correct all grammar, ensure a formal tone, break into paragraphs with subheadings for each major topic, and highlight action items in bold."),
    model: str = Form("gpt-4-turbo-preview"),
    max_tokens: int = Form(4000)
):
    logger.info(f"OpenAI GPT formatting endpoint called. Model: {model}, Instructions: '{formatting_instructions}'")

    if not openai_gpt_client:
        raise HTTPException(status_code=503, detail="OpenAI GPT service is not initialized (API key missing or invalid).")

    try:
        if len(transcript) > 200000:
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        
        full_prompt = f"Please apply the following formatting and polishing instructions to the provided transcript:\n\nInstructions: {formatting_instructions}\n\nTranscript to format:\n{transcript}"

        completion = openai_gpt_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            timeout=60.0,
        )
        ai_response = completion.choices[0].message.content
        logger.info(f"Successfully processed OpenAI GPT formatting request with model: {model}.")
        return {"formatted_transcript": ai_response}

    except openai.APIError as e:
        logger.error(f"OpenAI API Error during GPT formatting: {e.response}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e.response}")
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI API Timeout during GPT formatting: {e}")
        raise HTTPException(status_code=504, detail="OpenAI API timeout. Please try again with a shorter transcript.")
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI GPT formatting: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OpenAI Whisper & GPT Transcription/Formatting Service is running!"}