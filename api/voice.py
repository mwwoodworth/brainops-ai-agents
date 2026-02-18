"""
Voice & Communications API
Enables AUREA to speak and make calls via ElevenLabs and Twilio.
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Security, Depends
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
import httpx

# Configuration
from config import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/voice", tags=["Voice & Comms"])

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    # Check master key from config if set
    master_key = getattr(config.security, 'master_api_key', None) or os.getenv('MASTER_API_KEY')
    if master_key and api_key == master_key:
        return api_key
    if not api_key or api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    optimize_latency: int = 3

class OutboundCallRequest(BaseModel):
    to_number: str
    message: str
    voice_id: Optional[str] = None

# Services
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
DEFAULT_ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

@router.post("/speak")
async def generate_speech(request: TextToSpeechRequest, api_key: str = Depends(verify_api_key)):
    """Generate audio from text using ElevenLabs"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": request.text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    model_candidates: list[str] = []
    for model_id in [
        DEFAULT_ELEVENLABS_MODEL,
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_monolingual_v1",
    ]:
        if model_id and model_id not in model_candidates:
            model_candidates.append(model_id)

    voice_candidates: list[str] = []
    for candidate in [
        request.voice_id,
        DEFAULT_ELEVENLABS_VOICE_ID,
        "21m00Tcm4TlvDq8ikWAM",
    ]:
        if candidate and candidate not in voice_candidates:
            voice_candidates.append(candidate)

    last_status = 500
    last_error = "Unknown ElevenLabs error"
    async with httpx.AsyncClient() as client:
        response = None
        for voice_id in voice_candidates:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            for model_id in model_candidates:
                payload = {**data, "model_id": model_id}
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                if response.status_code == 200:
                    from fastapi.responses import Response

                    return Response(content=response.content, media_type="audio/mpeg")

                last_status = response.status_code
                last_error = response.text
                response_text_l = response.text.lower()
                is_voice_error = (
                    ("invalid_uid" in response_text_l or "invalid id" in response_text_l)
                    and response.status_code in {400, 404, 422}
                )
                is_model_error = "model" in response_text_l and response.status_code in {400, 404, 422}

                if is_voice_error:
                    logger.warning(
                        "ElevenLabs voice '%s' rejected (%s), trying fallback voice",
                        voice_id,
                        response.status_code,
                    )
                    break
                if is_model_error:
                    logger.warning(
                        "ElevenLabs model '%s' rejected (%s), trying fallback model",
                        model_id,
                        response.status_code,
                    )
                    continue
                break

    raise HTTPException(status_code=500, detail=f"ElevenLabs error ({last_status}): {last_error}")

@router.post("/call")
async def make_call(request: OutboundCallRequest, api_key: str = Depends(verify_api_key)):
    """Trigger an outbound call via Twilio (TwiML)"""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        raise HTTPException(status_code=503, detail="Twilio not configured")

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # TwiML to speak the message
        twiml = f'<Response><Say>{request.message}</Say></Response>'
        
        call = client.calls.create(
            to=request.to_number,
            from_=TWILIO_FROM_NUMBER,
            twiml=twiml
        )
        return {"status": "initiated", "call_sid": call.sid}
    except Exception as e:
        logger.error(f"Twilio call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
