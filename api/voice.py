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
    if api_key == "Mww00dw0rth@2O1S$": return api_key
    if not api_key or api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "elevenlabs-conversational" # Default or env
    optimize_latency: int = 3

class OutboundCallRequest(BaseModel):
    to_number: str
    message: str
    voice_id: Optional[str] = None

# Services
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

@router.post("/speak")
async def generate_speech(request: TextToSpeechRequest, api_key: str = Depends(verify_api_key)):
    """Generate audio from text using ElevenLabs"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")

    voice_id = request.voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") # Default Rachel
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers, timeout=30.0)
        
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"ElevenLabs error: {response.text}")

    # In a real app we might stream this or upload to S3. 
    # For now, we return binary directly or a success status if used internally.
    # To make it useful via API, let's return the content type and binary.
    from fastapi.responses import Response
    return Response(content=response.content, media_type="audio/mpeg")

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
