from __future__ import annotations

import os
import re
import tempfile
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

INFOBIP_BASE_URL = (os.getenv("INFOBIP_BASE_URL") or "l23znj.api-us.infobip.com").strip()
INFOBIP_API_KEY  = (os.getenv("INFOBIP_API_KEY")  or "").strip()
OPENAI_API_KEY   = (os.getenv("OPENAI_API_KEY")   or "").strip()

_base = INFOBIP_BASE_URL if INFOBIP_BASE_URL.startswith("http") else f"https://{INFOBIP_BASE_URL}"

_openai_client: Optional[AsyncOpenAI] = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def extract_audio_info(evt: dict) -> Optional[dict]:
    """Extrai informações de áudio do evento Infobip (busca em message e na raiz)."""
    message_obj = evt.get("message") or {}
    msg_type = (message_obj.get("type") or evt.get("type") or "").strip().upper()

    if msg_type not in ("AUDIO", "VOICE"):
        return None

    media_url = (
        message_obj.get("url")
        or message_obj.get("mediaUrl")
        or evt.get("url")
        or evt.get("mediaUrl")
    )
    media_id = (
        message_obj.get("id")
        or message_obj.get("mediaId")
        or evt.get("mediaId")
    )
    mime_type = (
        message_obj.get("mimeType")
        or message_obj.get("contentType")
        or evt.get("mimeType")
        or "audio/ogg"
    )

    return {
        "media_url": media_url,
        "media_id":  media_id,
        "mime_type": mime_type,
    }


async def download_audio(media_url: Optional[str] = None, media_id: Optional[str] = None) -> bytes:
    """Baixa o arquivo de áudio do Infobip."""
    headers = {
        "Authorization": f"App {INFOBIP_API_KEY}",
        "Accept": "*/*",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        if media_url:
            print(f"[AUDIO] baixando por media_url")
            r = await client.get(media_url, headers=headers)
            r.raise_for_status()
            return r.content

        if media_id:
            print(f"[AUDIO] baixando por media_id={media_id}")
            url = f"{_base}/whatsapp/1/inbound/media/{media_id}"
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.content

    raise RuntimeError("Nenhuma media_url ou media_id disponível para download.")


def _mime_to_suffix(mime_type: str) -> str:
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    if "wav" in mime_type:
        return ".wav"
    if "mp4" in mime_type or "m4a" in mime_type:
        return ".m4a"
    return ".ogg"


async def transcribe_bytes(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """Transcreve áudio usando OpenAI Whisper API (sem dependência de CPU local)."""
    suffix = _mime_to_suffix(mime_type)
    filename = f"audio{suffix}"

    client = _get_openai()
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes, mime_type),
        language="pt",
    )
    text = (response.text or "").strip()
    text = re.sub(r"[.,!?;:]+$", "", text).strip()
    print(f"[WHISPER API] text={text!r}")
    return text


async def transcribe_message(evt: dict) -> Optional[str]:
    """Entry point: dado o evento completo Infobip, retorna a transcrição ou None."""
    info = extract_audio_info(evt)
    if not info:
        return None

    try:
        audio_bytes = await download_audio(
            media_url=info["media_url"],
            media_id=info["media_id"],
        )
        print(f"[AUDIO] {len(audio_bytes)} bytes baixados")

        text = await transcribe_bytes(audio_bytes, info["mime_type"])
        return text or None

    except Exception as e:
        print(f"[AUDIO] ERRO na transcrição: {e}")
        return None
