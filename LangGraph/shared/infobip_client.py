from __future__ import annotations

import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

INFOBIP_BASE_URL  = (os.getenv("INFOBIP_BASE_URL")  or "l23znj.api-us.infobip.com").strip()
INFOBIP_API_KEY   = (os.getenv("INFOBIP_API_KEY")   or "68414d72f08bdafdeba0f53847c30c95-a7558d3e-dcbd-4eee-b1df-cdcddcdcfae5").strip()
INFOBIP_WA_SENDER = (os.getenv("INFOBIP_WA_SENDER") or "553499042606").strip()


def _base() -> str:
    b = INFOBIP_BASE_URL.rstrip("/")
    return b if b.startswith("http") else f"https://{b}"


def _headers() -> dict:
    return {
        "Authorization": f"App {INFOBIP_API_KEY}",
        "Content-Type": "application/json",
    }


def _fmt_cnpj(cnpj: str) -> str:
    c = re.sub(r"\D", "", cnpj or "")
    if len(c) != 14:
        return cnpj or ""
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:14]}"


async def send_typing_indicator(to: str) -> None:
    """Envia indicador de digitação via WhatsApp (Infobip)."""
    url = f"{_base()}/whatsapp/1/message/typingIndicator"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, headers=_headers(), json={
                "from": INFOBIP_WA_SENDER,
                "to": to,
            })
    except Exception:
        pass


async def send_whatsapp_text(to: str, text: str) -> dict:
    """Envia mensagem de texto via WhatsApp (Infobip)."""
    if not (INFOBIP_BASE_URL and INFOBIP_API_KEY and INFOBIP_WA_SENDER):
        raise RuntimeError("Variáveis INFOBIP não configuradas.")

    url = f"{_base()}/whatsapp/1/message/text"
    payload = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {"text": text},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_whatsapp_document(to: str, media_url: str, filename: str, caption: str = "") -> dict:
    """Envia um arquivo (ex: Excel) como documento via WhatsApp (Infobip)."""
    url = f"{_base()}/whatsapp/1/message/document"
    payload: dict = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {
            "mediaUrl": media_url,
            "filename": filename,
        },
    }
    if caption:
        payload["content"]["caption"] = caption
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_whatsapp_interactive_buttons(to: str, body_text: str, options: list[dict]) -> dict:
    """Envia menu com botões interativos (máx 3 opções)."""
    buttons = []
    for i, opt in enumerate(options, 1):
        name = (opt.get("name") or "").strip()
        title = name[:20] if name else f"Opção {i}"
        buttons.append({"type": "REPLY", "id": str(i), "title": title})

    url = f"{_base()}/whatsapp/1/message/interactive/buttons"
    payload = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {
            "body": {"text": body_text},
            "action": {"buttons": buttons},
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_whatsapp_interactive_list(to: str, body_text: str, options: list[dict]) -> dict:
    """Envia menu com lista interativa (qualquer número de opções)."""
    rows = []
    for i, opt in enumerate(options, 1):
        name = (opt.get("name") or "").strip()
        cnpj = _fmt_cnpj(opt.get("cnpj", ""))
        rows.append({
            "id": str(i),
            "title": name[:24] if name else cnpj,
            "description": cnpj,
        })

    url = f"{_base()}/whatsapp/1/message/interactive/list"
    payload = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {
            "body": {"text": body_text},
            "action": {
                "title": "Selecionar",
                "sections": [{"title": "Estabelecimentos", "rows": rows}],
            },
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_insight_buttons(
    to: str,
    insight_text: str,
    footer_text: str = "Esse insight foi útil para você?",
) -> dict:
    """Envia insight com botões de feedback SIM/NÃO."""
    url = f"{_base()}/whatsapp/1/message/interactive/buttons"
    payload = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {
            "body": {"text": f"💡 *Insight iMAIS*\n\n{insight_text}"},
            "action": {
                "buttons": [
                    {"type": "REPLY", "id": "insight_sim", "title": "Sim ✓"},
                    {"type": "REPLY", "id": "insight_nao", "title": "Não ✗"},
                ],
            },
            "footer": {"text": footer_text},
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_catalog_suggestion_buttons(to: str) -> dict:
    """Pergunta ao usuário se quer adicionar a pergunta ao catálogo (quando out_of_scope)."""
    url = f"{_base()}/whatsapp/1/message/interactive/buttons"
    payload = {
        "from": INFOBIP_WA_SENDER,
        "to": to,
        "content": {
            "body": {
                "text": "Ainda não consigo responder a essa pergunta. Gostaria de adicioná-la ao nosso catálogo para análise?"
            },
            "action": {
                "buttons": [
                    {"type": "REPLY", "id": "catalog_yes", "title": "Sim"},
                    {"type": "REPLY", "id": "catalog_no",  "title": "Não"},
                ],
            },
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_whatsapp_menu(to: str, options: list[dict]) -> None:
    """Envia menu de seleção de CNPJ como texto numerado com CNPJ."""
    lines = ["Identifiquei mais de um estabelecimento no seu número. Qual deles você quer consultar?\n"]
    for i, opt in enumerate(options, 1):
        name = opt.get("name") or _fmt_cnpj(opt.get("cnpj", ""))
        cnpj = _fmt_cnpj(opt.get("cnpj", ""))
        lines.append(f"{i}. {name} ({cnpj})")
    lines += ["\nResponda com o número da opção (ex: 1).",
              "Para trocar depois, envie 'mudar cnpj' ou 'mudar loja'."]
    await send_whatsapp_text(to, "\n".join(lines))
