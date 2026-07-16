from __future__ import annotations

import os
from datetime import datetime
import httpx
from dotenv import load_dotenv

load_dotenv()

LOGICAPP_EMAIL_URL  = os.getenv("LOGICAPP_EMAIL_URL", "").strip()
INSIGHT_EMAIL_TO    = os.getenv("INSIGHT_EMAIL_TO",  "pedro.teixeira@martins.com.br").strip()
INSIGHT_EMAIL_FROM  = os.getenv("INSIGHT_EMAIL_FROM", "imais@martins.com.br").strip()
REPORT_EMAIL_CC     = os.getenv("REPORT_EMAIL_CC", "").strip()  # emails extras separados por vírgula

_HEADERS = {
    "Content-Type": "application/json",
    "From": INSIGHT_EMAIL_FROM,
}


async def _post_email(title: str, html_message: str, to: str | None = None) -> None:
    if not LOGICAPP_EMAIL_URL:
        print("[EMAIL] LOGICAPP_EMAIL_URL não configurada — ignorado")
        return
    recipient = to or INSIGHT_EMAIL_TO
    payload = {
        "title":   title,
        "message": html_message,
        "color":   "Blue",
        "to":      recipient,
        "from":    INSIGHT_EMAIL_FROM,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(LOGICAPP_EMAIL_URL, headers=_HEADERS, json=payload)
            print(f"[EMAIL] enviado → {recipient} status={r.status_code}")
    except Exception as e:
        print(f"[EMAIL] ERRO: {e}")


async def _post_email_all(title: str, html_message: str) -> None:
    """Envia para INSIGHT_EMAIL_TO e para todos os emails em REPORT_EMAIL_CC."""
    await _post_email(title, html_message)
    if REPORT_EMAIL_CC:
        for extra in REPORT_EMAIL_CC.split(","):
            extra = extra.strip()
            if extra:
                await _post_email(title, html_message, to=extra)


async def send_insight_feedback_email(
    phone: str,
    question: str,
    answer: str,
    insight: str,
    feedback: str,
) -> None:
    """Envia feedback do insight via Azure Logic App."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    title = f"[iMAIS Insight] Feedback: {feedback} — {phone}"
    html = f"""
    <html><body>
      <h3>Feedback de Insight — iMAIS Analytics</h3>
      <p><b>Usuário (WhatsApp):</b> {phone}</p>
      <p><b>Data/Hora:</b> {now}</p>
      <hr>
      <p><b>Pergunta:</b> {question}</p>
      <p><b>Resposta enviada:</b><br>{answer.replace(chr(10), '<br>')}</p>
      <p><b>Insight gerado:</b><br>{insight.replace(chr(10), '<br>')}</p>
      <hr>
      <p><b>Feedback do cliente:</b> <strong>{feedback}</strong></p>
      <p style="color:gray;font-size:11px;">Email gerado automaticamente. Por favor, não responda.</p>
    </body></html>
    """
    await _post_email(title, html)


async def send_image_received_email(phone: str, cnpj: str = "") -> None:
    """Avisa por email que um cliente enviou uma FOTO (reconhecimento de imagem
    está temporariamente desativado — não anexa nem baixa a imagem em si).
    """
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    title = f"[iMAIS Estoque] Cliente enviou uma foto — {phone}"
    html = f"""
    <html><body>
      <h3>Cliente enviou uma foto pelo WhatsApp</h3>
      <p><b>Telefone:</b> {phone}</p>
      <p><b>CNPJ:</b> {cnpj or "-"}</p>
      <p><b>Data/Hora:</b> {now}</p>
      <p>O reconhecimento de imagem está temporariamente desativado (priorizando leitura de PDF).
      O cliente já recebeu uma mensagem avisando que essa funcionalidade ainda está em desenvolvimento.</p>
      <p style="color:gray;font-size:11px;">Email gerado automaticamente. Por favor, não responda.</p>
    </body></html>
    """
    await _post_email(title, html)


async def send_catalog_suggestion_email(
    phone: str,
    question: str,
    cnpj: str = "",
) -> None:
    """Envia sugestão de pergunta para o catálogo via Azure Logic App."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    title = "Nova pergunta para adicionar ao catálogo — iMAIS"
    html = f"""
    <html><body>
      <h3>Nova pergunta para adicionar ao catálogo do iMAIS</h3>
      <p><b>Telefone:</b> {phone}</p>
      <p><b>CNPJ:</b> {cnpj or "-"}</p>
      <p><b>Pergunta:</b> {question}</p>
      <p><b>Data/Hora:</b> {now}</p>
      <p style="color:gray;font-size:11px;">Email gerado automaticamente. Por favor, não responda.</p>
    </body></html>
    """
    await _post_email(title, html)
