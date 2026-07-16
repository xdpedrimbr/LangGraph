"""Leitura de relatórios de movimentação de estoque em PDF (ex: "Movimentação Interna
de Produtos" exportado de sistemas como Sambanet).

Diferente da foto manuscrita, este PDF é texto digital — sem ambiguidade de OCR — e já
traz o código interno/barra do produto por linha, permitindo casar direto no estoque
real sem busca fuzzy por nome.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
from typing import Optional

import httpx
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

INFOBIP_BASE_URL = (os.getenv("INFOBIP_BASE_URL") or "l23znj.api-us.infobip.com").strip()
INFOBIP_API_KEY  = (os.getenv("INFOBIP_API_KEY")  or "").strip()

_base = INFOBIP_BASE_URL if INFOBIP_BASE_URL.startswith("http") else f"https://{INFOBIP_BASE_URL}"


def extract_document_info(evt: dict) -> Optional[dict]:
    """Extrai url/id/mime de um documento PDF do evento Infobip (busca em message e raiz)."""
    message_obj = evt.get("message") or {}
    msg_type = (message_obj.get("type") or evt.get("type") or "").strip().upper()
    mime_type = (
        message_obj.get("mimeType")
        or message_obj.get("contentType")
        or evt.get("mimeType")
        or ""
    ).lower()

    if msg_type not in ("DOCUMENT", "FILE") and "pdf" not in mime_type:
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
    return {"media_url": media_url, "media_id": media_id, "mime_type": mime_type or "application/pdf"}


async def download_document(media_url: Optional[str] = None, media_id: Optional[str] = None) -> bytes:
    """Baixa o documento do Infobip por url ou media_id (mesmo endpoint usado para imagens)."""
    headers = {"Authorization": f"App {INFOBIP_API_KEY}", "Accept": "*/*"}
    async with httpx.AsyncClient(timeout=60) as client:
        if media_url:
            print("[PDF] baixando por media_url")
            r = await client.get(media_url, headers=headers)
            r.raise_for_status()
            return r.content
        if media_id:
            print(f"[PDF] baixando por media_id={media_id}")
            url = f"{_base}/whatsapp/1/inbound/media/{media_id}"
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.content
    raise RuntimeError("Nenhuma media_url ou media_id disponível para download do PDF.")


# ── Parsing do relatório "Movimentação Interna de Produtos" ──────────────────

# Linhas de dado começam com a legenda (Aberto/Faturado) + data dd/mm/aaaa.
_ROW_START_RE = re.compile(r'^[AF]\s+(\d{2}/\d{2}/\d{4})\s+(\S+)\s+(.*)$')
# As 5 últimas colunas numéricas (Quantidade, Preço C.M.V, Total C.M.V, Preço Venda, Total Venda).
_NUM_RE = re.compile(r'^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$')
_UNIT_TOKENS = {"kg", "un.", "un", "und", "g", "cx", "bd", "pct"}


def _is_fornecedor_start(tokens: list[str], i: int) -> bool:
    """Detecta o início do campo Fornecedor: token numérico (código) seguido de '-' e
    de uma palavra iniciada por maiúscula (nome do fornecedor). Ex: '393 - CITROBELL'.
    """
    if i + 2 >= len(tokens):
        return False
    return (
        tokens[i].isdigit() and len(tokens[i]) >= 2
        and tokens[i + 1] == "-"
        and tokens[i + 2][:1].isupper()
    )

_SKIP_PREFIXES = ("Total ", "Grupo:", "Centro de Receita", "Data ", "Movimentação", "Loja:",
                  "Período:", "Motivo:", "Quebra por:", "Legenda:")


def _to_float_br(s: str) -> Optional[float]:
    """Converte número no padrão brasileiro ('1.234,50' ou '2,500') para float."""
    s = (s or "").strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_row(line: str) -> Optional[dict]:
    """Tenta extrair {codigo, descricao, quantidade} de uma linha de dado da tabela."""
    m = _ROW_START_RE.match(line.strip())
    if not m:
        return None
    _data, barra, rest = m.groups()

    tokens = rest.split()
    # As últimas 5 colunas numéricas: Quantidade, Preço C.M.V, Total C.M.V, Preço Venda, Total Venda.
    num_idxs = [i for i, t in enumerate(tokens) if _NUM_RE.match(t)]
    if len(num_idxs) < 5:
        return None
    qty_idx = num_idxs[-5]
    quantidade = _to_float_br(tokens[qty_idx])
    if quantidade is None or quantidade <= 0:
        return None

    # Descrição: tudo antes do fornecedor ("<código> - <NOME>") e antes da coluna de quantidade.
    head = tokens[:qty_idx]
    desc_tokens: list[str] = []
    for i, t in enumerate(head):
        if _is_fornecedor_start(head, i):
            break
        desc_tokens.append(t)
    # Remove unidade solta no final da descrição (ex: "... kg", "... UND").
    while desc_tokens and desc_tokens[-1].lower().rstrip(".") in _UNIT_TOKENS:
        desc_tokens.pop()

    descricao = " ".join(desc_tokens).strip()
    if len(descricao) < 2:
        return None

    return {"codigo": barra.strip(), "descricao": descricao, "quantidade": quantidade}


def parse_movimentacao_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extrai as linhas de produto de um PDF de 'Movimentação Interna de Produtos'.

    Retorna lista de {codigo, descricao, quantidade}. Linhas de total/cabeçalho/grupo
    são ignoradas. Action é sempre 'saída' para este tipo de relatório (quebra/lixo).

    Nomes de fornecedor longos (ex: 'VIEIRA TANNUS& CIA LTDA') quebram em 2 linhas na
    tabela renderizada — acumula linhas de continuação até fechar uma linha de dado
    completa (ou até a próxima linha de dado / linha de total começar).
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_lines: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                all_lines.extend(text.split("\n"))
    except Exception as e:
        print(f"[PDF] ERRO ao processar: {e}")
        return []

    rows: list[dict] = []
    buffer: list[str] = []

    def _flush() -> None:
        if not buffer:
            return
        parsed = _parse_row(" ".join(buffer))
        if parsed:
            rows.append(parsed)
        buffer.clear()

    for raw_line in all_lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_SKIP_PREFIXES):
            _flush()
            continue
        if _ROW_START_RE.match(line):
            _flush()
            buffer.append(line)
        elif buffer:
            # Continuação do fornecedor quebrado em linha (não tem início de linha de dado).
            buffer.append(line)
    _flush()

    print(f"[PDF] {len(rows)} linhas de produto extraídas")
    return rows


async def read_pdf_message(evt: dict) -> Optional[list[dict]]:
    """Entry point: dado o evento Infobip de documento, retorna as linhas extraídas ou None."""
    info = extract_document_info(evt)
    if not info:
        return None
    try:
        pdf_bytes = await download_document(media_url=info["media_url"], media_id=info["media_id"])
        print(f"[PDF] {len(pdf_bytes)} bytes baixados")
        # pdfplumber é síncrono e pode ser lento em PDFs grandes (várias páginas/tabelas).
        # Roda em thread separada para não bloquear o event loop (e travar TODAS as
        # mensagens do servidor enquanto processa).
        rows = await asyncio.to_thread(parse_movimentacao_pdf, pdf_bytes)
        return rows or None
    except Exception as e:
        print(f"[PDF] ERRO ao ler documento: {e}")
        return None
