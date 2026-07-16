from __future__ import annotations

import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL   = (os.getenv("OPENAI_MODEL_ROUTER") or "gpt-5.4-nano").strip()

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


class _QuantityOut(BaseModel):
    quantity: Optional[float] = None
    valid:    bool


_NUMERIC_RE = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*$")


async def parse_quantity(text: str, unit: str) -> Optional[float]:
    """
    Interpreta uma quantidade em texto livre. Retorna None se não conseguir.

    Exemplos:
      ("dois quilos e meio", "KG") → 2.5
      ("5 unidades",         "UN") → 5
      ("2,5",                "KG") → 2.5
      ("uma dúzia",          "UN") → 12
      ("meio quilo",         "KG") → 0.5
    """
    text = (text or "").strip()
    if not text:
        return None

    # Caminho rápido: número puro (com vírgula ou ponto)
    if _NUMERIC_RE.match(text):
        try:
            return float(text.replace(",", "."))
        except ValueError:
            pass

    # Fallback: pede pro LLM interpretar
    unit_norm = (unit or "UN").strip().upper()
    prompt = (
        "O usuário informou uma quantidade em linguagem natural para um produto "
        f"vendido em '{unit_norm}'. Extraia a quantidade numérica.\n\n"
        f'Texto: "{text}"\n\n'
        "Regras:\n"
        f"- Se '{unit_norm}' = 'UN' (unidade), retorne número inteiro "
        "(1 dúzia = 12, 1 par = 2, meia dúzia = 6).\n"
        f"- Se '{unit_norm}' = 'KG' (quilo), aceite frações "
        "(meio = 0.5, 'e meio' soma +0.5, 'um quilo e duzentos' = 1.2).\n"
        "- Se o texto for ambíguo ou não numérico, valid=false."
    )

    try:
        resp = await _get_client().beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=_QuantityOut,
        )
        out = resp.choices[0].message.parsed
        if out and out.valid and out.quantity is not None and out.quantity > 0:
            return float(out.quantity)
    except Exception as e:
        print(f"[quantity_parser] ERRO: {e}")

    return None
