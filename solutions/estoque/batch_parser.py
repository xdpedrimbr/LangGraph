"""Parser de movimentação de estoque em linguagem natural.

Transforma uma mensagem livre (texto, transcrição de áudio ou OCR de foto) como
"saiu 5kg de tomate, 3 alface e baixa 2 banana" em uma lista estruturada de
itens {ação, produto, quantidade}. Inclui um pré-filtro barato (sem LLM) para
decidir se a mensagem sequer parece uma movimentação, evitando chamar o LLM em
toda pergunta de analytics.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
# Extração estruturada da lista — gpt-4o-mini é barato e bem mais confiável que o
# nano para entender cabeçalho de seção (saiu/entrou) e nome completo do produto.
OPENAI_MODEL   = (os.getenv("OPENAI_MODEL_ESTOQUE") or "gpt-4o-mini").strip()

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


# ── Pré-filtro barato (sem LLM) ───────────────────────────────────────────────

# Verbos que indicam SAÍDA (baixa) ou ENTRADA (lançamento).
_BAIXA_VERBS = (
    "saiu", "sairam", "saíram", "saida", "saída",
    "vendi", "vendeu", "vendido", "vendidos", "vendemos", "vendas de",
    "baixa", "baixar", "baixei", "baixou", "dar baixa", "da baixa",
    "tirei", "tira", "tirar", "retira", "retirar", "retirei", "removi",
)
_LANCA_VERBS = (
    "entrou", "entraram", "entrada", "chegou", "chegaram", "chegada",
    "comprei", "comprou", "compramos", "recebi", "recebeu", "recebemos",
    "lancar", "lançar", "lancei", "lancou", "lançou", "lancamento", "lançamento",
    "repus", "repor", "abasteci",
)
_ALL_VERBS = _BAIXA_VERBS + _LANCA_VERBS

# Palavras de pergunta — se a frase for uma pergunta, NÃO é movimentação.
_QUESTION_WORDS = {"quanto", "quantos", "quantas", "qual", "quais", "como", "quando", "onde", "quem", "porque", "por que"}

_NUMBER_WORDS = {
    "um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco", "seis",
    "sete", "oito", "nove", "dez", "duzia", "dúzia", "meia", "meio",
}


def _norm(text: str) -> str:
    t = (text or "").lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def looks_like_movement(text: str) -> bool:
    """Heurística conservadora: a mensagem parece uma movimentação de estoque?

    Exige (a) um verbo de movimentação, (b) algum número (dígito ou por extenso)
    e (c) que NÃO seja uma pergunta. Mantém o LLM longe das perguntas de analytics.
    """
    raw = (text or "").strip()
    if not raw or len(raw) < 3:
        return False

    norm = _norm(raw)
    words = set(re.findall(r"[a-z]+", norm))

    # Pergunta explícita → não é movimentação ("quanto vendi de tomate?")
    if "?" in raw or (words & _QUESTION_WORDS):
        return False

    has_verb = any(v in norm for v in (_norm(v) for v in _ALL_VERBS))
    if not has_verb:
        return False

    has_number = bool(re.search(r"\d", norm)) or bool(words & _NUMBER_WORDS)
    return has_number


# ── Parser via LLM ────────────────────────────────────────────────────────────

class BatchItem(BaseModel):
    action:   Literal["baixar", "lancar"]
    product:  str
    quantity: Optional[float] = None
    unit:     Optional[str] = None


class BatchParseOut(BaseModel):
    is_movement: bool
    items:       list[BatchItem]


_SYSTEM = (
    "Você interpreta anotações de estoque de um supermercado brasileiro. O lojista "
    "manda, por texto/áudio/foto, uma lista do que SAIU (vendeu) ou ENTROU (comprou/recebeu) "
    "no estoque. Sua tarefa é extrair a lista estruturada de itens.\n\n"
    "AÇÃO por item:\n"
    "- 'baixar' (saída): saiu, vendi, vendeu, baixa, baixar, tirei, retirei, removi.\n"
    "- 'lancar' (entrada): entrou, chegou, comprei, recebi, lancei, repus, abasteci.\n"
    "- ⚠️ CABEÇALHO DE SEÇÃO: quando uma palavra de ação aparece e é seguida de vários itens "
    "(uma lista), essa ação vale para TODOS os itens seguintes ATÉ aparecer outra palavra de ação. "
    "Ex: 'Saiu: 2 alface, 2 almeirão. Entrou: 1 alface, 5 tomate' → os 2 primeiros são 'baixar' e "
    "os 2 últimos são 'lancar'.\n"
    "- Se NÃO houver verbo nenhum em toda a mensagem, use a ação padrão 'baixar'.\n\n"
    "Para cada item extraia:\n"
    "- product: o NOME COMPLETO do produto exatamente como escrito, incluindo variedade, marca, "
    "tamanho/peso e embalagem. NUNCA encurte para a primeira palavra. "
    "Ex: '5 tomate cereja bandeja 300g' → product='tomate cereja bandeja 300g' (NÃO 'tomate'). "
    "Remova APENAS a quantidade e a unidade solta do início.\n"
    "- quantity: número (null se não informado). Vírgula é separador DECIMAL no padrão "
    "brasileiro, não de milhar: '3,292' significa 3.292 (não 3292).\n"
    "- unit: KG, UN, CX, etc. se informado claramente como unidade de medida; senão null. "
    "Atenção: 'bandeja 300g' faz parte do NOME do produto, não é a unidade.\n\n"
    "Exemplos:\n"
    "- 'saiu 5kg de tomate italiano, 3 alface lisa e baixa 2 banana prata' → 3 itens baixar: "
    "tomate italiano(5,KG), alface lisa(3,null), banana prata(2,null)\n"
    "- 'Saiu 2 alface lisa, 2 tomate mel. Entrou 5 tomate cereja bandeja 300g' → "
    "baixar: alface lisa(2), tomate mel(2); lancar: tomate cereja bandeja 300g(5)\n"
    "- 'chegou 10 caixas de leite integral' → 1 item lancar: leite integral(10,CX)\n\n"
    "is_movement = false se a mensagem for uma PERGUNTA ou análise (ex: 'quanto vendi de tomate?', "
    "'como estão minhas vendas?') e não uma anotação de movimentação. Nesse caso, items = [].\n"
    "Retorne SOMENTE o JSON estruturado."
)


async def parse_movement(text: str, *, force: bool = False) -> BatchParseOut:
    """Extrai os itens de movimentação da mensagem.

    Se `force=False` e o pré-filtro indicar que não parece movimentação, retorna
    is_movement=False sem chamar o LLM. Use `force=True` para fotos/listas (onde o
    cliente claramente enviou uma lista, mesmo sem verbos).
    """
    if not force and not looks_like_movement(text):
        return BatchParseOut(is_movement=False, items=[])

    try:
        resp = await _get_client().beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": (text or "").strip()},
            ],
            response_format=BatchParseOut,
        )
        out = resp.choices[0].message.parsed
        if out is None:
            return BatchParseOut(is_movement=False, items=[])

        # Em modo force (foto), confiamos na lista mesmo que o LLM hesite no flag.
        if force and out.items:
            out.is_movement = True
        return out
    except Exception as e:
        print(f"[batch_parser] ERRO: {e}")
        return BatchParseOut(is_movement=False, items=[])
