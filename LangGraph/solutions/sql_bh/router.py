from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from solutions.sql_bh.graph_agent import build_agent

# ── Agent singleton ───────────────────────────────────────────────────────────

_agent = None


def init():
    """Chamado no lifespan do app para inicializar o agente BH."""
    global _agent
    _agent = build_agent()


# ── Schemas ───────────────────────────────────────────────────────────────────

class BhMessageRequest(BaseModel):
    message: str


class BhMessageResponse(BaseModel):
    answer: str


# ── Handler ───────────────────────────────────────────────────────────────────

async def handle_message(req: BhMessageRequest) -> BhMessageResponse:
    """Processa mensagem pelo grafo BH. Sem CNPJ, sem phone — consulta direta."""
    if not req.message:
        raise HTTPException(status_code=400, detail="message é obrigatório.")

    # Thread fixo para POC (sem identificação de usuário)
    thread_id = "bh_poc"

    inp = {
        "messages": [{"role": "user", "content": req.message}],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        out = await _agent.ainvoke(inp, config=config)
        print("DEBUG BH out =", out)
    except Exception as e:
        print("ERRO no grafo BH:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return BhMessageResponse(
        answer=out.get("answer") or "Não consegui processar sua mensagem. Tente novamente.",
    )


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/bh", tags=["BH Sellout"])


@router.post("/message", response_model=BhMessageResponse)
async def api_message(req: BhMessageRequest):
    return await handle_message(req)
