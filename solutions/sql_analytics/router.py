from __future__ import annotations

import re
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.db_client import get_cnpjs_for_phone
from shared.session_store import (
    get_session,
    set_cnpj,
    set_pending_options,
    extend_session,
    clear_session,
)
from solutions.sql_analytics.graph_agent import build_agent

# ── Agent singleton ───────────────────────────────────────────────────────────

_agent = None


async def init(checkpointer=None):
    """Chamado no lifespan do app para inicializar o agente."""
    global _agent
    _agent = build_agent(checkpointer=checkpointer)
    # LangGraph 1.0.x substitui o checkpointer durante compile() — forçamos aqui
    if checkpointer is not None:
        _agent.checkpointer = checkpointer


# ── Schemas ───────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    phone:   str
    message: str
    cnpj:    Optional[str] = None  # portal envia direto; WhatsApp busca pelo phone


class MessageResponse(BaseModel):
    phone:        str
    answer:       str
    cnpj:         Optional[str] = None
    menu_options: Optional[list] = None
    messages:     Optional[list[str]] = None
    insight:         Optional[str] = None
    metric:          Optional[str] = None
    suggest_catalog: bool = False
    excel_url:       Optional[str] = None
    # sinais para a base de ML (instrumentação de NLU/erros)
    intent:          Optional[str] = None
    sql_generated:   Optional[str] = None
    sql_error:       Optional[str] = None
    had_error:       bool = False


# ── Helpers de menu / troca de CNPJ ───────────────────────────────────────────

_CHANGE_CNPJ_PATTERNS = [
    "mudar cnpj", "trocar cnpj", "alterar cnpj", "outro cnpj",
    "mudar loja", "trocar loja", "alterar loja", "outra loja",
    "mudar estabelecimento", "trocar estabelecimento", "outro estabelecimento",
    "mudar empresa", "trocar empresa", "outra empresa",
    "mudar filial", "trocar filial", "outra filial",
]


def _is_change_cnpj_command(message: str) -> bool:
    msg = (message or "").strip().lower()
    return any(pattern in msg for pattern in _CHANGE_CNPJ_PATTERNS)


def _format_cnpj_display(cnpj: str) -> str:
    c = re.sub(r"\D", "", cnpj or "")
    if len(c) != 14:
        return cnpj or ""
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:14]}"


_MENU_HEADER = "Identifiquei mais de um estabelecimento no seu número. Qual deles você quer consultar?"


def _build_menu(options: list[dict]) -> str:
    """Texto de fallback (portal ou quando botões falham)."""
    lines = [_MENU_HEADER, ""]
    for i, opt in enumerate(options, 1):
        name = opt.get("name") or _format_cnpj_display(opt.get("cnpj", ""))
        cnpj = _format_cnpj_display(opt.get("cnpj", ""))
        lines.append(f"{i}. {name} ({cnpj})")
    lines += ["", "Responda com o número da opção (ex: 1).",
              "Para trocar depois, envie 'mudar cnpj' ou 'mudar loja'."]
    return "\n".join(lines)


def _build_selection_confirmation(option: dict, user_name: str | None = None) -> str:
    name = option.get("name") or _format_cnpj_display(option.get("cnpj", ""))
    greeting = f"Olá, *{user_name}*!" if user_name else "Olá!"
    return (
        f"✅ Selecionado: *{name}*.\n\n"
        f"{greeting} Sou a *Inteligência Artificial do iMais*. O que gostaria de me perguntar?\n\n"
        f"_Você pode trocar de estabelecimento a qualquer momento enviando 'mudar cnpj' ou 'mudar loja'._"
    )


# ── Processamento do grafo ────────────────────────────────────────────────────

async def _process_graph(
    req: MessageRequest,
    cnpj: str,
    source: Literal["whatsapp", "portal"],
) -> MessageResponse:
    """Invoca o grafo LangGraph com o CNPJ resolvido."""
    if source == "portal":
        thread_id = f"portal_{cnpj}" if cnpj else f"portal_{req.phone}"
    else:
        thread_id = f"{req.phone}_{cnpj}" if cnpj else req.phone

    inp = {
        "phone":    req.phone or "",
        "cnpj":     cnpj or None,
        "messages": [{"role": "user", "content": req.message}],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        out = await _agent.ainvoke(inp, config=config)
        print("DEBUG out =", out)
    except Exception as e:
        print("ERRO no grafo:", e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    answer          = out.get("answer") or "Não consegui processar sua mensagem. Tente novamente."
    messages        = (
        out.get("output_messages")
        if out.get("output_messages_question") == req.message
        else None
    )
    insight         = out.get("insight") or None
    metric          = (out.get("extracted_params") or {}).get("metric") or None
    intent          = out.get("intent") or None
    suggest_catalog = intent == "out_of_scope"
    excel_url       = out.get("excel_url") or None
    sql_generated   = out.get("sql") or None
    sql_error       = out.get("sql_error") or None
    # heurística de erro: query falhou ou resposta é fallback ("não consegui ...")
    had_error       = bool(sql_error) or answer.strip().lower().startswith("não consegui")

    return MessageResponse(
        phone=req.phone or "",
        answer=answer,
        cnpj=cnpj or None,
        messages=messages,
        insight=insight,
        metric=metric,
        suggest_catalog=suggest_catalog,
        excel_url=excel_url,
        intent=intent,
        sql_generated=sql_generated,
        sql_error=sql_error,
        had_error=had_error,
    )


# ── Handler comum ─────────────────────────────────────────────────────────────

async def handle_message(req: MessageRequest, source: Literal["whatsapp", "portal"] = "whatsapp") -> MessageResponse:
    """Processa mensagem pelo grafo. Usado pelo WhatsApp e pelo portal."""
    if not req.message:
        raise HTTPException(status_code=400, detail="message é obrigatório.")

    # ── Portal: CNPJ vem direto da sessão do frontend, sem menu ───────────────
    if source == "portal":
        cnpj = (req.cnpj or "").strip()
        if not cnpj:
            raise HTTPException(status_code=400, detail="cnpj é obrigatório para o portal.")
        return await _process_graph(req, cnpj, source)

    # ── WhatsApp: lógica de seleção de CNPJ com sessão de 30 min ──────────────
    phone = (req.phone or "").strip()
    if not phone:
        raise HTTPException(status_code=400, detail="phone é obrigatório para WhatsApp.")

    message_stripped = req.message.strip()

    # 1) Comando de troca de CNPJ → limpa sessão e força menu de novo
    if _is_change_cnpj_command(message_stripped):
        clear_session(phone)
        cnpjs = await get_cnpjs_for_phone(phone)
        if len(cnpjs) == 0:
            return MessageResponse(
                phone=phone,
                answer="Olá! Esse número está vinculado à *Inteligência Artificial do iMais*. Caso tenha alguma dúvida sobre nossos serviços, entre em contato com o suporte:\n- WhatsApp: (34) 99912-7261\n- Capitais e regiões metropolitanas: 3003-1266\n- Outras regiões: 0800-729-5217",
                cnpj=None,
            )
        if len(cnpjs) == 1:
            set_cnpj(phone, cnpjs[0]["cnpj"])
            return MessageResponse(
                phone=phone,
                answer=_build_selection_confirmation(cnpjs[0]),
                cnpj=cnpjs[0]["cnpj"],
            )
        set_pending_options(phone, cnpjs)
        return MessageResponse(phone=phone, answer=_MENU_HEADER, cnpj=None, menu_options=cnpjs)

    session = get_session(phone)

    # 2) Sessão aguardando seleção — espera "1", "2", etc
    if session and session.get("pending_options"):
        options = session["pending_options"]
        try:
            idx = int(message_stripped) - 1
        except ValueError:
            # Salva a mensagem como pendente (para processar após seleção do CNPJ)
            pending_activation = session.get("pending_activation")
            pending_msg = session.get("pending_message") or message_stripped
            set_pending_options(phone, options, pending_message=pending_msg, pending_activation=pending_activation)
            return MessageResponse(
                phone=phone,
                answer=f"Por favor, selecione um estabelecimento respondendo com o número da opção.\n\n{_build_menu(options)}",
                cnpj=None,
            )
        if not (0 <= idx < len(options)):
            return MessageResponse(
                phone=phone,
                answer=f"Opção inválida. Digite um número entre 1 e {len(options)}.",
                cnpj=None,
            )
        selected = options[idx]
        pending_activation = session.get("pending_activation")
        pending_message    = session.get("pending_message")
        set_cnpj(phone, selected["cnpj"])

        confirmation = (
            f"✅ Selecionado: *{selected.get('name') or _format_cnpj_display(selected['cnpj'])}*.\n\n"
        )

        # Ativação pendente de outro módulo (ex: estoque)
        if pending_activation == "estoque":
            from solutions.estoque.router import init_estoque_session
            from shared.session_store import set_active_solution

            set_active_solution(phone, "estoque")
            menu_text = init_estoque_session(phone)
            return MessageResponse(
                phone=phone,
                answer=confirmation + menu_text,
                cnpj=selected["cnpj"],
                messages=[confirmation.strip(), menu_text],
            )

        # Pergunta original pendente — processa como mensagem separada da confirmação
        if pending_message:
            req_pending = MessageRequest(phone=req.phone, message=pending_message, cnpj=selected["cnpj"])
            resp = await _process_graph(req_pending, selected["cnpj"], source)
            return MessageResponse(
                phone=phone,
                answer=confirmation + resp.answer,
                cnpj=selected["cnpj"],
                messages=[confirmation.strip(), resp.answer],
                insight=resp.insight,
                metric=resp.metric,
            )

        # Busca nome do usuário pelo telefone para personalizar saudação
        _user_name: str | None = None
        if source == "whatsapp" and phone:
            try:
                from shared.db_client import get_user_name_for_phone
                _user_name = await get_user_name_for_phone(phone, selected["cnpj"])
            except Exception:
                pass

        return MessageResponse(
            phone=phone,
            answer=_build_selection_confirmation(selected, user_name=_user_name),
            cnpj=selected["cnpj"],
        )

    # 3) Sessão com CNPJ ativo → renova TTL e processa normalmente
    if session and session.get("cnpj"):
        cnpj = session["cnpj"]
        extend_session(phone)
        return await _process_graph(req, cnpj, source)

    # 4) Sem sessão → busca CNPJs do telefone
    cnpjs = await get_cnpjs_for_phone(phone)
    if len(cnpjs) == 0:
        return MessageResponse(
            phone=phone,
            answer="Olá! Esse número está vinculado à *Inteligência Artificial do iMais*. Caso tenha alguma dúvida sobre nossos serviços, entre em contato com o suporte:\n- WhatsApp: (34) 99912-7261\n- Capitais e regiões metropolitanas: 3003-1266\n- Outras regiões: 0800-729-5217",
            cnpj=None,
        )
    if len(cnpjs) == 1:
        cnpj = cnpjs[0]["cnpj"]
        set_cnpj(phone, cnpj)
        return await _process_graph(req, cnpj, source)

    # Múltiplos CNPJs → apresenta o menu (salva mensagem original para reprocessar após seleção)
    set_pending_options(phone, cnpjs, pending_message=message_stripped)
    return MessageResponse(phone=phone, answer=_MENU_HEADER, cnpj=None, menu_options=cnpjs)


# ── Router (chamada direta via API) ──────────────────────────────────────────

router = APIRouter(prefix="/sql-analytics", tags=["SQL Analytics"])


@router.post("/message", response_model=MessageResponse)
async def api_message(req: MessageRequest):
    return await handle_message(req)
