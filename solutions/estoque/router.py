from __future__ import annotations

import asyncio
import os
import re
from typing import Literal

from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel

import unicodedata

from shared.session_store import (
    clear_estoque_state,
    get_active_solution,
    get_estoque_state,
    get_session,
    set_active_solution,
    update_estoque_state,
)
from solutions.estoque.batch_parser import parse_movement
from solutions.estoque.db_ops import (
    get_current_stock,
    get_srk_cli,
    search_products,
    update_stock,
)
from solutions.estoque.quantity_parser import parse_quantity
from shared.solution_router import DEFAULT_SOLUTION


# ── Schemas (compartilhados com sql_analytics para uniformidade) ──────────────

class MessageRequest(BaseModel):
    phone:   str
    message: str
    cnpj:    str | None = None


class MessageResponse(BaseModel):
    phone:        str
    answer:       str
    cnpj:         str | None = None
    menu_options: list | None = None


# ── Estados da máquina ────────────────────────────────────────────────────────

STEP_MENU              = "MENU"
STEP_AWAIT_PRODUCT     = "AWAIT_PRODUCT"
STEP_AWAIT_CHOICE      = "AWAIT_CHOICE"
STEP_AWAIT_QUANTITY    = "AWAIT_QUANTITY"
STEP_AWAIT_CONFIRMATION = "AWAIT_CONFIRMATION"
STEP_AWAIT_CONTINUE    = "AWAIT_CONTINUE"
STEP_BATCH_RESOLVE     = "BATCH_RESOLVE"
STEP_BATCH_CONFIRM     = "BATCH_CONFIRM"
STEP_BATCH_EDIT_MENU     = "BATCH_EDIT_MENU"
STEP_BATCH_EDIT_PRODUCT  = "BATCH_EDIT_PRODUCT"
STEP_BATCH_EDIT_QUANTITY = "BATCH_EDIT_QUANTITY"
STEP_AWAIT_PRODUCT_QTY  = "AWAIT_PRODUCT_QTY"

ACTION_BAIXAR = "baixar"
ACTION_LANCAR = "lancar"

# Mapeamento número do menu → (TIPO_OPERACAO, label de exibição)
OPERACOES: dict[str, tuple[str, str]] = {
    "1": ("LIXO",             "Lixo ou Perda"),
    "2": ("USO_INTERNO",      "Uso Interno"),
    "3": ("PRODUCAO_INTERNA", "Produção Interna"),
    "4": ("LANCHE",           "Lanche ou Refeição"),
    "5": ("DOACAO",           "Doação ou Brinde"),
    "6": ("DEGUSTACAO",       "Degustação"),
    "7": ("ROUBO",            "Roubo"),
    "8": ("BAIXA",            "Baixa"),
}


def init():
    """Hook do lifespan. Sem inicialização específica."""
    pass


# ── Helpers de exibição ───────────────────────────────────────────────────────

def _menu_text() -> str:
    ops = "\n".join(f"{num} - {label}" for num, (_, label) in OPERACOES.items())
    return (
        "*Módulo Estoque* 📦\n\n"
        "Selecione o tipo de movimentação:\n\n"
        f"{ops}\n\n"
        "_Para sair a qualquer momento, envie 'sair estoque'._"
    )


def _action_label(action: str) -> str:
    """Label amigável para exibição dado o TIPO_OPERACAO ou ACTION_* do batch."""
    for _, (tipo, label) in OPERACOES.items():
        if action == tipo:
            return label
    return "baixar" if action == ACTION_BAIXAR else "lançar"


_PAGE_SIZE = 10


def _total_pages(options: list[dict]) -> int:
    if not options:
        return 0
    return max(1, (len(options) + _PAGE_SIZE - 1) // _PAGE_SIZE)


def _format_options(options: list[dict], page: int = 0) -> str:
    """Renderiza uma página de produtos (10 por página) com CEAN na frente
    e rodapé de paginação quando houver mais de uma página.
    """
    if not options:
        return ""

    total = _total_pages(options)
    page = max(0, min(page, total - 1))
    start = page * _PAGE_SIZE
    items = options[start:start + _PAGE_SIZE]

    lines = []
    for i, opt in enumerate(items, 1):
        cean = (opt.get("cean") or "").strip()
        desc = (opt.get("descricao") or "")[:35]
        unit = opt.get("unidade") or "UN"
        qtd  = float(opt.get("qnt_estoque") or 0)
        prefix = f"{cean} - " if cean else ""
        lines.append(f"{i}. {prefix}{desc} - {qtd:g} {unit}")

    text = "\n".join(lines)

    if total > 1:
        nav = []
        if page < total - 1:
            nav.append("'próximo' para ver mais")
        if page > 0:
            nav.append("'anterior' para página anterior")
        text += f"\n\n_Página {page + 1}/{total}"
        if nav:
            text += " — " + " ou ".join(nav)
        text += "._"

    return text


def _action_word(action: str) -> str:
    return "baixar" if action == ACTION_BAIXAR else "lançar"


def _session_key_for(phone: str, cnpj: str | None, source: str) -> str:
    if source == "whatsapp":
        return phone or ""
    return f"portal_{cnpj or ''}"


def _ok(req: MessageRequest, answer: str) -> MessageResponse:
    return MessageResponse(phone=req.phone or "", answer=answer, cnpj=req.cnpj)


# ── Parsing de "produto + quantidade" em mensagem única ──────────────────────

_UNIT_ALIASES_PARSE: dict[str, str] = {
    "kg": "KG", "kilo": "KG", "kilos": "KG", "quilo": "KG", "quilos": "KG",
    "g":  "G",  "gr":   "G",  "grama": "G",  "gramas": "G",
    "un": "UN", "unidade": "UN", "unidades": "UN",
}

# Captura: (produto) (numero) [unidade] [e meio]
# Ex: "banana prata 800", "3354 1.5kg", "alface 1 quilo e meio"
_TRAILING_QTY_RE = re.compile(
    r'^(.*?)\s+(\d+(?:[.,]\d+)?)\s*'
    r'(kg|kilo|kilos|quilo|quilos|gr?|grama|gramas|un|unidade|unidades)?'
    r'(?:\s+e\s+meio)?\s*$',
    re.IGNORECASE,
)


def _parse_product_qty_message(message: str) -> tuple[str, float | None, str | None]:
    """Extrai (produto, qty_raw, unit_raw) de uma mensagem "produto peso".

    qty_raw: o número como digitado (sem conversão de unidade).
    unit_raw: "KG", "G", "UN" ou None (None = gramas assumed).
    """
    msg = (message or "").strip()
    m = _TRAILING_QTY_RE.match(msg)
    if not m or not m.group(1).strip():
        return msg, None, None

    product = m.group(1).strip()
    try:
        num = float(m.group(2).replace(",", "."))
    except (ValueError, AttributeError):
        return msg, None, None

    if re.search(r"\be\s+meio\b", msg, re.IGNORECASE):
        num += 0.5

    unit_raw = _UNIT_ALIASES_PARSE.get((m.group(3) or "").lower())
    return product, num, unit_raw


def _normalize_qty_for_unit(raw_qty: float, unit_raw: str | None, product_unit: str) -> float:
    """Converte a quantidade inserida para a unidade do produto no estoque.

    Se unit_raw é None ou "G" e o produto é KG, divide por 1000 (grama → KG).
    Se unit_raw é "KG", usa direto.
    Se produto é UN, usa o número sem conversão.
    """
    pu = product_unit.upper()
    if unit_raw == "KG":
        return raw_qty
    if unit_raw in ("G", None):
        return raw_qty / 1000 if pu == "KG" else raw_qty
    return raw_qty  # UN ou outro


def _parse_product_qty_list(message: str) -> list[tuple[str, float | None, str | None]]:
    """Divide uma mensagem em segmentos (produto, qty, unit) para lotes.

    Separa por quebras de linha primeiro; se houver só uma linha, separa por
    vírgula seguida de não-dígito (evita quebrar decimais como '1,5kg').
    """
    msg = (message or "").strip()
    lines = [ln.strip() for ln in msg.split("\n") if ln.strip()]
    if len(lines) <= 1:
        lines = [s.strip() for s in re.split(r",\s*(?=[^\d])", msg) if s.strip()]
    return [_parse_product_qty_message(ln) for ln in lines if ln]


# ── Detecção de pergunta off-topic (analytics) ────────────────────────────────
# Usado quando o usuário envia uma pergunta de vendas/faturamento dentro do
# módulo de estoque — respondemos com um lembrete amigável.

# Sinais fortes: termo praticamente inexistente em nome de produto.
_ANALYTICS_HARD_KEYWORDS = (
    "faturamento", "fatura ",
    "ticket médio", "ticket medio",
    "diagnóstico", "diagnostico",
    "previsão", "previsao",
    "tendência", "tendencia",
    "curva abc",
    "concorrent", "concorrência", "concorrencia",
)

# Sinais fracos: só dispara se a frase tiver formato de pergunta.
_ANALYTICS_SOFT_KEYWORDS = (
    "vendas", "vendi", "vender",
    "comprei", "compras",
    "gastos", "gastei",
    "margem", "lucro",
    "ranking", "mais vendido", "menos vendido",
    "mercado",
    "comparar", "comparação", "comparacao",
    "análise", "analise",
)

_QUESTION_WORDS = {"qual", "quanto", "quantos", "quantas", "quando", "como", "porque"}


def _looks_like_analytics_question(message: str) -> bool:
    """Heurística pra detectar pergunta de analytics (vendas/faturamento) no
    meio do fluxo de estoque. Conservadora: não dispara em nomes de produto curtos.
    """
    msg = (message or "").lower().strip()
    if not msg:
        return False

    # Termo forte → dispara
    if any(kw in msg for kw in _ANALYTICS_HARD_KEYWORDS):
        return True

    # Termo fraco + estrutura de pergunta → dispara
    has_question_mark = "?" in msg
    has_question_word = any(w in _QUESTION_WORDS for w in msg.split())
    if (has_question_mark or has_question_word) and any(kw in msg for kw in _ANALYTICS_SOFT_KEYWORDS):
        return True

    return False


def _off_topic_message() -> str:
    return (
        "📦 Você está no *módulo de estoque*.\n\n"
        "Pra fazer perguntas sobre vendas, faturamento ou outras análises, "
        "primeiro saia digitando *sair estoque*."
    )


# ── Comando "voltar" (retrocede um estado) ────────────────────────────────────

_BACK_COMMANDS = {"voltar", "menu anterior", "back"}

# Comandos de paginação dentro do AWAIT_CHOICE (não confundir com voltar de menu).
_NEXT_PAGE_COMMANDS = {"próximo", "proximo", "next", "mais", "+"}
_PREV_PAGE_COMMANDS = {"anterior", "ant", "previous", "-"}



def _is_back_command(message: str) -> bool:
    return (message or "").strip().lower() in _BACK_COMMANDS


def _ask_product_text(action: str) -> str:
    label = _action_label(action)
    return (
        f"Qual produto para *{label}*?\n\n"
        f"_Digite parte do nome (ex: 'alface') ou 'voltar' para o menu._"
    )


# ── Inicialização e reset (chamados pelo dispatcher) ──────────────────────────

def init_estoque_session(session_key: str) -> str:
    """Reseta o estado e retorna o texto do menu inicial."""
    clear_estoque_state(session_key)
    update_estoque_state(session_key, step=STEP_MENU)
    return _menu_text()


# ── Handlers de cada estado ───────────────────────────────────────────────────

async def _handle_menu(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip()

    op = OPERACOES.get(msg)
    if op:
        tipo, label = op
        update_estoque_state(session_key, step=STEP_AWAIT_PRODUCT_QTY, action=tipo)
        return _ok(req, f"*{label}* selecionado.\n\nInforme o(s) produto(s).")

    if _looks_like_analytics_question(msg):
        return _ok(req, _off_topic_message())

    return _ok(req, f"Opção inválida. Digite um número de 1 a 8.\n\n{_menu_text()}")


async def _ensure_srk_cli(req: MessageRequest, session_key: str) -> int | None:
    """Resolve e cacheia o SRK_CLI a partir do CNPJ.

    Ordem de resolução:
    1. Cache na sessão de estoque (srk_cli já calculado)
    2. CNPJ do request (portal)
    3. CNPJ da sessão principal do WhatsApp (sql_analytics o guarda lá)
    4. Resolução pelo número de telefone (usuário entrou direto sem sessão)
    """
    state = get_estoque_state(session_key)
    srk = state.get("srk_cli")
    if srk:
        return int(srk)

    cnpj = (req.cnpj or "").strip()

    # Passo 3: CNPJ na sessão (definido pelo sql_analytics no WhatsApp)
    if not cnpj:
        from shared.session_store import get_session
        session = get_session(session_key)
        cnpj = ((session or {}).get("cnpj") or "").strip()

    # Passo 4: resolve pelo telefone (usuário nunca usou analytics antes)
    if not cnpj and req.phone:
        from shared.db_client import get_cnpj_for_phone
        cnpj = (await get_cnpj_for_phone(req.phone) or "").strip()

    if not cnpj:
        return None

    srk = await get_srk_cli(cnpj)
    if srk is None:
        return None
    update_estoque_state(session_key, srk_cli=int(srk))
    return int(srk)


async def _build_confirmation_from_product(
    req: MessageRequest,
    session_key: str,
    selected: dict,
    raw_qty: float | None,
    unit_raw: str | None,
    action: str,
    srk_cli: int,
) -> MessageResponse:
    """Monta a tela de confirmação após o produto ter sido resolvido."""
    unit = (selected.get("unidade") or "UN").upper()

    if raw_qty is None:
        return _ok(req,
            f"Não encontrei a quantidade na sua mensagem.\n\n"
            f"Informe novamente com a quantidade. Ex:\n"
            f"_{selected['descricao']} 800g_ ou _{selected['descricao']} 1.5kg_"
        )

    qty = _normalize_qty_for_unit(raw_qty, unit_raw, unit)

    if qty <= 0:
        return _ok(req, "A quantidade precisa ser maior que zero. Tente novamente.")

    if unit == "UN" and qty != int(qty):
        return _ok(req, "Para produtos em UN a quantidade precisa ser inteira. Tente novamente.")

    fresh = await get_current_stock(int(srk_cli), selected["codigo"])
    if fresh is None:
        return _ok(req, f"O produto *{selected.get('descricao', '')}* não foi encontrado. Tente outro.")

    current = float(fresh.get("qnt_estoque") or 0)
    delta    = -qty
    expected = current + delta
    cean     = (selected.get("cean") or "").strip()
    codigo   = (selected.get("codigo") or "").strip()
    id_label = cean if cean else codigo

    update_estoque_state(
        session_key,
        step=STEP_AWAIT_CONFIRMATION,
        selected=selected,
        pending={
            "qty": qty, "delta": delta,
            "current": current, "expected": expected, "unit": unit,
        },
    )

    return _ok(req,
        f"📋 *Confirme a operação:*\n\n"
        f"Produto: *{id_label} — {selected['descricao']}*\n"
        f"Tipo: *{_action_label(action)}*\n"
        f"Quantidade: {qty:g} {unit}\n\n"
        f"_Responda *sim* para confirmar ou *não* para cancelar._"
    )


async def _handle_await_product_qty(req: MessageRequest, session_key: str) -> MessageResponse:
    """Recebe 'produto quantidade' em mensagem única, resolve produto e monta confirmação."""
    msg = (req.message or "").strip()

    if _is_back_command(msg):
        update_estoque_state(session_key, step=STEP_MENU, action=None)
        return _ok(req, f"Voltando ao menu.\n\n{_menu_text()}")

    if _looks_like_analytics_question(msg):
        return _ok(req, _off_topic_message())

    state   = get_estoque_state(session_key)
    action  = state.get("action") or "BAIXA"
    srk_cli = await _ensure_srk_cli(req, session_key)
    if srk_cli is None:
        return _ok(req, "Não consegui identificar seu cadastro. Entre em contato com o suporte.")

    segments = _parse_product_qty_list(msg)

    # ── Lote (múltiplos produtos) ────────────────────────────────────────────
    if len(segments) > 1:
        async def _cat(seg: tuple) -> dict:
            product_text, raw_qty, unit_raw = seg
            item = await _categorize_batch_item(srk_cli, {
                "action": ACTION_BAIXAR,
                "product": product_text,
                "quantity": raw_qty,
                "unit": unit_raw,
            }, exclude_zero_stock=False)
            # Normaliza grama → KG após resolver o produto
            if item.get("selected") and item.get("quantity") is not None:
                pu = (item["selected"].get("unidade") or "UN").upper()
                item["quantity"] = _normalize_qty_for_unit(float(item["quantity"]), unit_raw, pu)
            return item

        items = await asyncio.gather(*(_cat(s) for s in segments))
        prev = get_active_solution(session_key)
        batch = {
            "items": list(items),
            "srk_cli": srk_cli,
            "prev_solution": prev,
            "tipo_operacao": action,
        }
        set_active_solution(session_key, "estoque")
        return _ok(req, _advance_batch(session_key, batch))

    # ── Produto único ────────────────────────────────────────────────────────
    product_text, raw_qty, unit_raw = segments[0] if segments else (msg, None, None)

    if not product_text or len(product_text.strip()) < 1:
        return _ok(req,
            "Não entendi. Informe o produto e a quantidade juntos.\n"
            "_Ex: banana prata 800 | alface 1.5kg | 3354 800g_"
        )

    # Código puro (numérico) → tenta busca exata por código primeiro
    product_text = product_text.strip()
    is_code = re.sub(r"\D", "", product_text) == re.sub(r"\s", "", product_text) and product_text.replace(" ", "").isdigit()

    selected: dict | None = None
    if is_code:
        from solutions.estoque.db_ops import get_product_by_code
        selected = await get_product_by_code(srk_cli, product_text)

    if selected is not None:
        return await _build_confirmation_from_product(req, session_key, selected, raw_qty, unit_raw, action, srk_cli)

    # Busca por nome
    options = await search_products(srk_cli, product_text, exclude_zero_stock=False)
    if not options:
        return _ok(req, f"Não encontrei nenhum produto com '{product_text}'. Tente outro nome ou código.")

    exact = [o for o in options if _norm_desc(o.get("descricao")) == _norm_desc(product_text)]
    if len(options) == 1 or len(exact) == 1:
        chosen = exact[0] if len(exact) == 1 else options[0]
        return await _build_confirmation_from_product(req, session_key, chosen, raw_qty, unit_raw, action, srk_cli)

    smart = await _llm_pick_from_candidates(product_text, options[:_BATCH_CANDIDATE_LIMIT])
    if smart:
        return await _build_confirmation_from_product(req, session_key, smart, raw_qty, unit_raw, action, srk_cli)

    update_estoque_state(
        session_key,
        step=STEP_AWAIT_CHOICE,
        options=options, page=0,
        pending_qty={"raw": raw_qty, "unit_raw": unit_raw},
    )
    return _ok(req,
        f"Não tenho certeza de qual produto é *{product_text}*. Qual destes?\n\n"
        f"{_format_options(options, 0)}\n\n"
        f"_Digite o número do produto, ou 'voltar' para tentar novamente._"
    )


async def _handle_await_product(req: MessageRequest, session_key: str) -> MessageResponse:
    term = (req.message or "").strip()

    if _is_back_command(term):
        update_estoque_state(
            session_key, step=STEP_MENU,
            action=None, options=None, selected=None, pending=None,
        )
        return _ok(req, "Voltando ao menu.\n\n" + _menu_text())

    if _looks_like_analytics_question(term):
        return _ok(req, _off_topic_message())

    if len(term) < 2:
        return _ok(req, "Digite ao menos 2 caracteres do nome do produto.")

    srk_cli = await _ensure_srk_cli(req, session_key)
    if srk_cli is None:
        return _ok(req, "Não consegui identificar seu cadastro. Entre em contato com o suporte.")

    options = await search_products(srk_cli, term, exclude_zero_stock=True)
    if not options:
        all_matches = await search_products(srk_cli, term, exclude_zero_stock=False)
        if all_matches:
            return _ok(req,
                f"Encontrei {len(all_matches)} produto(s) com '{term}', "
                f"mas todos estão com saldo zerado. Não é possível registrar esta movimentação.\n\n"
                f"_Digite outro termo ou envie 'sair estoque'._"
            )
        return _ok(req, f"Não encontrei nenhum produto com '{term}' no seu estoque. Tente outro termo ou envie 'sair estoque'.")

    # Reseta paginação ao iniciar uma nova busca.
    update_estoque_state(session_key, step=STEP_AWAIT_CHOICE, options=options, page=0)
    return _ok(req,
        f"Encontrei {len(options)} produto(s):\n\n"
        f"{_format_options(options, 0)}\n\n"
        f"_Digite o número do produto desejado, ou 'voltar' para buscar outro._"
    )


async def _handle_await_choice(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip().lower()
    state = get_estoque_state(session_key)
    options = state.get("options") or []
    action = state.get("action") or "BAIXA"
    page = int(state.get("page") or 0)
    total = _total_pages(options)

    if _is_back_command(msg):
        update_estoque_state(
            session_key, step=STEP_AWAIT_PRODUCT_QTY,
            options=None, selected=None, page=None, pending_qty=None,
        )
        return _ok(req,
            "Tudo bem. Informe novamente o produto e a quantidade:\n"
            "_Ex: banana prata 800 | alface 1.5kg | 3354 800g_"
        )

    # Paginação
    if msg in _NEXT_PAGE_COMMANDS:
        if page + 1 < total:
            new_page = page + 1
            update_estoque_state(session_key, page=new_page)
            return _ok(req,
                f"{_format_options(options, new_page)}\n\n"
                f"_Digite o número do produto desejado, ou 'voltar' para buscar outro._"
            )
        return _ok(req, f"Você já está na última página ({total}/{total}).")

    if msg in _PREV_PAGE_COMMANDS:
        if page > 0:
            new_page = page - 1
            update_estoque_state(session_key, page=new_page)
            return _ok(req,
                f"{_format_options(options, new_page)}\n\n"
                f"_Digite o número do produto desejado, ou 'voltar' para buscar outro._"
            )
        return _ok(req, "Você já está na primeira página.")

    # Seleção numérica (1..N dentro da página)
    try:
        idx_in_page = int(msg) - 1
    except ValueError:
        return _ok(req, f"Digite o número da opção desejada.\n\n{_format_options(options, page)}")

    page_items = min(_PAGE_SIZE, len(options) - page * _PAGE_SIZE)
    if not (0 <= idx_in_page < page_items):
        return _ok(req, f"Opção inválida. Digite um número entre 1 e {page_items}.")

    global_idx = page * _PAGE_SIZE + idx_in_page
    selected = options[global_idx]

    srk_cli = state.get("srk_cli") or await _ensure_srk_cli(req, session_key)
    if srk_cli is None:
        return _ok(req, "Não consegui identificar seu cadastro. Entre em contato com o suporte.")

    pending_qty = state.get("pending_qty") or {}
    return await _build_confirmation_from_product(
        req, session_key, selected,
        pending_qty.get("raw"), pending_qty.get("unit_raw"),
        action, srk_cli,
    )


async def _handle_await_quantity(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip()
    state = get_estoque_state(session_key)
    selected = state.get("selected") or {}
    action   = state.get("action") or "BAIXA"
    srk_cli  = state.get("srk_cli")

    if not selected or srk_cli is None:
        text = init_estoque_session(session_key)
        return _ok(req, "Algo deu errado. Vamos começar de novo.\n\n" + text)

    if _is_back_command(msg):
        options = state.get("options") or []
        if options:
            page = int(state.get("page") or 0)
            update_estoque_state(session_key, step=STEP_AWAIT_CHOICE, selected=None)
            return _ok(req,
                f"Voltando à seleção:\n\n{_format_options(options, page)}\n\n"
                f"_Digite o número do produto desejado, ou 'voltar' para buscar outro._"
            )
        update_estoque_state(session_key, step=STEP_AWAIT_PRODUCT, selected=None, page=None)
        return _ok(req, "Voltando.\n\n" + _ask_product_text(action))

    unit = (selected.get("unidade") or "UN").upper()
    qty = await parse_quantity(msg, unit)
    if qty is None or qty <= 0:
        return _ok(req, f"Não consegui interpretar a quantidade '{msg}'. Tente novamente (ex: 5, 2.5, 'dois quilos e meio').")

    if unit == "UN" and qty != int(qty):
        return _ok(req, f"Para produtos vendidos em UN, a quantidade precisa ser inteira. Tente novamente.")

    # ── Verificação A: re-busca o produto p/ confirmar que existe e pegar saldo fresco
    fresh = await get_current_stock(int(srk_cli), selected["codigo"])
    if fresh is None:
        update_estoque_state(session_key, step=STEP_AWAIT_PRODUCT, selected=None)
        return _ok(req,
            f"O produto *{selected.get('descricao','')}* não foi encontrado no seu estoque "
            f"(pode ter sido removido). Digite outro produto ou envie 'sair estoque'."
        )

    current = float(fresh.get("qnt_estoque") or 0)
    delta = -qty
    expected = current + delta

    # Salva o pendente e mostra preview p/ confirmação
    update_estoque_state(
        session_key,
        step=STEP_AWAIT_CONFIRMATION,
        pending={
            "qty": qty,
            "delta": delta,
            "current": current,
            "expected": expected,
            "unit": unit,
        },
    )

    return _ok(req,
        f"📋 *Confirme a operação:*\n\n"
        f"Produto: *{selected['descricao']}*\n"
        f"Tipo: *{_action_label(action)}*\n"
        f"Quantidade: {qty:g} {unit}\n\n"
        f"_Responda *'sim'* para confirmar, *'não'* para corrigir a quantidade, "
        f"ou *'voltar'* para escolher outro produto._"
    )


_CONFIRM_YES = {"sim", "s", "ok", "confirma", "confirmar", "1"}
_CONFIRM_NO  = {"nao", "não", "n", "cancela", "cancelar", "2"}


async def _handle_await_confirmation(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip().lower()
    state = get_estoque_state(session_key)
    selected = state.get("selected") or {}
    action   = state.get("action") or "BAIXA"
    srk_cli  = state.get("srk_cli")
    pending  = state.get("pending") or {}

    if not selected or srk_cli is None or not pending:
        text = init_estoque_session(session_key)
        return _ok(req, "Algo deu errado. Vamos começar de novo.\n\n" + text)

    if _is_back_command(msg):
        options = state.get("options") or []
        if options:
            page = int(state.get("page") or 0)
            update_estoque_state(session_key, step=STEP_AWAIT_CHOICE, selected=None, pending=None)
            return _ok(req,
                f"Voltando à seleção:\n\n{_format_options(options, page)}\n\n"
                f"_Digite o número do produto desejado, ou 'voltar' para buscar outro._"
            )
        update_estoque_state(session_key, step=STEP_AWAIT_PRODUCT, selected=None, pending=None, page=None)
        return _ok(req, "Voltando.\n\n" + _ask_product_text(action))

    if msg in _CONFIRM_NO:
        update_estoque_state(session_key, step=STEP_AWAIT_QUANTITY, pending=None)
        unit = pending.get("unit", "UN")
        current = float(pending.get("current") or 0)
        return _ok(req,
            f"Operação cancelada. Saldo atual: {current:g} {unit}.\n\n"
            f"Digite uma nova quantidade para *{_action_label(action)}* em *{selected['descricao']}*."
        )

    if msg not in _CONFIRM_YES:
        return _ok(req, "Não entendi. Responda *sim* para confirmar, *não* para corrigir a quantidade, ou 'voltar' para escolher outro produto.")

    qty      = float(pending.get("qty") or 0)
    delta    = float(pending.get("delta") or 0)
    expected = float(pending.get("expected") or 0)
    unit     = pending.get("unit") or "UN"

    try:
        await update_stock(int(srk_cli), selected["codigo"], delta, tipo_operacao=action)
    except Exception as e:
        print(f"[estoque] ERRO update_stock: {e}")
        return _ok(req, "Não consegui registrar a movimentação agora. Tente novamente em instantes.")

    # ── Verificação B: re-busca para confirmar que o INSERT foi refletido
    after = await get_current_stock(int(srk_cli), selected["codigo"])
    if after is None:
        print(f"[estoque] AVISO: produto não encontrado após INSERT | codigo={selected['codigo']}")
        return _ok(req, "A operação foi enviada, mas não consegui confirmar o novo saldo. Verifique no portal.")

    new_balance = float(after.get("qnt_estoque") or 0)
    if abs(new_balance - expected) > 1e-6:
        print(f"[estoque] AVISO: saldo após INSERT divergente | esperado={expected} obtido={new_balance}")
        return _ok(req,
            f"⚠️ A operação foi registrada, mas o saldo final ficou diferente do esperado.\n"
            f"*{selected['descricao']}*\n"
            f"Saldo atual: {new_balance:g} {unit} (esperado: {expected:g})\n\n"
            f"_Verifique no portal. Pode haver outra movimentação em andamento._"
        )

    update_estoque_state(
        session_key, step=STEP_AWAIT_CONTINUE,
        selected=None, pending=None, options=None, page=None,
    )

    return _ok(req,
        f"✅ *{_action_label(action)}* registrada com sucesso!\n"
        f"*{selected['descricao']}* — {qty:g} {unit}\n"
        f"Novo saldo: {new_balance:g} {unit}\n\n"
        f"Quer registrar mais alguma movimentação?\n"
        f"_Informe o próximo produto e quantidade (ex: alface 800g), 'menu' para ver as opções ou 'sair estoque' para finalizar._"
    )


async def _handle_await_continue(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip().lower()

    if msg in ("nao", "não", "n", "no"):
        return _ok(req, "Para finalizar o módulo de estoque, envie 'sair estoque'.")

    if _looks_like_analytics_question(req.message or ""):
        return _ok(req, _off_topic_message())

    # "menu" ou "voltar" → volta ao menu principal para escolher outro tipo
    if msg in ("menu", "voltar", "voltar ao menu", "opções", "opcoes"):
        update_estoque_state(session_key, step=STEP_MENU, action=None)
        return _ok(req, f"Voltando ao menu.\n\n{_menu_text()}")

    # Qualquer outra mensagem é tratada como produto+quantidade, mantendo o tipo atual.
    update_estoque_state(session_key, step=STEP_AWAIT_PRODUCT_QTY)
    return await _handle_await_product_qty(req, session_key)


# ── Fluxo rápido / em lote (texto, áudio ou foto) ─────────────────────────────
# Permite "saiu 5kg tomate, 3 alface, baixa 2 banana" ou foto com a lista, sem
# passar pelo menu. Reconhece os produtos, pede desambiguação quando não tem
# certeza e confirma tudo de uma vez.

_BATCH_CANDIDATE_LIMIT = 8   # quantos semelhantes mostrar na desambiguação

# ── Resolução inteligente via LLM quando busca normal não acha nada ───────────

_SMART_RESOLVE_MODEL = (os.getenv("OPENAI_MODEL_ESTOQUE") or "gpt-4o-mini").strip()
_MEASURE_RE = re.compile(r'^\d+[a-z]*$')   # tokens como "300g", "500g", "1"

_smart_openai: AsyncOpenAI | None = None


def _get_smart_openai() -> AsyncOpenAI:
    global _smart_openai
    if _smart_openai is None:
        _smart_openai = AsyncOpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
    return _smart_openai


async def _llm_pick_from_candidates(ocr_name: str, candidates: list[dict]) -> dict | None:
    """Pede ao LLM que escolha o produto mais provável dado o nome OCR e a lista de candidatos.

    Retorna o produto selecionado ou None se o LLM considerar genuinamente ambíguo (responde 0).
    Usado tanto em casos ambiguous (múltiplos matches) quanto not_found (busca ampla).
    """
    candidate_lines = "\n".join(
        f"{i + 1}. {c['descricao']}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"OCR de nota manuscrita leu o produto como: '{ocr_name}'\n"
        f"Produtos reais no estoque:\n{candidate_lines}\n\n"
        f"Qual é o mais provável que o cliente quis dizer?\n"
        f"- O OCR pode ter confundido letras parecidas (c↔u, n↔h, ei↔ai, r↔v).\n"
        f"- Se o nome OCR contiver peso/tamanho (ex: 300g), prefira o candidato com esse tamanho.\n"
        f"- Se o nome OCR contiver uma marca ou variedade, use como dica.\n"
        f"- Responda APENAS com o número. Se for genuinamente impossível decidir (muito ambíguo), responda 0."
    )
    try:
        resp = await _get_smart_openai().chat.completions.create(
            model=_SMART_RESOLVE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "0").strip().split()[0]
        idx = int(raw) - 1
        if 0 <= idx < len(candidates):
            print(f"[llm_pick] '{ocr_name}' → '{candidates[idx]['descricao']}'")
            return candidates[idx]
    except Exception as e:
        print(f"[llm_pick] ERRO: {e}")
    return None


async def _smart_resolve_product(srk_cli: int, ocr_name: str, action: str) -> dict | None:
    """Quando a busca normal retorna 0 resultados, faz uma busca ampla pelo produto
    base (primeira palavra significativa) e pede ao LLM que escolha o candidato certo.

    Ex: 'tomate uva 300g' → busca 'tomate' → LLM escolhe TOMATE CEREJA BANDEJA 300G.
    """
    norm = " ".join(
        c for c in unicodedata.normalize("NFD", (ocr_name or "").lower())
        if unicodedata.category(c) != "Mn"
    )
    words = [w for w in norm.split() if len(w) >= 3 and not _MEASURE_RE.match(w)]
    if not words:
        return None

    base = words[0]
    candidates = await search_products(
        srk_cli, base, exclude_zero_stock=(action == ACTION_BAIXAR), limit=20
    )
    if not candidates:
        return None

    return await _llm_pick_from_candidates(ocr_name, candidates)


def _norm_desc(s: str) -> str:
    t = (s or "").lower().strip()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return " ".join(t.split())


async def _categorize_batch_item(srk_cli: int, item: dict, *, exclude_zero_stock: bool = True) -> dict:
    """Resolve um item parseado contra o estoque real. Preenche status/candidates/selected."""
    action = item.get("action") or ACTION_BAIXAR
    term   = (item.get("product") or "").strip()
    qty    = item.get("quantity")
    unit   = item.get("unit")

    out = {
        "action": action, "term": term, "quantity": qty, "unit": unit,
        "status": "not_found", "candidates": None, "selected": None,
    }
    if len(term) < 2:
        return out

    options = await search_products(srk_cli, term, exclude_zero_stock=exclude_zero_stock)
    if not options:
        if exclude_zero_stock:
            all_matches = await search_products(srk_cli, term, exclude_zero_stock=False)
            if all_matches:
                out["status"] = "zero_stock"
                return out

        # Busca normal não achou nada — tenta resolver via select amplo + LLM.
        # Ex: OCR leu "tomate uva 300g" → busca "tomate" → LLM escolhe TOMATE CEREJA 300G.
        smart = await _smart_resolve_product(srk_cli, term, action)
        if smart:
            out["selected"] = smart
            out["status"] = "resolved" if qty is not None else "need_qty"
            return out

        out["status"] = "not_found"
        return out

    # Match exato pelo nome → resolve direto
    exact = [o for o in options if _norm_desc(o.get("descricao")) == _norm_desc(term)]
    if len(options) == 1 or len(exact) == 1:
        out["selected"] = exact[0] if len(exact) == 1 else options[0]
        out["status"] = "resolved" if qty is not None else "need_qty"
        return out

    # Múltiplos candidatos — tenta auto-selecionar via LLM antes de perguntar ao usuário.
    # Ex: "tomate cereja 300g" → [CEREJA BANDEJA 300G, CEREJA RAMA 200G] → LLM escolhe o 300G.
    cands = options[:_BATCH_CANDIDATE_LIMIT]
    smart = await _llm_pick_from_candidates(term, cands)
    if smart:
        out["selected"] = smart
        out["status"] = "resolved" if qty is not None else "need_qty"
        return out

    # LLM não conseguiu decidir — mostra opções ao usuário
    out["candidates"] = cands
    out["status"] = "ambiguous"
    return out


async def _categorize_pdf_item(srk_cli: int, codigo: str, descricao: str, qty: float) -> dict:
    """Resolve uma linha de PDF (código + descrição + quantidade) contra o estoque real.

    Tenta primeiro casamento EXATO por código/CEAN (muito mais confiável que busca por
    nome). Só cai na busca fuzzy por descrição se o código não bater — produtos do PDF
    do cliente podem ter código diferente do nosso cadastro, mas a descrição costuma bater.
    Ação é sempre 'baixar': este fluxo é só para relatórios de quebra/saída.
    """
    from solutions.estoque.db_ops import get_product_by_code

    exact = await get_product_by_code(srk_cli, codigo)
    if exact is not None:
        return {
            "action": ACTION_BAIXAR, "term": descricao, "quantity": qty, "unit": None,
            "status": "resolved", "candidates": None, "selected": exact,
        }

    # Código não bateu — cai no mesmo caminho de busca fuzzy por descrição.
    return await _categorize_batch_item(srk_cli, {
        "action": ACTION_BAIXAR, "product": descricao, "quantity": qty, "unit": None,
    })


def _first_pending_item(items: list[dict]) -> tuple[int, dict] | None:
    """Primeiro item que ainda precisa de atenção (desambiguar produto ou pedir qtd)."""
    for i, it in enumerate(items):
        if it.get("status") in ("ambiguous", "need_qty"):
            return i, it
    return None


def _fmt_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, opt in enumerate(candidates, 1):
        desc = (opt.get("descricao") or "")[:38]
        unit = opt.get("unidade") or "UN"
        qtd  = float(opt.get("qnt_estoque") or 0)
        lines.append(f"{i}. {desc} — {qtd:g} {unit}")
    return "\n".join(lines)


def _numbered_resolved(items: list[dict]) -> list[dict]:
    """Itens resolvidos, na mesma ordem de exibição (saída primeiro, depois entrada).
    O número de exibição de cada item é a posição (1-based) nesta lista.
    """
    baixas = [it for it in items if it.get("status") == "resolved" and it["action"] == ACTION_BAIXAR]
    lancas = [it for it in items if it.get("status") == "resolved" and it["action"] == ACTION_LANCAR]
    return baixas + lancas


# Acima disso, a lista completa fica grande demais pro WhatsApp (relatórios de PDF podem
# ter 100+ linhas). Em vez de listar todo item resolvido (a maioria estará certa — casou
# por código/nome exato), mostra só a contagem e detalha o que precisa de atenção
# (não encontrados etc.). 'ver lista' força o detalhe completo se precisar editar.
_SUMMARY_MODE_THRESHOLD = 15
_WARN_NAMES_LIMIT = 30


def _build_batch_confirmation(items: list[dict], *, force_full: bool = False) -> str:
    """Monta o resumo final da movimentação em lote, com itens numerados p/ edição.

    Em lotes grandes (> _SUMMARY_MODE_THRESHOLD), mostra só a contagem dos resolvidos
    (a lista completa fica disponível enviando 'ver lista'), e detalha os que precisam
    de atenção (não encontrados/sem saldo/pulados), que tendem a ser bem menos.
    """
    ordered = _numbered_resolved(items)
    baixas_n = sum(1 for it in items if it.get("status") == "resolved" and it["action"] == ACTION_BAIXAR)
    lancas_n = len(ordered) - baixas_n
    nf      = [it for it in items if it.get("status") == "not_found"]
    zero    = [it for it in items if it.get("status") == "zero_stock"]
    skipped = [it for it in items if it.get("status") == "skip"]

    def _line(num: int, it: dict) -> str:
        sel  = it["selected"]
        qty  = float(it["quantity"] or 0)
        unit = sel.get("unidade") or "UN"
        cur  = float(sel.get("qnt_estoque") or 0)
        sign = "-" if it["action"] == ACTION_BAIXAR else "+"
        delta = -qty if it["action"] == ACTION_BAIXAR else qty
        exp = cur + delta
        if it["action"] == ACTION_BAIXAR and qty > cur:
            return f"{num}. {sel['descricao']} — {sign}{qty:g} {unit} ⚠️ (saldo é só {cur:g})"
        return f"{num}. {sel['descricao']} — {sign}{qty:g} {unit} (saldo {cur:g} → {exp:g})"

    def _section(label: str, group: list[dict], offset: int) -> str:
        return label + "\n" + "\n".join(_line(offset + i + 1, it) for i, it in enumerate(group))

    use_summary = len(ordered) > _SUMMARY_MODE_THRESHOLD and not force_full

    blocks = ["📋 *Confirme a movimentação:*"]
    if use_summary:
        if baixas_n:
            blocks.append(f"\n🔻 *SAÍDA / baixa (-)*: {baixas_n} produto(s) prontos")
        if lancas_n:
            blocks.append(f"\n🔺 *ENTRADA / lançamento (+)*: {lancas_n} produto(s) prontos")
        blocks.append("\n_Envie *'ver lista'* para o detalhe completo item a item._")
    else:
        if baixas_n:
            blocks.append("\n" + _section("🔻 *SAÍDA / baixa (-)*", ordered[:baixas_n], 0))
        if lancas_n:
            blocks.append("\n" + _section("🔺 *ENTRADA / lançamento (+)*", ordered[baixas_n:], baixas_n))

    def _warn_names(group: list[dict]) -> str:
        shown = ", ".join(f'"{it["term"]}"' for it in group[:_WARN_NAMES_LIMIT])
        if len(group) > _WARN_NAMES_LIMIT:
            shown += f" e mais {len(group) - _WARN_NAMES_LIMIT}"
        return shown

    warns = []
    if nf:
        warns.append(f"❓ Não encontrei ({len(nf)}): " + _warn_names(nf))
    if zero:
        warns.append(f"⚠️ Sem saldo para baixar ({len(zero)}): " + _warn_names(zero))
    if skipped:
        warns.append(f"⏭️ Pulados ({len(skipped)}): " + _warn_names(skipped))
    if warns:
        blocks.append("\n" + "\n".join(warns))

    if not ordered:
        return (
            "Não consegui identificar nenhum produto válido para movimentar.\n\n"
            + ("\n".join(warns) + "\n\n" if warns else "")
            + "_Tente enviar de novo, ex: 'saiu 5kg de tomate, 3 alface'._"
        )

    blocks.append(
        "\n_Responda *sim* para confirmar tudo, *não* para cancelar, "
        "ou o *número* do item para corrigir produto/quantidade._"
    )
    return "\n".join(blocks)


def _restore_solution(session_key: str, batch: dict) -> None:
    """Encerra o lote e devolve o usuário à solução anterior.

    Se ele já estava no módulo de estoque, mantém ativo em AWAIT_CONTINUE;
    senão (entrou pelo atalho de fora), volta para a análise (sql_analytics).
    """
    prev = (batch or {}).get("prev_solution")
    update_estoque_state(session_key, batch=None)
    if prev == "estoque":
        update_estoque_state(session_key, step=STEP_AWAIT_CONTINUE)
    else:
        clear_estoque_state(session_key)
        set_active_solution(session_key, DEFAULT_SOLUTION)


def _exit_batch(req: MessageRequest, session_key: str, batch: dict, message: str) -> MessageResponse:
    prev = (batch or {}).get("prev_solution")
    _restore_solution(session_key, batch)
    if prev == "estoque":
        message += "\n\nQuer movimentar mais algum produto? _Envie a lista ou 'sair estoque'._"
    else:
        message += "\n\n_Voltei ao modo de análise. Pode me perguntar o que quiser._"
    return _ok(req, message)


def _advance_batch(session_key: str, batch: dict) -> str:
    """Pede a próxima desambiguação/quantidade ou monta a confirmação final.
    Persiste o estado e retorna o texto a enviar.
    """
    items = batch["items"]
    pend = _first_pending_item(items)
    if pend is not None:
        _, it = pend
        if it["status"] == "ambiguous":
            text = (
                f"Não tenho certeza de qual produto é *{it['term']}*. Qual destes?\n\n"
                f"{_fmt_candidates(it['candidates'])}\n\n"
                f"_Digite o número, ou 0 para pular este item._"
            )
        else:  # need_qty
            sel  = it["selected"]
            unit = sel.get("unidade") or "UN"
            text = (
                f"Quanto de *{sel['descricao']}* você quer "
                f"{_action_word(it['action'])}? (em {unit})\n\n"
                f"_Digite a quantidade, ou 0 para pular._"
            )
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_RESOLVE)
        return text

    # Nenhum item pendente → confirmação (ou encerra se nada resolveu)
    resolved = [it for it in items if it.get("status") == "resolved"]
    text = _build_batch_confirmation(items)
    if not resolved:
        _restore_solution(session_key, batch)
        return text
    update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
    return text


async def _handle_batch_resolve(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip()
    state = get_estoque_state(session_key)
    batch = state.get("batch") or {}
    items = batch.get("items") or []

    if not items:
        return _exit_batch(req, session_key, batch, "A sessão expirou. Envie a lista novamente, por favor.")

    if _is_back_command(msg) or msg.lower() in ("cancelar", "cancela"):
        return _exit_batch(req, session_key, batch, "Movimentação cancelada. Nada foi alterado. 👍")

    pend = _first_pending_item(items)
    if pend is None:
        return _ok(req, _advance_batch(session_key, batch))
    _, it = pend

    if msg.lower() in ("0", "pular", "skip"):
        it["status"] = "skip"
        return _ok(req, _advance_batch(session_key, batch))

    if it["status"] == "ambiguous":
        cands = it.get("candidates") or []
        try:
            sel_idx = int(msg) - 1
        except ValueError:
            return _ok(req, "Digite o número de uma das opções, ou 0 para pular este item.")
        if not (0 <= sel_idx < len(cands)):
            return _ok(req, f"Opção inválida. Digite um número entre 1 e {len(cands)}, ou 0 para pular.")
        it["selected"] = cands[sel_idx]
        it["status"] = "resolved" if it.get("quantity") is not None else "need_qty"
        return _ok(req, _advance_batch(session_key, batch))

    # need_qty
    sel  = it["selected"] or {}
    unit = (sel.get("unidade") or "UN").upper()
    qty = await parse_quantity(msg, unit)
    if qty is None or qty <= 0:
        return _ok(req, f"Não entendi a quantidade '{msg}'. Tente de novo (ex: 5, 2.5), ou 0 para pular.")
    if unit == "UN" and qty != int(qty):
        return _ok(req, "Para produtos vendidos em UN a quantidade precisa ser inteira. Tente de novo.")
    it["quantity"] = qty
    it["status"] = "resolved"
    return _ok(req, _advance_batch(session_key, batch))


# Sets próprios do lote: SEM "1"/"2", já que esses números agora selecionam
# itens para edição (diferente do _CONFIRM_YES/_CONFIRM_NO do fluxo de item único).
_BATCH_CONFIRM_YES = {"sim", "s", "ok", "confirma", "confirmar"}
_BATCH_CONFIRM_NO  = {"nao", "não", "n", "cancela", "cancelar"}


def _edit_menu_text(it: dict) -> str:
    sel  = it["selected"]
    sign = "-" if it["action"] == ACTION_BAIXAR else "+"
    qty  = float(it["quantity"] or 0)
    unit = sel.get("unidade") or "UN"
    return (
        f"Editando: *{sel['descricao']}* ({sign}{qty:g} {unit})\n\n"
        f"1 - Mudar produto\n2 - Mudar quantidade\n3 - Remover da lista\n\n"
        f"_Ou 'voltar' para cancelar a edição._"
    )


async def _handle_batch_confirm(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip().lower()
    state = get_estoque_state(session_key)
    batch = state.get("batch") or {}
    items = batch.get("items") or []
    srk_cli = batch.get("srk_cli")

    if not items or srk_cli is None:
        return _exit_batch(req, session_key, batch, "A sessão expirou. Envie a lista novamente, por favor.")

    if msg in _BATCH_CONFIRM_NO or _is_back_command(msg) or msg in ("cancelar", "cancela"):
        return _exit_batch(req, session_key, batch, "Movimentação cancelada. Nada foi alterado. 👍")

    if msg in ("ver lista", "ver tudo", "lista completa", "ver"):
        return _ok(req, _build_batch_confirmation(items, force_full=True))

    if msg in _BATCH_CONFIRM_YES:
        pass  # segue para a aplicação abaixo
    else:
        # Tenta interpretar como número de item para editar
        ordered = _numbered_resolved(items)
        try:
            num = int(msg)
        except ValueError:
            return _ok(req, "Responda *sim* para confirmar tudo, *não* para cancelar, ou o número do item para editar.")
        if not (1 <= num <= len(ordered)):
            return _ok(req, f"Número inválido. Escolha entre 1 e {len(ordered)}, ou *sim*/*não*.")

        target = ordered[num - 1]
        idx = next(i for i, it in enumerate(items) if it is target)
        batch["editing_index"] = idx
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_EDIT_MENU)
        return _ok(req, _edit_menu_text(target))

    resolved_count = sum(1 for it in items if it.get("status") == "resolved")
    # Para lotes grandes (ex: PDF com muitos itens), avisa que vai demorar um pouco —
    # sem isso o cliente fica sem feedback durante os updates sequenciais no banco.
    if resolved_count > 5 and not session_key.startswith("portal_") and req.phone:
        try:
            from shared.infobip_client import send_whatsapp_text
            await send_whatsapp_text(
                to=req.phone,
                text=f"⚙️ Aplicando {resolved_count} movimentação(ões), aguarde...",
            )
        except Exception as e:
            print(f"[estoque] AVISO: falha ao enviar status de progresso: {e}")

    # Aplica em blocos paralelos com heartbeat — sequencial seria muito lento para
    # lotes grandes (PDF). Usa um lock por CÓDIGO de produto: se o mesmo produto aparecer
    # em várias linhas (comum em relatórios de quebra), serializa só essas, evitando
    # perder atualização por leitura/escrita concorrente no mesmo saldo.
    locks: dict[str, asyncio.Lock] = {}

    def _lock_for(codigo: str) -> asyncio.Lock:
        if codigo not in locks:
            locks[codigo] = asyncio.Lock()
        return locks[codigo]

    async def _apply_one(it: dict) -> tuple[str, str]:
        sel  = it["selected"]
        qty  = float(it["quantity"] or 0)
        unit = sel.get("unidade") or "UN"
        codigo = sel["codigo"]

        async with _lock_for(codigo):
            fresh = await get_current_stock(int(srk_cli), codigo)
            if fresh is None:
                return "failed", f"• {sel['descricao']} — produto não encontrado"
            cur = float(fresh.get("qnt_estoque") or 0)
            if it["action"] == ACTION_BAIXAR and qty > cur:
                return "failed", f"• {sel['descricao']} — saldo insuficiente ({cur:g} {unit})"

            delta = -qty if it["action"] == ACTION_BAIXAR else qty
            tipo_op = batch.get("tipo_operacao") or (None if it["action"] != ACTION_BAIXAR else "BAIXA")
            try:
                await update_stock(int(srk_cli), codigo, delta, tipo_operacao=tipo_op)
                verb = "baixados" if it["action"] == ACTION_BAIXAR else "lançados"
                sign = "-" if it["action"] == ACTION_BAIXAR else "+"
                return "applied", f"• {sel['descricao']} — {sign}{qty:g} {unit} {verb} (saldo {cur + delta:g})"
            except Exception as e:
                print(f"[estoque] ERRO update_stock (batch): {e}")
                return "failed", f"• {sel['descricao']} — erro ao gravar"

    resolved_items = [it for it in items if it.get("status") == "resolved"]
    results = await _run_with_heartbeat(
        req, session_key, resolved_items, _apply_one, concurrency=_PDF_CONCURRENCY, label="aplicando as movimentações"
    )

    applied: list[str] = [msg for kind, msg in results if kind == "applied"]
    failed:  list[str] = [msg for kind, msg in results if kind == "failed"]

    parts = []
    if applied:
        if len(applied) > _SUMMARY_MODE_THRESHOLD:
            parts.append(f"✅ *Movimentações finalizadas!* {len(applied)} produto(s) atualizados com sucesso.")
        else:
            parts.append("✅ *Movimentações finalizadas!*\n" + "\n".join(applied))
    if failed:
        parts.append(f"⚠️ *Não aplicados ({len(failed)}):*\n" + "\n".join(failed[:_WARN_NAMES_LIMIT]))
        if len(failed) > _WARN_NAMES_LIMIT:
            parts[-1] += f"\n_... e mais {len(failed) - _WARN_NAMES_LIMIT}._"
    if not parts:
        parts.append("Nada foi movimentado.")
    return _exit_batch(req, session_key, batch, "\n\n".join(parts))


async def _handle_batch_edit_menu(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip().lower()
    state = get_estoque_state(session_key)
    batch = state.get("batch") or {}
    items = batch.get("items") or []
    idx = batch.get("editing_index")

    if idx is None or not (0 <= idx < len(items)):
        return _ok(req, _advance_batch(session_key, batch))
    it = items[idx]

    if _is_back_command(msg):
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    if msg == "1":
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_EDIT_PRODUCT)
        return _ok(req, f"Digite o novo nome do produto para substituir *{it['selected']['descricao']}*:")

    if msg == "2":
        unit = it["selected"].get("unidade") or "UN"
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_EDIT_QUANTITY)
        return _ok(req,
            f"Quanto de *{it['selected']['descricao']}* você quer "
            f"{_action_word(it['action'])}? (em {unit})"
        )

    if msg == "3":
        it["status"] = "skip"
        batch.pop("editing_index", None)
        return _ok(req, _advance_batch(session_key, batch))

    return _ok(req, "Digite 1 (mudar produto), 2 (mudar quantidade), 3 (remover), ou 'voltar'.")


async def _handle_batch_edit_product(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip()
    state = get_estoque_state(session_key)
    batch = state.get("batch") or {}
    items = batch.get("items") or []
    idx = batch.get("editing_index")
    srk_cli = batch.get("srk_cli")

    if idx is None or not (0 <= idx < len(items)) or srk_cli is None:
        return _ok(req, _advance_batch(session_key, batch))
    it = items[idx]

    if _is_back_command(msg):
        batch.pop("edit_candidates", None)
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    cands = batch.get("edit_candidates")
    if cands:
        try:
            sel_idx = int(msg) - 1
        except ValueError:
            return _ok(req, "Digite o número de uma das opções, ou 'voltar' para cancelar.")
        if not (0 <= sel_idx < len(cands)):
            return _ok(req, f"Opção inválida. Digite um número entre 1 e {len(cands)}.")
        it["selected"] = cands[sel_idx]
        batch.pop("edit_candidates", None)
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    term = msg.strip()
    if len(term) < 2:
        return _ok(req, "Digite ao menos 2 caracteres do nome do produto.")

    options = await search_products(srk_cli, term, exclude_zero_stock=(it["action"] == ACTION_BAIXAR))
    if not options:
        smart = await _smart_resolve_product(srk_cli, term, it["action"])
        if smart:
            it["selected"] = smart
            batch.pop("editing_index", None)
            update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
            return _ok(req, _build_batch_confirmation(items))
        return _ok(req, f"Não encontrei nenhum produto com '{term}'. Tente outro termo, ou 'voltar' para cancelar.")

    exact = [o for o in options if _norm_desc(o.get("descricao")) == _norm_desc(term)]
    if len(options) == 1 or len(exact) == 1:
        it["selected"] = exact[0] if len(exact) == 1 else options[0]
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    smart = await _llm_pick_from_candidates(term, options[:_BATCH_CANDIDATE_LIMIT])
    if smart:
        it["selected"] = smart
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    cands = options[:_BATCH_CANDIDATE_LIMIT]
    batch["edit_candidates"] = cands
    update_estoque_state(session_key, batch=batch, step=STEP_BATCH_EDIT_PRODUCT)
    return _ok(req,
        f"Não tenho certeza. Qual destes?\n\n{_fmt_candidates(cands)}\n\n"
        f"_Digite o número, ou 'voltar' para cancelar._"
    )


async def _handle_batch_edit_quantity(req: MessageRequest, session_key: str) -> MessageResponse:
    msg = (req.message or "").strip()
    state = get_estoque_state(session_key)
    batch = state.get("batch") or {}
    items = batch.get("items") or []
    idx = batch.get("editing_index")

    if idx is None or not (0 <= idx < len(items)):
        return _ok(req, _advance_batch(session_key, batch))
    it = items[idx]

    if _is_back_command(msg):
        batch.pop("editing_index", None)
        update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
        return _ok(req, _build_batch_confirmation(items))

    sel  = it["selected"] or {}
    unit = (sel.get("unidade") or "UN").upper()
    qty = await parse_quantity(msg, unit)
    if qty is None or qty <= 0:
        return _ok(req, f"Não entendi a quantidade '{msg}'. Tente de novo (ex: 5, 2.5), ou 'voltar' para cancelar.")
    if unit == "UN" and qty != int(qty):
        return _ok(req, "Para produtos vendidos em UN a quantidade precisa ser inteira. Tente de novo.")

    it["quantity"] = qty
    batch.pop("editing_index", None)
    update_estoque_state(session_key, batch=batch, step=STEP_BATCH_CONFIRM)
    return _ok(req, _build_batch_confirmation(items))


async def _resolve_cnpj_srk(req: MessageRequest, session_key: str, source: str) -> int | None:
    """Resolve CNPJ (request → sessão → telefone), confirma que o cliente tem estoque
    e retorna o SRK_CLI, ou None se não foi possível resolver."""
    cnpj = (req.cnpj or "").strip()
    if not cnpj:
        cnpj = ((get_session(session_key) or {}).get("cnpj") or "").strip()
    if not cnpj and source == "whatsapp" and req.phone:
        from shared.db_client import get_cnpj_for_phone
        cnpj = (await get_cnpj_for_phone(req.phone) or "").strip()
    if not cnpj:
        return None

    srk_cli = await get_srk_cli(cnpj)
    return int(srk_cli) if srk_cli is not None else None


async def try_fast_path(
    req: MessageRequest, source: str, *, force: bool = False
) -> MessageResponse | None:
    """Tenta interpretar a mensagem como uma movimentação rápida/em lote.

    Retorna uma MessageResponse se reconheceu e tratou; retorna None se não parece
    movimentação (o dispatcher segue o fluxo normal). `force=True` para fotos/listas.
    """
    from solutions.estoque.batch_parser import looks_like_movement

    text = (req.message or "").strip()
    if not text and not force:
        return None
    if not force and not looks_like_movement(text):
        return None

    session_key = _session_key_for(req.phone or "", req.cnpj, source)
    srk_cli = await _resolve_cnpj_srk(req, session_key, source)
    if srk_cli is None:
        return None

    parsed = await parse_movement(text, force=force)
    if not parsed.is_movement or not parsed.items:
        return None

    items: list[dict] = []
    for pit in parsed.items:
        cat = await _categorize_batch_item(srk_cli, {
            "action":   pit.action,
            "product":  pit.product,
            "quantity": pit.quantity,
            "unit":     pit.unit,
        })
        items.append(cat)

    prev = get_active_solution(session_key)
    batch = {"items": items, "srk_cli": srk_cli, "prev_solution": prev}

    # Entra em modo estoque para sustentar o diálogo multi-turno da confirmação
    set_active_solution(session_key, "estoque")
    update_estoque_state(session_key, srk_cli=srk_cli)

    return _ok(req, _advance_batch(session_key, batch))


# Limite de consultas simultâneas ao Databricks durante a categorização do PDF —
# alto suficiente para acelerar bastante, baixo suficiente para não sobrecarregar o warehouse.
_PDF_CONCURRENCY = 10
_PROGRESS_INTERVAL_SEC = 25  # intervalo mínimo entre mensagens de "ainda trabalhando"


async def _maybe_send_progress(req: MessageRequest, session_key: str, text: str) -> None:
    """Envia uma mensagem de status no WhatsApp, se aplicável (não bloqueia em erro)."""
    if session_key.startswith("portal_") or not req.phone:
        return
    try:
        from shared.infobip_client import send_whatsapp_text
        await send_whatsapp_text(to=req.phone, text=text)
    except Exception as e:
        print(f"[estoque] AVISO: falha ao enviar progresso: {e}")


async def _run_with_heartbeat(
    req: MessageRequest, session_key: str,
    work_items: list, worker, *, concurrency: int, label: str,
) -> list:
    """Processa `work_items` em blocos paralelos (até `concurrency` por vez), mandando
    uma mensagem de status periódica (a cada ~_PROGRESS_INTERVAL_SEC) enquanto durar,
    para o usuário não pensar que travou em lotes grandes (ex: PDF com 100+ linhas).
    """
    import time as _time

    results: list = []
    total = len(work_items)
    last_update = _time.monotonic()

    for start in range(0, total, concurrency):
        chunk = work_items[start:start + concurrency]
        chunk_results = await asyncio.gather(*(worker(it) for it in chunk))
        results.extend(chunk_results)

        done = start + len(chunk)
        now = _time.monotonic()
        if done < total and now - last_update >= _PROGRESS_INTERVAL_SEC:
            await _maybe_send_progress(
                req, session_key, f"⚙️ Ainda {label}... {done}/{total} concluídos."
            )
            last_update = now

    return results


async def try_fast_path_pdf(
    req: MessageRequest, source: str, rows: list[dict]
) -> MessageResponse | None:
    """Processa linhas já extraídas de um PDF de movimentação (código + descrição + qtd).

    Cada linha tenta casar por código exato antes de cair na busca fuzzy por descrição.
    Ação é sempre 'baixar' (relatórios deste tipo são sempre de saída/quebra).
    """
    if not rows:
        return None

    session_key = _session_key_for(req.phone or "", req.cnpj, source)
    srk_cli = await _resolve_cnpj_srk(req, session_key, source)
    if srk_cli is None:
        return None

    # Processa em blocos paralelos com heartbeat — sequencial seria inviável para
    # relatórios de PDF com 100+ linhas: cada item bate no Databricks (que faz polling
    # a cada 0.5s por consulta), então 200 itens sequenciais levariam minutos.
    async def _worker(row: dict) -> dict:
        return await _categorize_pdf_item(
            srk_cli, row.get("codigo") or "", row.get("descricao") or "", float(row.get("quantidade") or 0)
        )

    items = await _run_with_heartbeat(
        req, session_key, rows, _worker, concurrency=_PDF_CONCURRENCY, label="cruzando com seu estoque"
    )

    prev = get_active_solution(session_key)
    batch = {"items": items, "srk_cli": srk_cli, "prev_solution": prev}

    set_active_solution(session_key, "estoque")
    update_estoque_state(session_key, srk_cli=srk_cli)

    return _ok(req, _advance_batch(session_key, batch))


# ── Roteamento interno ────────────────────────────────────────────────────────

_STEP_HANDLERS = {
    STEP_MENU:                _handle_menu,
    STEP_AWAIT_PRODUCT_QTY:  _handle_await_product_qty,
    STEP_AWAIT_PRODUCT:       _handle_await_product,    # legado / batch
    STEP_AWAIT_CHOICE:        _handle_await_choice,
    STEP_AWAIT_QUANTITY:      _handle_await_quantity,   # legado / batch
    STEP_AWAIT_CONFIRMATION:  _handle_await_confirmation,
    STEP_AWAIT_CONTINUE:      _handle_await_continue,
    STEP_BATCH_RESOLVE:       _handle_batch_resolve,
    STEP_BATCH_CONFIRM:       _handle_batch_confirm,
    STEP_BATCH_EDIT_MENU:     _handle_batch_edit_menu,
    STEP_BATCH_EDIT_PRODUCT:  _handle_batch_edit_product,
    STEP_BATCH_EDIT_QUANTITY: _handle_batch_edit_quantity,
}


async def handle_message(req: MessageRequest, source: Literal["whatsapp", "portal"] = "whatsapp") -> MessageResponse:
    """Processa uma mensagem dentro do módulo de estoque."""
    session_key = _session_key_for(req.phone or "", req.cnpj, source)
    state = get_estoque_state(session_key)
    step  = state.get("step")

    # Sem estado: mostra o menu (caso o handler seja chamado sem ativação prévia).
    if not step:
        text = init_estoque_session(session_key)
        return _ok(req, text)

    handler = _STEP_HANDLERS.get(step)
    if not handler:
        text = init_estoque_session(session_key)
        return _ok(req, text)

    return await handler(req, session_key)


# ── API direta (testes) ───────────────────────────────────────────────────────

router = APIRouter(prefix="/estoque", tags=["Estoque"])


@router.post("/message", response_model=MessageResponse)
async def api_message(req: MessageRequest):
    return await handle_message(req)
