from __future__ import annotations

import asyncio
import os
import re
import traceback
from contextlib import asynccontextmanager

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


import secrets
import shared.conversation_logger as conv_log
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from shared.audio_transcriber import transcribe_message
from shared.db_client import get_cnpjs_for_phone
from shared.email_client import send_insight_feedback_email, send_catalog_suggestion_email, send_image_received_email
from shared.infobip_client import send_catalog_suggestion_buttons, send_whatsapp_text, send_whatsapp_menu, send_whatsapp_document
from shared.session_store import (
    set_active_solution, get_active_solution, clear_estoque_state,
    get_session, set_pending_options, set_cnpj,
    set_pending_insight, get_pending_insight, clear_pending_insight,
    set_pending_catalog_suggestion, get_pending_catalog_suggestion, clear_pending_catalog_suggestion,
)
from shared.solution_router import DEFAULT_SOLUTION, match_activation, is_exit_command

# ── Sessões admin (in-memory: token → usuario) ────────────────────────────────
_admin_sessions: dict[str, str] = {}


def _get_admin_user(request: Request) -> str | None:
    token = request.cookies.get("admin_token")
    return _admin_sessions.get(token) if token else None

# ── Routers de cada solução ───────────────────────────────────────────────────
from solutions.sql_analytics.router import init as sql_analytics_init
from solutions.sql_analytics.router import router as sql_analytics_router
from solutions.sql_analytics.router import handle_message as sql_analytics_handle, MessageRequest, MessageResponse

from solutions.sql_bh.router import init as sql_bh_init
from solutions.sql_bh.router import router as sql_bh_router
from solutions.sql_bh.router import handle_message as bh_handle_message, BhMessageRequest

from solutions.estoque.router import init as estoque_init
from solutions.estoque.router import router as estoque_router
from solutions.estoque.router import (
    handle_message as estoque_handle,
    init_estoque_session,
    try_fast_path as estoque_try_fast_path,
)
from shared.image_reader import read_image_message

# Reconhecimento de foto desativado por hora (priorizando PDF) — religar via .env.
IMAGE_RECOGNITION_ENABLED = (os.getenv("IMAGE_RECOGNITION_ENABLED") or "0").strip() == "1"

# ── Dispatch de soluções (chave = nome registrado em solution_router.py) ──────
_SOLUTION_HANDLERS = {
    "sql_analytics": sql_analytics_handle,
    "estoque":       estoque_handle,
}

# Cache de user_name por (phone, cnpj) — evita bater no Databricks toda mensagem
_user_name_cache: dict[str, tuple[str, float]] = {}
_fantasia_cache:  dict[str, tuple[str, float]] = {}
_USER_NAME_CACHE_TTL = 3600  # 1 hora

async def _get_user_name_cached(phone: str, cnpj: str) -> str:
    import time as _time
    key = f"{phone}_{cnpj}"
    cached = _user_name_cache.get(key)
    if cached and _time.time() < cached[1]:
        return cached[0]
    try:
        from shared.db_client import get_user_name_for_phone
        name = await get_user_name_for_phone(phone, cnpj) or ""
    except Exception:
        name = ""
    _user_name_cache[key] = (name, _time.time() + _USER_NAME_CACHE_TTL)
    return name


async def _get_fantasia_cached(cnpj: str) -> str:
    """Retorna nome fantasia (razão social) para um CNPJ, com cache de 1h."""
    import time as _time
    if not cnpj:
        return ""
    cached = _fantasia_cache.get(cnpj)
    if cached and _time.time() < cached[1]:
        return cached[0]
    try:
        from shared.db_client import get_nome_fantasia_for_cnpj
        name = await get_nome_fantasia_for_cnpj(cnpj) or ""
    except Exception:
        name = ""
    _fantasia_cache[cnpj] = (name, _time.time() + _USER_NAME_CACHE_TTL)
    return name


async def _resolve_display_name(user_name: str, cnpj: str) -> str:
    """Retorna user_name se preenchido; senão tenta nome fantasia; senão '—'."""
    if user_name and user_name.strip():
        return user_name.strip()
    fantasia = await _get_fantasia_cached(cnpj or "")
    return fantasia if fantasia else "—"

# ── Lifespan (inicializa todas as soluções) ───────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    import pathlib
    from datetime import date

    db_path = os.getenv("SQLITE_PATH", "checkpoints.db")
    db_file = pathlib.Path(db_path)

    if db_file.exists():
        file_date = date.fromtimestamp(db_file.stat().st_mtime)
        if file_date < date.today():
            db_file.unlink()
            print(f"[lifespan] checkpoints.db do dia {file_date} removido ✓")

    async with aiosqlite.connect(db_path) as conn:
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        print("[lifespan] SQLite checkpointer pronto ✓")

        conv_db = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
        conv_log.setup(conv_db)
        conv_log.setup_users(conv_db)
        await sql_analytics_init(checkpointer)
        sql_bh_init()
        estoque_init()
        # Scheduler diário movido para scripts/daily_report.py (Windows Task Scheduler)

        yield

    print("[lifespan] SQLite encerrado")


app = FastAPI(title="iMAIS Chat API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _strip_markdown(text: str) -> str:
    """Remove formatação Markdown (negrito, itálico) para exibição no portal."""
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)  # *bold* e **bold**
    text = re.sub(r"_+([^_]+)_+", r"\1", text)     # _italic_
    return text


async def _dispatch(req: MessageRequest, source: str) -> MessageResponse:
    """Roteia a mensagem para a solução correta e invoca seu handler."""
    phone = req.phone or ""

    # Usa phone para WhatsApp; para portal usa cnpj como chave de sessão
    session_key = phone if source == "whatsapp" else f"portal_{req.cnpj or ''}"

    # 1) Comando de saída → verificar ANTES de activation para "sair estoque"
    #    não casar com "estoque" da lista de ativação.
    if is_exit_command(req.message):
        active = get_active_solution(session_key)
        if active == "estoque":
            clear_estoque_state(session_key)
        set_active_solution(session_key, DEFAULT_SOLUTION)
        print(f"[dispatch] {session_key} → saiu de '{active}' → '{DEFAULT_SOLUTION}'")
        return MessageResponse(
            phone=phone,
            answer="✅ Você saiu do módulo de estoque. O que gostaria de analisar?",
            cnpj=req.cnpj,
        )

    # 2) Comando de ativação de módulo?
    solution = match_activation(req.message)
    if solution:
        if solution == "estoque":
            # Resolve CNPJ: portal passa direto, WhatsApp busca na sessão ou pelo telefone
            cnpj_for_check = req.cnpj or ""
            if not cnpj_for_check and source == "whatsapp":
                cnpj_for_check = (get_session(session_key) or {}).get("cnpj") or ""

            # Sem CNPJ na sessão WhatsApp: resolve pelo telefone
            if not cnpj_for_check and source == "whatsapp" and phone:
                cnpjs = await get_cnpjs_for_phone(phone)
                if len(cnpjs) == 0:
                    return MessageResponse(
                        phone=phone,
                        answer="Olá! Esse número está vinculado à *Inteligência Artificial do iMais*. Caso tenha alguma dúvida sobre nossos serviços, entre em contato com o suporte:\n- WhatsApp: (34) 99912-7261\n- Capitais e regiões metropolitanas: 3003-1266\n- Outras regiões: 0800-729-5217",
                        cnpj=None,
                    )
                if len(cnpjs) == 1:
                    cnpj_for_check = cnpjs[0]["cnpj"]
                    set_cnpj(phone, cnpj_for_check)
                else:
                    # Múltiplos CNPJs: pede seleção antes de ativar estoque
                    set_pending_options(phone, cnpjs, pending_activation="estoque")
                    lines = ["Para acessar o módulo de estoque, selecione o estabelecimento:", ""]
                    for i, opt in enumerate(cnpjs, 1):
                        name = opt.get("name") or opt["cnpj"]
                        lines.append(f"{i}. {name}")
                    lines += ["", "Responda com o número da opção (ex: 1)."]
                    print(f"[dispatch] {session_key} → seleção de CNPJ pendente para estoque")
                    return MessageResponse(phone=phone, answer="\n".join(lines), cnpj=None, menu_options=cnpjs)

            set_active_solution(session_key, solution)
            print(f"[dispatch] {session_key} → ativou '{solution}'")
            text = init_estoque_session(session_key)
            return MessageResponse(phone=phone, answer=text, cnpj=req.cnpj)

        set_active_solution(session_key, solution)
        print(f"[dispatch] {session_key} → ativou '{solution}'")
        return MessageResponse(phone=phone, answer="Modo ativado.", cnpj=req.cnpj)

    # 3) Roteia para a solução ativa na sessão
    active = get_active_solution(session_key)
    handler = _SOLUTION_HANDLERS.get(active) or sql_analytics_handle
    print(f"[dispatch] {session_key} → '{active}'")
    return await handler(req, source=source)


# ── Inclui routers ────────────────────────────────────────────────────────────
app.include_router(sql_analytics_router)
app.include_router(sql_bh_router)
app.include_router(estoque_router)


# ── Portal (chamada direta do frontend) ───────────────────────────────────────

@app.post("/portal/message")
async def portal_message(request: Request):
    """Recebe mensagens do portal web. CNPJ vem da sessão do usuário."""
    body = await request.json()

    phone = (body.get("phone") or "").strip()
    message = (body.get("message") or "").strip()
    cnpj = (body.get("cnpj") or "").strip()

    if not message:
        return JSONResponse(content={"error": "message é obrigatório"}, status_code=400)
    if not cnpj:
        return JSONResponse(content={"error": "cnpj é obrigatório para o portal"}, status_code=400)

    print(f"[PORTAL] Mensagem recebida | cnpj={cnpj}: {message[:80]}")

    try:
        req = MessageRequest(phone=phone, message=message, cnpj=cnpj)
        resp = await _dispatch(req, source="portal")
        _insight_p = getattr(resp, "insight", None) or ""
        _thread_p  = f"portal_{cnpj}"
        _uname_p = await _get_user_name_cached(phone or "", cnpj)
        conv_log.log(
            thread_id=_thread_p, phone=phone, cnpj=cnpj, canal="portal",
            user_name=_uname_p,
            question=message, answer=resp.answer,
            metric=getattr(resp, "metric", "") or "",
            insight=_insight_p,
        )
        if _insight_p:
            set_pending_insight(_thread_p, {
                "insight":  _insight_p,
                "question": message,
                "answer":   resp.answer,
            })
        return JSONResponse(content={
            "phone":   resp.phone,
            "answer":  _strip_markdown(resp.answer),
            "insight": _strip_markdown(_insight_p) if _insight_p else None,
            "cnpj": resp.cnpj,
        })
    except Exception as e:
        print(f"[PORTAL] ERRO: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"error": "Erro ao processar mensagem. Tente novamente."},
            status_code=500,
        )


async def _send_insight_async(phone: str, insight: str, question: str, answer: str, thread_id: str = "") -> None:
    """Envia insight com botões SIM/NÃO em background após a resposta principal."""
    from shared.infobip_client import send_insight_buttons
    try:
        set_pending_insight(phone, {"insight": insight, "question": question, "answer": answer, "thread_id": thread_id})
        await send_insight_buttons(phone, insight)
        print(f"[INSIGHT] Enviado para {phone}: {insight[:60]}")
    except Exception as e:
        print(f"[INSIGHT] ERRO ao enviar: {e}")


# ── Webhook Infobip (WhatsApp inbound) ────────────────────────────────────────

# Dedup de mensagens inbound — o Infobip pode reenviar o webhook (retry) com um
# messageId DIFERENTE a cada tentativa, mas a mídia (foto/PDF) referenciada é a mesma.
# Por isso checamos várias chaves candidatas (messageId, media id, media url) — basta
# UMA bater com algo já visto pra considerar duplicata. TTL generoso (retries podem
# vir minutos depois).
_processed_message_keys: dict[str, float] = {}
_MESSAGE_DEDUP_TTL = 1800  # 30 minutos


def _already_processed(keys: list[str]) -> bool:
    import time as _time
    keys = [k for k in keys if k]
    if not keys:
        return False
    now = _time.time()
    # Limpeza oportunista de entradas expiradas
    expired = [k for k, t in _processed_message_keys.items() if now - t > _MESSAGE_DEDUP_TTL]
    for k in expired:
        _processed_message_keys.pop(k, None)
    if any(k in _processed_message_keys for k in keys):
        return True
    for k in keys:
        _processed_message_keys[k] = now
    return False


def _split_whatsapp_text(text: str, limit: int = 3500) -> list[str]:
    """Quebra um texto longo em pedaços abaixo do limite do WhatsApp (~4096 chars),
    respeitando as fronteiras entre blocos (parágrafos separados por linha em branco).
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    atual = ""
    for bloco in text.split("\n\n"):
        candidato = bloco if not atual else f"{atual}\n\n{bloco}"
        if len(candidato) <= limit:
            atual = candidato
            continue
        if atual:
            chunks.append(atual)
        # bloco isolado ainda pode passar do limite → fatia bruta
        while len(bloco) > limit:
            chunks.append(bloco[:limit])
            bloco = bloco[limit:]
        atual = bloco
    if atual:
        chunks.append(atual)
    return chunks


async def _process_whatsapp_message(msg: dict) -> None:
    """Processa uma mensagem inbound do Infobip. Roda em background (asyncio.create_task)
    para não bloquear a resposta HTTP do webhook — processamentos longos (PDF/foto com
    muitos itens) faziam o Infobip considerar timeout e reenviar a mesma mensagem,
    causando respostas duplicadas.
    """
    sender = (msg.get("from") or "").strip()
    message_obj = msg.get("message") or {}
    # Infobip pode enviar o type dentro de message ou na raiz do evento
    msg_type = (message_obj.get("type") or msg.get("type") or "").upper()

    print(f"[WHATSAPP] Evento recebido de {sender} | type={msg_type}")

    if msg_type in ("INTERACTIVE_BUTTON_REPLY", "INTERACTIVE_LIST_REPLY"):
        button_id = str(message_obj.get("id") or "").strip()

        # Sugestão de catálogo (catalog_yes / catalog_no)
        if button_id in ("catalog_yes", "catalog_no") and sender:
            pending = get_pending_catalog_suggestion(sender)
            feedback = "Sim ✓" if button_id == "catalog_yes" else "Não ✗"
            # Registra o feedback na base de aprendizado (#2)
            _cnpj_cat = (pending or {}).get("cnpj") or ""
            _tid_cat  = f"{sender}_{_cnpj_cat}" if _cnpj_cat else sender
            _db_cat   = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
            conv_log.update_catalog_feedback(_tid_cat, feedback, _db_cat)
            if button_id == "catalog_yes" and pending:
                asyncio.create_task(send_catalog_suggestion_email(
                    phone=sender,
                    question=pending.get("question", ""),
                    cnpj=pending.get("cnpj", ""),
                ))
                await send_whatsapp_text(to=sender, text="Obrigado! Vamos analisar e adicionar ao catálogo. 📋")
            else:
                await send_whatsapp_text(to=sender, text="Tudo bem! Se tiver outras perguntas, estou aqui. 😊")
            clear_pending_catalog_suggestion(sender)
            return

        # Feedback de insight (SIM/NÃO)
        if button_id in ("insight_sim", "insight_nao") and sender:
            pending  = get_pending_insight(sender)
            feedback = "Sim ✓" if button_id == "insight_sim" else "Não ✗"
            print(f"[INSIGHT] Feedback recebido de {sender}: {feedback}")
            if pending:
                asyncio.create_task(send_insight_feedback_email(
                    phone=sender,
                    question=pending.get("question", ""),
                    answer=pending.get("answer", ""),
                    insight=pending.get("insight", ""),
                    feedback=feedback,
                ))
                # Salva feedback no SQLite
                _tid = pending.get("thread_id") or sender
                _db  = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
                conv_log.update_insight_feedback(_tid, feedback, _db)
                clear_pending_insight(sender)
            await send_whatsapp_text(to=sender, text="Obrigado pelo feedback! 🙏")
            return

        text = button_id
    elif msg_type in ("AUDIO", "VOICE"):
        print(f"[WHATSAPP] Áudio detectado de {sender} — transcrevendo...")
        try:
            text = await transcribe_message(msg) or ""
            if text:
                print(f"[WHATSAPP] Transcrição OK: {text[:80]}")
            else:
                print(f"[WHATSAPP] Transcrição vazia de {sender}")
                await send_whatsapp_text(
                    to=sender,
                    text="Não consegui entender o áudio. Pode digitar sua mensagem?",
                )
        except Exception as e:
            print(f"[WHATSAPP] ERRO ao transcrever áudio de {sender}: {e}")
            traceback.print_exc()
            await send_whatsapp_text(
                to=sender,
                text="Não consegui processar o áudio. Pode digitar sua mensagem?",
            )
            text = ""
    elif msg_type in ("IMAGE", "PICTURE", "PHOTO"):
        if not sender:
            return

        if not IMAGE_RECOGNITION_ENABLED:
            print(f"[WHATSAPP] Imagem detectada de {sender} — reconhecimento desativado, avisando por email.")
            try:
                cnpj_img = (get_session(sender) or {}).get("cnpj") or ""
                asyncio.create_task(send_image_received_email(sender, cnpj_img))
            except Exception as e:
                print(f"[WHATSAPP] AVISO: falha ao notificar email de imagem recebida: {e}")
            await send_whatsapp_text(
                to=sender,
                text="📷 Recebemos sua foto! Ainda estamos desenvolvendo o reconhecimento "
                     "automático de imagens. Por enquanto, envie por *texto*, *áudio* ou *PDF*.",
            )
            return

        # ── Reconhecimento de imagem (OCR) — desativado por padrão, ver flag acima ──
        print(f"[WHATSAPP] Imagem detectada de {sender} — lendo lista (OCR)...")
        await send_whatsapp_text(to=sender, text="📷 Lendo a foto...")
        try:
            ocr = await read_image_message(msg) or ""
        except Exception as e:
            print(f"[WHATSAPP] ERRO ao ler imagem de {sender}: {e}")
            traceback.print_exc()
            ocr = ""

        if not ocr:
            await send_whatsapp_text(
                to=sender,
                text="📷 Não consegui ler a lista da foto. Pode enviar uma foto mais nítida, ou mandar a lista por texto/áudio?",
            )
            return

        print(f"[WHATSAPP] OCR da lista: {ocr[:80]}")
        req_img = MessageRequest(phone=sender, message=ocr)
        try:
            resp_img = await estoque_try_fast_path(req_img, "whatsapp", force=True)
        except Exception as e:
            print(f"[WHATSAPP] ERRO fast-path imagem de {sender}: {e}")
            traceback.print_exc()
            resp_img = None

        if resp_img is None:
            await send_whatsapp_text(
                to=sender,
                text="📷 Li a foto, mas não identifiquei produtos do seu estoque. Tente uma foto mais nítida ou envie a lista por texto.",
            )
        else:
            await send_whatsapp_text(to=sender, text=resp_img.answer)
        return
    elif msg_type in ("DOCUMENT", "FILE"):
        print(f"[WHATSAPP] Documento detectado de {sender} | payload={message_obj}")
        if not sender:
            return
        await send_whatsapp_text(
            to=sender,
            text="📄 Recebemos seu PDF! A movimentação de estoque via PDF está em desenvolvimento. Por enquanto, use o menu de estoque enviando *movimentar estoque*.",
        )
        return
    else:
        text = (message_obj.get("text") or message_obj.get("body") or "").strip()

    if not sender or not text:
        return

    print(f"[WHATSAPP] Mensagem recebida de {sender}: {text[:80]}")

    try:
        await send_whatsapp_text(to=sender, text="...")
        req = MessageRequest(phone=sender, message=text)
        resp = await _dispatch(req, source="whatsapp")

        if resp.menu_options:
            await send_whatsapp_menu(sender, resp.menu_options)
        elif getattr(resp, "messages", None):
            for m in resp.messages:
                if m and m.strip():
                    await send_whatsapp_text(to=sender, text=m)
        else:
            for _chunk in _split_whatsapp_text(resp.answer):
                await send_whatsapp_text(to=sender, text=_chunk)
            if getattr(resp, "excel_url", None):
                from datetime import date as _date
                from urllib.parse import urlparse, parse_qs as _parse_qs
                _parsed_excel_url = urlparse(resp.excel_url)
                if _parsed_excel_url.path.endswith("/api/reposicao-hortifruti/export"):
                    _fname = f"resumo_sugestao_compra_{_date.today().isoformat()}.xlsx"
                    _caption = "📊 Resumo da sugestao de compra por fornecedor"
                else:
                    _qs = _parse_qs(_parsed_excel_url.query)
                    _di = (_qs.get("data_inicio") or [None])[0]
                    _df = (_qs.get("data_fim")    or [None])[0]
                    if _di and _df:
                        _fname = f"perdas_{_di}_{_df}.xlsx"
                    elif _di:
                        _fname = f"perdas_{_di}.xlsx"
                    else:
                        _fname = f"perdas_{_date.today().isoformat()}.xlsx"
                    _caption = "📊 Relatório de perdas em Excel"
                await send_whatsapp_document(to=sender, media_url=resp.excel_url, filename=_fname, caption=_caption)
        print(f"[WHATSAPP] Resposta enviada para {sender}: {resp.answer[:80]}")

        # Loga a conversa
        _thread = sender
        _active = get_active_solution(sender)
        if _active == "sql_analytics":
            _cnpj = (get_session(sender) or {}).get("cnpj") or ""
            _thread = f"{sender}_{_cnpj}" if _cnpj else sender
            _uname = await _get_user_name_cached(sender, _cnpj)
            conv_log.log(
                thread_id=_thread,
                phone=sender,
                cnpj=_cnpj,
                canal="whatsapp",
                user_name=_uname,
                question=text,
                answer=resp.answer,
                metric=getattr(resp, "metric", "") or "",
                insight=getattr(resp, "insight", "") or "",
                intent=getattr(resp, "intent", "") or "",
                sql_generated=getattr(resp, "sql_generated", "") or "",
                sql_error=getattr(resp, "sql_error", "") or "",
                had_error=bool(getattr(resp, "had_error", False)),
                out_of_scope=bool(getattr(resp, "suggest_catalog", False)),
            )

        # Envia insight em background após a resposta principal
        insight = getattr(resp, "insight", None)
        if insight:
            asyncio.create_task(_send_insight_async(
                phone=sender,
                insight=insight,
                question=text,
                answer=resp.answer,
                thread_id=_thread,
            ))

        # Pergunta se quer adicionar ao catálogo (out_of_scope)
        if getattr(resp, "suggest_catalog", False):
            cnpj_for_catalog = (get_session(sender) or {}).get("cnpj") or ""
            set_pending_catalog_suggestion(sender, text, cnpj_for_catalog)
            asyncio.create_task(send_catalog_suggestion_buttons(sender))

    except Exception as e:
        print(f"[WHATSAPP] ERRO processando mensagem de {sender}: {e}")
        traceback.print_exc()
        try:
            await send_whatsapp_text(
                to=sender,
                text="Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente em instantes.",
            )
        except Exception:
            pass


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    """Recebe mensagens inbound do Infobip. Responde IMEDIATAMENTE (200 OK) e processa
    cada mensagem em background — assim o Infobip nunca espera o processamento (que pode
    levar minutos num PDF grande) e não reenvia/duplica a entrega por timeout.
    """
    body = await request.json()

    results = body.get("results") or []
    for msg in results:
        message_obj = msg.get("message") or {}
        # Várias chaves candidatas: o "messageId" do envelope pode mudar a cada retry do
        # Infobip, mas a mídia referenciada (foto/PDF) é a mesma — então checamos também
        # o id/URL da mídia, que tende a ser estável entre tentativas.
        candidate_keys = [
            str(msg.get("messageId") or "").strip(),
            str(message_obj.get("id") or "").strip(),
            str(message_obj.get("mediaId") or "").strip(),
            str(message_obj.get("url") or "").strip(),
            str(message_obj.get("mediaUrl") or "").strip(),
        ]
        if _already_processed(candidate_keys):
            print(f"[WHATSAPP] Mensagem duplicada ignorada (retry do Infobip) | keys={candidate_keys}")
            continue
        asyncio.create_task(_process_whatsapp_message(msg))

    return JSONResponse(content={"status": "ok"}, status_code=200)


# ── Admin: autenticação ───────────────────────────────────────────────────────

def _html_head(title: str) -> str:
    return (f"<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title>"
            f"<link rel='stylesheet' href='/static/admin.css'></head>")


@app.get("/admin/login")
async def admin_login_page(error: str = ""):
    err = "<p class='login-error'>Usuário ou senha incorretos.</p>" if error else ""
    html = (_html_head("iMAIS — Login") +
        f"<body><div class='login-wrap'><div class='login-card'>"
        f"<div class='login-brand'>"
        f"<div class='login-icon'>💬</div>"
        f"<div class='login-name'>iMAIS Admin</div>"
        f"<div class='login-sub'>Portal de Conversas</div>"
        f"</div>{err}"
        f"<form method='post' action='/admin/login'>"
        f"<div class='field'><label>Usuário</label>"
        f"<input type='text' name='usuario' placeholder='Digite seu usuário' required autofocus></div>"
        f"<div class='field'><label>Senha</label>"
        f"<input type='password' name='senha' placeholder='Digite sua senha' required></div>"
        f"<button type='submit' class='btn-login'>Entrar</button>"
        f"</form></div></div></body></html>")
    return HTMLResponse(html)


@app.post("/admin/login")
async def admin_login(request: Request):
    form = await request.form()
    usuario = (form.get("usuario") or "").strip()
    senha   = (form.get("senha")   or "").strip()
    db_path = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
    if conv_log.check_credentials(usuario, senha, db_path):
        token = secrets.token_hex(32)
        _admin_sessions[token] = usuario
        resp = RedirectResponse("/admin/conversations", status_code=302)
        resp.set_cookie("admin_token", token, httponly=True, samesite="lax")
        return resp
    return RedirectResponse("/admin/login?error=1", status_code=302)


@app.get("/admin/logout")
async def admin_logout():
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie("admin_token")
    return resp


# ── Admin: conversas e relatório diário ──────────────────────────────────────

@app.get("/admin/conversations")
async def admin_conversations(request: Request, date: str = ""):
    """Portal de conversas — todas ou filtradas por data YYYY-MM-DD."""
    if not _get_admin_user(request):
        return RedirectResponse("/admin/login", status_code=302)
    from datetime import date as _date
    db_path = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
    print(f"[PORTAL] Lendo de: {os.path.abspath(db_path)}")
    threads = conv_log.list_threads(db_path, for_date=date or None)
    today   = _date.today().isoformat()
    label   = f"Filtro: {date}" if date else "Todas as conversas"

    # Pre-fetch display names (user_name → fantasia fallback) for all threads
    display_names: dict[str, str] = {}
    for t in threads:
        display_names[t["thread_id"]] = await _resolve_display_name(
            t.get("user_name") or "", t.get("cnpj") or ""
        )

    canal_badge = lambda t: ("<span class='badge badge-whatsapp'>WhatsApp</span>" if t.get("canal","whatsapp")=="whatsapp"
                             else "<span class='badge badge-portal'>Portal</span>")
    rows_html = "".join(
        f"<tr onclick=\"location.href='/admin/conversations/{t['thread_id']}'\"> "
        f"<td><a href='/admin/conversations/{t['thread_id']}' class='thread-id'>{t['thread_id']}</a></td>"
        f"<td>{t['phone']}</td>"
        f"<td>{display_names.get(t['thread_id'], '—')}</td>"
        f"<td>{t['cnpj'] or '—'}</td>"
        f"<td>{canal_badge(t)}</td>"
        f"<td><span class='msg-count'>{t['total_msgs']}</span></td>"
        f"<td style='color:var(--gray)'>{t['first_at']}</td>"
        f"<td style='color:var(--gray)'>{t['last_at']}</td></tr>"
        for t in threads
    )
    total_msgs = sum(t["total_msgs"] for t in threads)
    empty = "<tr><td colspan='7'><div class='empty'><div class='empty-icon'>💬</div><div class='empty-text'>Nenhuma conversa encontrada</div></div></td></tr>"
    html = (_html_head("iMAIS — Conversas") +
        f"<body>"
        f"<div class='topbar'>"
        f"  <div class='topbar-logo'>iMAIS Admin <span>Portal de Conversas</span></div>"
        f"  <a href='/admin/ml-stats' class='topbar-link'>🧠 Base de ML</a>"
        f"  <a href='/admin/daily-report?date={date or today}' class='topbar-link'>📧 Relatório</a>"
        f"  <a href='/admin/logout' class='topbar-link'>Sair →</a>"
        f"</div>"
        f"<div class='container'>"
        f"  <div class='page-header'>"
        f"    <div class='page-title'>Conversas</div>"
        f"    <div class='page-sub'>{label} — {len(threads)} thread(s) · {total_msgs} mensagens</div>"
        f"  </div>"
        f"  <div class='stats-row'>"
        f"    <div class='stat-card'><div class='stat-n'>{len(threads)}</div><div class='stat-l'>Threads ativas</div></div>"
        f"    <div class='stat-card'><div class='stat-n'>{total_msgs}</div><div class='stat-l'>Total de mensagens</div></div>"
        f"    <div class='stat-card'><div class='stat-n'>{sum(1 for t in threads if t.get('canal')=='whatsapp')}</div><div class='stat-l'>WhatsApp</div></div>"
        f"    <div class='stat-card'><div class='stat-n'>{sum(1 for t in threads if t.get('canal')=='portal')}</div><div class='stat-l'>Portal</div></div>"
        f"  </div>"
        f"  <div class='card'>"
        f"    <div class='filter-bar'>"
        f"      <form method='get' style='display:flex;gap:8px;align-items:center'>"
        f"        <label>Data:</label>"
        f"        <input type='date' name='date' value='{date}'>"
        f"        <button type='submit' class='btn btn-primary'>Filtrar</button>"
        f"        <a href='/admin/conversations' class='btn btn-outline'>Todas</a>"
        f"      </form>"
        f"    </div>"
        f"    <div class='table-wrap'>"
        f"    <table>"
        f"      <thead><tr><th>Thread</th><th>Telefone</th><th>Usuário</th><th>CNPJ</th><th>Canal</th><th>Msgs</th><th>Início</th><th>Último contato</th></tr></thead>"
        f"      <tbody>{rows_html or empty}</tbody>"
        f"    </table></div>"
        f"  </div>"
        f"</div></body></html>")
    return HTMLResponse(html)


@app.get("/admin/ml-stats")
async def admin_ml_stats(request: Request):
    """Painel da base de aprendizado (etapas #1 NLU e #2 feedback)."""
    if not _get_admin_user(request):
        return RedirectResponse("/admin/login", status_code=302)
    import html as _html
    db_path = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
    stats     = conv_log.training_stats(db_path)
    intents   = conv_log.intent_breakdown(db_path)
    problemas = conv_log.problem_samples(db_path)

    def _g(k):  # get numérico seguro
        return stats.get(k) or 0

    total = _g("total")
    def _pct(n):
        return f"{(100.0 * n / total):.1f}%" if total else "0%"

    stat_cards = "".join([
        f"<div class='stat-card'><div class='stat-n'>{total}</div><div class='stat-l'>Interações totais</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('erros')}</div><div class='stat-l'>Com erro ({_pct(_g('erros'))})</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('reformulacoes')}</div><div class='stat-l'>Reformulações ({_pct(_g('reformulacoes'))})</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('fora_catalogo')}</div><div class='stat-l'>Fora do catálogo</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('insight_fb')}</div><div class='stat-l'>Feedbacks de insight</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('catalog_fb')}</div><div class='stat-l'>Feedbacks de catálogo</div></div>",
        f"<div class='stat-card'><div class='stat-n'>{_g('intents_distintos')}</div><div class='stat-l'>Intenções distintas</div></div>",
    ])

    intent_rows = "".join(
        f"<tr><td>{_html.escape(str(i['intent']))}</td>"
        f"<td><span class='msg-count'>{i['total']}</span></td>"
        f"<td style='color:{'#c0392b' if i['erros'] else 'var(--gray)'}'>{i['erros']}</td>"
        f"<td style='color:{'#c0392b' if i['reformulacoes'] else 'var(--gray)'}'>{i['reformulacoes']}</td>"
        f"<td style='color:{'#c0392b' if i['insight_nao'] else 'var(--gray)'}'>{i['insight_nao']}</td></tr>"
        for i in intents
    ) or "<tr><td colspan='5'><div class='empty'><div class='empty-text'>Sem dados ainda</div></div></td></tr>"

    def _flags(p):
        f = []
        if p.get("had_error"):        f.append("<span class='badge' style='background:#fdecea;color:#c0392b'>erro</span>")
        if p.get("is_reformulation"): f.append("<span class='badge' style='background:#fef5e7;color:#b9770e'>reformulou</span>")
        if p.get("out_of_scope"):     f.append("<span class='badge' style='background:#eaf2f8;color:#2471a3'>fora escopo</span>")
        return " ".join(f)

    problem_rows = "".join(
        f"<tr><td style='color:var(--gray);white-space:nowrap'>{_html.escape(str(p.get('created_at') or ''))}</td>"
        f"<td>{_html.escape(str(p.get('question') or ''))[:120]}</td>"
        f"<td>{_html.escape(str(p.get('intent') or '—'))}</td>"
        f"<td>{_flags(p)}</td>"
        f"<td style='color:#c0392b;font-size:12px'>{_html.escape(str(p.get('sql_error') or ''))[:80]}</td></tr>"
        for p in problemas
    ) or "<tr><td colspan='5'><div class='empty'><div class='empty-icon'>🎉</div><div class='empty-text'>Nenhuma interação problemática registrada</div></div></td></tr>"

    html = (_html_head("iMAIS — Base de ML") +
        f"<body>"
        f"<div class='topbar'>"
        f"  <div class='topbar-logo'>iMAIS Admin <span>Base de Aprendizado</span></div>"
        f"  <a href='/admin/conversations' class='topbar-link'>💬 Conversas</a>"
        f"  <a href='/admin/logout' class='topbar-link'>Sair →</a>"
        f"</div>"
        f"<div class='container'>"
        f"  <div class='page-header'>"
        f"    <div class='page-title'>Base de Aprendizado</div>"
        f"    <div class='page-sub'>Sinais coletados para as etapas de ML — NLU (#1) e feedback (#2)</div>"
        f"  </div>"
        f"  <div class='stats-row'>{stat_cards}</div>"
        f"  <div class='card'>"
        f"    <div class='page-header' style='padding:16px'><div class='page-title' style='font-size:18px'>Por intenção</div>"
        f"      <div class='page-sub'>Onde o classificador/SQL mais erram — prioridade da etapa #1</div></div>"
        f"    <div class='table-wrap'><table>"
        f"      <thead><tr><th>Intenção</th><th>Total</th><th>Erros</th><th>Reformulações</th><th>Insight “Não”</th></tr></thead>"
        f"      <tbody>{intent_rows}</tbody>"
        f"    </table></div>"
        f"  </div>"
        f"  <div class='card' style='margin-top:20px'>"
        f"    <div class='page-header' style='padding:16px'><div class='page-title' style='font-size:18px'>Interações problemáticas</div>"
        f"      <div class='page-sub'>Casos concretos para revisar e alimentar o aprendizado</div></div>"
        f"    <div class='table-wrap'><table>"
        f"      <thead><tr><th>Quando</th><th>Pergunta</th><th>Intenção</th><th>Sinais</th><th>Erro SQL</th></tr></thead>"
        f"      <tbody>{problem_rows}</tbody>"
        f"    </table></div>"
        f"  </div>"
        f"</div></body></html>")
    return HTMLResponse(html)


@app.get("/admin/conversations/{thread_id}")
async def admin_thread(request: Request, thread_id: str):
    """Histórico completo de uma thread."""
    if not _get_admin_user(request):
        return RedirectResponse("/admin/login", status_code=302)
    db_path = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
    history = conv_log.get_history(thread_id, db_path)

    canal = history[0].get("canal", "whatsapp") if history else "whatsapp"
    phone = history[0].get("phone", "") if history else ""
    cnpj  = history[0].get("cnpj", "") if history else ""
    canal_badge = ("<span class='badge badge-whatsapp'>WhatsApp</span>" if canal == "whatsapp"
                   else "<span class='badge badge-portal'>Portal</span>")

    msgs_html = ""
    for h in history:
        q   = (h.get("question") or "").replace("<", "&lt;")
        a   = (h.get("answer")   or "").replace("<", "&lt;").replace("\n", "<br>")
        ins = (h.get("insight")  or "").replace("<", "&lt;").replace("\n", "<br>")
        metric = h.get("metric") or ""
        ts = h.get("created_at", "")
        metric_tag = f"<span class='badge badge-metric'>{metric}</span>" if metric else ""
        msgs_html += (
            f"<div class='msg-wrap user'>"
            f"  <div class='msg-label'>👤 Usuário &nbsp;{metric_tag}</div>"
            f"  <div class='bubble bubble-user'>{q}</div>"
            f"  <div class='msg-time'>{ts}</div>"
            f"</div>"
            f"<div class='msg-wrap bot'>"
            f"  <div class='msg-label'>🤖 iMAIS</div>"
            f"  <div class='bubble bubble-bot'>{a}</div>"
            + (f"  <div class='bubble-insight'><div class='bubble-insight-header'>💡 Insight iMAIS</div>{ins}</div>" if ins else "") +
            f"</div>"
        )

    html = (_html_head(f"Thread — {thread_id}") +
        f"<body>"
        f"<div class='topbar'>"
        f"  <div class='topbar-logo'>iMAIS Admin <span>Conversa</span></div>"
        f"  <a href='/admin/logout' class='topbar-link'>Sair →</a>"
        f"</div>"
        f"<div class='container'>"
        f"  <div class='breadcrumb'><a href='/admin/conversations'>← Conversas</a> / {thread_id}</div>"
        f"  <div class='page-header'>"
        f"    <div class='page-title'>Conversa</div>"
        f"    <div class='page-sub'>{len(history)} mensagens</div>"
        f"  </div>"
        f"  <div class='info-grid'>"
        f"    <div class='info-item'><div class='info-label'>Thread</div><div class='info-value' style='font-family:monospace;font-size:12px'>{thread_id}</div></div>"
        f"    <div class='info-item'><div class='info-label'>Telefone</div><div class='info-value'>{phone or '—'}</div></div>"
        f"    <div class='info-item'><div class='info-label'>CNPJ</div><div class='info-value'>{cnpj or '—'}</div></div>"
        f"    <div class='info-item'><div class='info-label'>Canal</div><div class='info-value'>{canal_badge}</div></div>"
        f"  </div>"
        f"  <div class='card'><div class='chat-feed'>"
        + (msgs_html or "<div class='empty'><div class='empty-icon'>💬</div><div class='empty-text'>Nenhuma mensagem</div></div>") +
        f"  </div></div>"
        f"</div></body></html>")
    return HTMLResponse(html)


@app.get("/admin/daily-report")
async def admin_daily_report(request: Request, date: str = ""):
    """Envia relatório diário por email."""
    if not _get_admin_user(request):
        return RedirectResponse("/admin/login", status_code=302)
    from datetime import date as _date
    target = date or _date.today().isoformat()
    db_path = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
    summary = conv_log.daily_summary(target, db_path)

    # Pre-fetch display names (user_name → fantasia fallback) for all threads
    rpt_names: dict[str, str] = {}
    for t in summary["threads"]:
        rpt_names[t["thread_id"]] = await _resolve_display_name(
            t.get("user_name") or "", t.get("cnpj") or ""
        )

    threads_rows = "".join(
        f"<tr><td>{t['thread_id']}</td><td>{t['phone']}</td>"
        f"<td>{rpt_names.get(t['thread_id'], '—')}</td>"
        f"<td>{t['cnpj'] or '—'}</td><td>{t['total_msgs']}</td>"
        f"<td>{t['first_at']}</td><td>{t['last_at']}</td></tr>"
        for t in summary["threads"]
    )

    detail_html = ""
    for tid, msgs in summary["details"].items():
        uname = rpt_names.get(tid, "—")
        for m in msgs:
            q = (m.get("question") or "").replace("<", "&lt;")
            a = (m.get("answer") or "").replace("<", "&lt;").replace("\n", "<br>")
            detail_html += (
                f"<tr><td>{tid}</td><td>{uname}</td>"
                f"<td>{m.get('created_at','')}</td><td>{q}</td><td>{a}</td></tr>"
            )

    html_body = f"""
    <h2>📊 Relatório Diário iMAIS — {target}</h2>
    <p><b>Threads ativas:</b> {len(summary['threads'])} &nbsp;|&nbsp;
    <b>Total de mensagens:</b> {summary['total_msgs']}</p>
    <h3>Resumo por thread</h3>
    <table border='1' cellpadding='6' style='border-collapse:collapse;font-size:13px'>
    <tr style='background:#f4f4f4'><th>Thread</th><th>Telefone</th><th>Usuário</th><th>CNPJ</th>
    <th>Msgs</th><th>Início</th><th>Último</th></tr>
    {threads_rows or '<tr><td colspan=7>Nenhuma thread</td></tr>'}
    </table>
    <h3>Conversas do dia</h3>
    <table border='1' cellpadding='6' style='border-collapse:collapse;font-size:12px;width:100%'>
    <tr style='background:#f4f4f4'><th>Thread</th><th>Usuário</th><th>Hora</th><th>Pergunta</th><th>Resposta</th></tr>
    {detail_html or '<tr><td colspan=5>Nenhuma mensagem</td></tr>'}
    </table>
    <p style='color:gray;font-size:11px'>Email gerado automaticamente. Por favor, não responda.</p>
    """

    from shared.email_client import _post_email_all
    await _post_email_all(f"[iMAIS] Relatório diário — {target}", html_body)
    return JSONResponse({"status": "email enviado", "date": target,
                         "threads": len(summary["threads"]), "msgs": summary["total_msgs"]})


# ── Portal: feedback de insight ──────────────────────────────────────────────

@app.post("/portal/insight/feedback")
async def portal_insight_feedback(request: Request):
    """Recebe feedback (sim/nao) do insight exibido no portal."""
    body    = await request.json()
    cnpj    = (body.get("cnpj")     or "").strip()
    phone   = (body.get("phone")    or "").strip()
    feedback = (body.get("feedback") or "").strip()  # "sim" ou "nao"

    if not cnpj or not feedback:
        return JSONResponse({"error": "cnpj e feedback são obrigatórios"}, status_code=400)

    session_key = f"portal_{cnpj}"
    pending = get_pending_insight(session_key)
    if pending and feedback.lower() in ("sim", "nao", "yes", "no"):
        label = "Sim ✓" if feedback.lower() in ("sim", "yes") else "Não ✗"
        asyncio.create_task(send_insight_feedback_email(
            phone=phone or cnpj,
            question=pending.get("question", ""),
            answer=pending.get("answer", ""),
            insight=pending.get("insight", ""),
            feedback=label,
        ))
        # Salva feedback no SQLite
        _db = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")
        conv_log.update_insight_feedback(session_key, label, _db)
        clear_pending_insight(session_key)

    return JSONResponse({"status": "ok"})


# ── Portal BH (POC sem CNPJ) ─────────────────────────────────────────────────

@app.post("/portal/message/bh")
async def portal_message_bh(request: Request):
    """Recebe mensagens do portal web para a POC BH. Sem CNPJ."""
    body = await request.json()

    message = (body.get("message") or "").strip()

    if not message:
        return JSONResponse(content={"error": "message é obrigatório"}, status_code=400)

    print(f"[PORTAL-BH] Mensagem recebida: {message[:80]}")

    try:
        req = BhMessageRequest(message=message)
        resp = await bh_handle_message(req)
        return JSONResponse(content={"answer": _strip_markdown(resp.answer)})
    except Exception as e:
        print(f"[PORTAL-BH] ERRO: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"error": "Erro ao processar mensagem. Tente novamente."},
            status_code=500,
        )


# ── Webhook Infobip BH (WhatsApp inbound — POC sem CNPJ) ─────────────────────

@app.post("/whatsapp/webhook/bh")
async def whatsapp_webhook_bh(request: Request):
    """Recebe mensagens inbound do Infobip para a POC BH e responde via WhatsApp."""
    body = await request.json()

    results = body.get("results") or []
    for msg in results:
        sender = (msg.get("from") or "").strip()
        message_obj = msg.get("message") or {}
        text = (message_obj.get("text") or message_obj.get("body") or "").strip()

        if not sender or not text:
            continue

        print(f"[WHATSAPP-BH] Mensagem recebida de {sender}: {text[:80]}")

        try:
            await send_whatsapp_text(to=sender, text="...")
            req = BhMessageRequest(message=text)
            resp = await bh_handle_message(req)
            answer = resp.answer

            await send_whatsapp_text(to=sender, text=answer)
            print(f"[WHATSAPP-BH] Resposta enviada para {sender}: {answer[:80]}")

        except Exception as e:
            print(f"[WHATSAPP-BH] ERRO processando mensagem de {sender}: {e}")
            traceback.print_exc()
            try:
                await send_whatsapp_text(
                    to=sender,
                    text="Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente em instantes.",
                )
            except Exception:
                pass

    return JSONResponse(content={"status": "ok"}, status_code=200)


# ── Health check (comum) ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/debug/message")
async def debug_message(request: Request):
    """Simula mensagem WhatsApp pelo telefone, sem passar pela Infobip.
    Body: { "phone": "34996658741", "message": "quanto vendi hoje?" }
    """
    from shared.db_client import normalize_phone, get_cnpjs_for_phone
    body    = await request.json()
    phone   = normalize_phone(body.get("phone") or "")
    if phone and not phone.startswith("55"):
        phone = "55" + phone
    # 55 + DDD(2) + 8 dígitos = 12 → insere 9 após o DDD (igual ao _phone_norm_sql)
    if len(phone) == 12:
        phone = phone[:4] + "9" + phone[4:]
    message = (body.get("message") or "").strip()

    if not phone:
        return JSONResponse({"error": "phone é obrigatório"}, status_code=400)
    if not message:
        return JSONResponse({"error": "message é obrigatório"}, status_code=400)

    cnpjs = await get_cnpjs_for_phone(phone)
    if not cnpjs:
        return JSONResponse({"error": f"Nenhum CNPJ encontrado para o telefone {phone}"}, status_code=404)

    cnpj = cnpjs[0]["cnpj"]
    print(f"[DEBUG MSG] phone={phone} → cnpj={cnpj}: {message[:80]}")

    try:
        req  = MessageRequest(phone=phone, message=message, cnpj=cnpj)
        resp = await _dispatch(req, source="portal")
        insight = getattr(resp, "insight", None) or ""
        return JSONResponse({
            "phone":   phone,
            "cnpj":    cnpj,
            "answer":  _strip_markdown(resp.answer),
            "insight": _strip_markdown(insight) if insight else None,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Export de perdas em Excel ────────────────────────────────────────────────

@app.api_route("/api/perdas/export", methods=["GET", "HEAD"])
async def export_perdas_excel(
    cnpj: str,
    data_inicio: str | None = None,
    data_fim: str | None = None,
):
    """Gera e retorna o histórico de movimentações manuais em Excel (.xlsx)."""
    from datetime import date, timedelta
    from fastapi.responses import Response as _Resp
    from solutions.sql_analytics.graph_agent import (
        _PERDAS_HISTORICO_SQL,
        _gerar_excel_perdas,
    )
    from shared.db_client import parse_rows, run_query, cleanup_sql

    if not cnpj:
        return JSONResponse({"error": "cnpj é obrigatório"}, status_code=400)

    today = date.today()
    if data_inicio and data_fim:
        data_filter = f"DATE(e.DATA_RELATORIO) BETWEEN '{data_inicio}' AND '{data_fim}'"
        periodo = f"{data_inicio} até {data_fim}"
    elif data_inicio:
        data_filter = f"DATE(e.DATA_RELATORIO) >= '{data_inicio}'"
        periodo = f"desde {data_inicio}"
    else:
        start = (today - timedelta(days=6)).isoformat()
        data_filter = f"DATE(e.DATA_RELATORIO) >= '{start}'"
        periodo = "últimos 7 dias"

    try:
        sql = cleanup_sql(_PERDAS_HISTORICO_SQL.format(cnpj=cnpj, data_filter=data_filter))
        rows = parse_rows(await run_query(sql))
        excel_bytes = _gerar_excel_perdas(rows, periodo)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    filename = f"perdas_{cnpj}_{today.isoformat()}.xlsx"
    return _Resp(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.api_route("/api/reposicao-hortifruti/export", methods=["GET", "HEAD"])
async def export_reposicao_hortifruti_excel(cnpj: str):
    """Gera o resumo da sugestão de compra por fornecedor em Excel."""
    from datetime import date
    from fastapi.responses import Response as _Resp
    from solutions.sql_analytics.graph_agent import (
        _HORTIFRUTI_REPORT_SQL,
        _gerar_excel_resumo_reposicao,
        _triage_hortifruti,
    )
    from shared.db_client import cleanup_sql, parse_rows, run_query

    if not cnpj:
        return JSONResponse({"error": "cnpj é obrigatório"}, status_code=400)

    try:
        eligibility_filter = """
  AND c.DATA_ULTIMA_COMPRA >= ref.max_data - INTERVAL 21 DAYS
  AND GREATEST(
    COALESCE(v.CONSUMO_CICLO, 0),
    COALESCE(mv.MEDIA_VENDA_SEMANAL, 0),
    COALESCE(h.MEDIA_COMPRA_SEMANAL, 0)
  ) > s.SALDO_ATUAL"""
        rows = parse_rows(await run_query(cleanup_sql(_HORTIFRUTI_REPORT_SQL.format(
            cnpj=cnpj,
            eligibility_filter=eligibility_filter,
            product_filter="",
        ))))
        excel_bytes = _gerar_excel_resumo_reposicao(_triage_hortifruti(rows))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    filename = f"resumo_sugestao_compra_{cnpj}_{date.today().isoformat()}.xlsx"
    return _Resp(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
