from __future__ import annotations

import time
from threading import Lock
from typing import Optional

SESSION_TTL_SECONDS = 30 * 60  # 30 minutos

_sessions: dict[str, dict] = {}
_lock = Lock()


def get_session(phone: str) -> Optional[dict]:
    """Retorna a sessão válida (não expirada) para o telefone, ou None."""
    with _lock:
        s = _sessions.get(phone)
        if not s:
            return None
        if time.time() >= s.get("expires_at", 0):
            _sessions.pop(phone, None)
            return None
        return dict(s)


def set_cnpj(phone: str, cnpj: str) -> None:
    """Grava o CNPJ selecionado na sessão e reseta o TTL. Preserva active_solution."""
    with _lock:
        existing = _sessions.get(phone) or {}
        _sessions[phone] = {
            "cnpj":            cnpj,
            "pending_options": None,
            "active_solution": existing.get("active_solution") or "sql_analytics",
            "expires_at":      time.time() + SESSION_TTL_SECONDS,
        }


def set_pending_options(
    phone: str,
    options: list[dict],
    pending_message: str | None = None,
    pending_activation: str | None = None,
) -> None:
    """Marca que o usuário precisa escolher um CNPJ. Preserva active_solution."""
    with _lock:
        existing = _sessions.get(phone) or {}
        _sessions[phone] = {
            "cnpj":               None,
            "pending_options":    options,
            "pending_message":    pending_message,
            "pending_activation": pending_activation,
            "active_solution":    existing.get("active_solution") or "sql_analytics",
            "expires_at":         time.time() + SESSION_TTL_SECONDS,
        }


def set_active_solution(phone: str, solution: str) -> None:
    """Grava a solução ativa na sessão sem alterar o CNPJ."""
    with _lock:
        existing = _sessions.get(phone) or {}
        existing["active_solution"] = solution
        existing.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[phone] = existing


def get_active_solution(phone: str) -> str:
    """Retorna a solução ativa para o telefone. Default: 'sql_analytics'."""
    session = get_session(phone)
    return (session or {}).get("active_solution") or "sql_analytics"


def extend_session(phone: str) -> None:
    """Renova o TTL da sessão por mais 30 minutos."""
    with _lock:
        s = _sessions.get(phone)
        if s:
            s["expires_at"] = time.time() + SESSION_TTL_SECONDS


def clear_session(phone: str) -> None:
    """Remove a sessão (força o usuário a escolher CNPJ de novo)."""
    with _lock:
        _sessions.pop(phone, None)


# ── Estado por solução (sub-dict isolado) ─────────────────────────────────────

def get_estoque_state(session_key: str) -> dict:
    """Retorna o sub-dict de estado do módulo estoque, ou {} se não existir."""
    s = get_session(session_key)
    return (s or {}).get("estoque") or {}


def update_estoque_state(session_key: str, **fields) -> None:
    """Mescla campos no estado do estoque sem alterar o restante da sessão."""
    with _lock:
        s = _sessions.get(session_key) or {}
        estoque = s.get("estoque") or {}
        for k, v in fields.items():
            if v is None:
                estoque.pop(k, None)
            else:
                estoque[k] = v
        s["estoque"] = estoque
        s.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[session_key] = s


def set_pending_catalog_suggestion(phone: str, question: str, cnpj: str = "") -> None:
    with _lock:
        s = _sessions.get(phone) or {}
        s["pending_catalog"] = {"question": question, "cnpj": cnpj}
        s.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[phone] = s


def get_pending_catalog_suggestion(phone: str) -> dict | None:
    s = get_session(phone)
    return (s or {}).get("pending_catalog")


def clear_pending_catalog_suggestion(phone: str) -> None:
    with _lock:
        s = _sessions.get(phone)
        if s:
            s.pop("pending_catalog", None)


def set_pending_insight(phone: str, data: dict) -> None:
    """Guarda o insight pendente de feedback (question, answer, insight)."""
    with _lock:
        s = _sessions.get(phone) or {}
        s["pending_insight"] = data
        s.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[phone] = s


def get_pending_insight(phone: str) -> dict | None:
    """Retorna o insight pendente de feedback, ou None."""
    s = get_session(phone)
    return (s or {}).get("pending_insight")


def clear_pending_insight(phone: str) -> None:
    """Remove o insight pendente após receber o feedback."""
    with _lock:
        s = _sessions.get(phone)
        if s:
            s.pop("pending_insight", None)


def clear_estoque_state(session_key: str) -> None:
    """Remove apenas o sub-dict do estoque, preservando CNPJ/active_solution."""
    with _lock:
        s = _sessions.get(session_key)
        if s and "estoque" in s:
            s.pop("estoque", None)


def set_pending_insight_theme(phone: str) -> None:
    """Marca que o próximo turno é uma resposta de tema para geração de insight."""
    with _lock:
        s = _sessions.get(phone) or {}
        s["pending_insight_theme"] = True
        s.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[phone] = s


def get_pending_insight_theme(phone: str) -> bool:
    """Retorna True se o próximo turno deve ser tratado como tema de insight."""
    s = get_session(phone)
    return bool((s or {}).get("pending_insight_theme"))


def clear_pending_insight_theme(phone: str) -> None:
    """Remove o flag de tema de insight pendente."""
    with _lock:
        s = _sessions.get(phone)
        if s:
            s.pop("pending_insight_theme", None)


def set_last_vague_insight_metric(phone: str, metric: str) -> None:
    """Lembra qual métrica foi usada na última vez que o cliente pediu um
    "insight"/"análise" genérico — usado para rotacionar o tema e evitar
    repetir sempre a mesma resposta (ex: sempre faturamento)."""
    with _lock:
        s = _sessions.get(phone) or {}
        s["last_vague_insight_metric"] = metric
        s.setdefault("expires_at", time.time() + SESSION_TTL_SECONDS)
        _sessions[phone] = s


def get_last_vague_insight_metric(phone: str) -> str | None:
    """Retorna a métrica usada da última vez que o cliente pediu um insight genérico."""
    s = get_session(phone)
    return (s or {}).get("last_vague_insight_metric")
