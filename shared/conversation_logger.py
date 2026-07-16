"""Log simples de conversas em SQLite — tabela separada dos checkpoints LangGraph."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import date, datetime
from typing import Optional

_DB_PATH = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")


def _conn(db_path: str = _DB_PATH) -> sqlite3.Connection:
    import os as _os
    abs_path = _os.path.abspath(db_path)
    c = sqlite3.connect(abs_path)
    c.row_factory = sqlite3.Row
    return c


def _hash(senha: str) -> str:
    return hashlib.sha256(senha.encode()).hexdigest()


def setup_users(db_path: str = _DB_PATH) -> None:
    """Cria portal_usuarios e insere admin padrão se ainda não existir."""
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS portal_usuarios (
                usuario TEXT PRIMARY KEY,
                senha   TEXT NOT NULL
            )
        """)
        c.execute(
            "INSERT OR IGNORE INTO portal_usuarios (usuario, senha) VALUES (?, ?)",
            ("admin", _hash("abc12345678xyz")),
        )


def check_credentials(usuario: str, senha: str, db_path: str = _DB_PATH) -> bool:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT senha FROM portal_usuarios WHERE usuario = ?", (usuario,)
        ).fetchone()
    return bool(row and row["senha"] == _hash(senha))


def setup(db_path: str = _DB_PATH) -> None:
    """Cria a tabela conversations se não existir."""
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                phone       TEXT,
                cnpj        TEXT,
                canal       TEXT DEFAULT 'whatsapp',
                question    TEXT,
                answer      TEXT,
                metric      TEXT,
                insight     TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversations(thread_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_date  ON conversations(created_at)")
        # Colunas adicionadas incrementalmente (ALTER idempotente).
        # Base de ML: intenção, SQL gerado, sinais de erro e feedback.
        for coldef in (
            "user_name TEXT",
            "insight_feedback TEXT",
            "intent TEXT",             # intenção classificada (#1 NLU)
            "sql_generated TEXT",      # SQL executado (#1 SQL auto-corretivo)
            "sql_error TEXT",          # erro da query, se houve (aprender com erros)
            "had_error INTEGER",       # 1 = resposta foi fallback de erro
            "out_of_scope INTEGER",    # 1 = caiu fora do catálogo
            "is_reformulation INTEGER",# 1 = repetição da pergunta anterior (falha implícita)
            "catalog_feedback TEXT",   # sim/não da sugestão de catálogo (#2)
        ):
            try:
                c.execute(f"ALTER TABLE conversations ADD COLUMN {coldef}")
            except Exception:
                pass


def _is_reformulation(c: sqlite3.Connection, thread_id: str, question: str,
                      window_min: int = 30) -> bool:
    """Heurística: a pergunta atual repete (quase) a anterior da mesma thread dentro
    da janela de tempo? Sinal implícito de que a resposta anterior não resolveu."""
    q = (question or "").strip().lower()
    if not q:
        return False
    row = c.execute(
        """SELECT question FROM conversations
           WHERE thread_id = ?
             AND created_at >= datetime('now','localtime', ?)
           ORDER BY created_at DESC LIMIT 1""",
        (thread_id, f"-{int(window_min)} minutes"),
    ).fetchone()
    if not row:
        return False
    prev = (row["question"] or "").strip().lower()
    if not prev:
        return False
    # igualdade exata ou forte sobreposição de tokens
    if prev == q:
        return True
    a, b = set(prev.split()), set(q.split())
    if not a or not b:
        return False
    jaccard = len(a & b) / len(a | b)
    return jaccard >= 0.8


def log(
    thread_id: str,
    phone: str,
    cnpj: str,
    question: str,
    answer: str,
    metric: str = "",
    insight: str = "",
    canal: str = "whatsapp",
    user_name: str = "",
    intent: str = "",
    sql_generated: str = "",
    sql_error: str = "",
    had_error: bool = False,
    out_of_scope: bool = False,
    db_path: str = _DB_PATH,
) -> None:
    """Registra uma conversa com os sinais de NLU/erro para a base de ML."""
    try:
        with _conn(db_path) as c:
            reformulation = _is_reformulation(c, thread_id, question)
            c.execute(
                "INSERT INTO conversations "
                "(thread_id, phone, cnpj, canal, user_name, question, answer, metric, insight, "
                " intent, sql_generated, sql_error, had_error, out_of_scope, is_reformulation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, phone, cnpj, canal, user_name or "", question, answer,
                 metric or "", insight or "", intent or "", sql_generated or "",
                 sql_error or "", 1 if had_error else 0, 1 if out_of_scope else 0,
                 1 if reformulation else 0),
            )
    except Exception as e:
        print(f"[CONV LOG] ERRO: {e}")

    import os as _os
    print(f"[CONV LOG] Salvo em: {_os.path.abspath(db_path)}")


def list_threads(db_path: str = _DB_PATH, for_date: Optional[str] = None) -> list[dict]:
    """Lista threads com resumo. Se for_date (YYYY-MM-DD), filtra por data."""
    date_filter = f"AND date(created_at) = '{for_date}'" if for_date else ""
    with _conn(db_path) as c:
        rows = c.execute(f"""
            SELECT
                thread_id,
                phone,
                cnpj,
                canal,
                MAX(user_name) AS user_name,
                COUNT(*) AS total_msgs,
                MIN(created_at) AS first_at,
                MAX(created_at) AS last_at
            FROM conversations
            WHERE 1=1 {date_filter}
            GROUP BY thread_id
            ORDER BY last_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_history(thread_id: str, db_path: str = _DB_PATH, for_date: str = "") -> list[dict]:
    """Retorna as trocas de uma thread em ordem cronológica.
    Se for_date (YYYY-MM-DD) informado, filtra apenas as mensagens daquele dia.
    """
    date_filter = f"AND date(created_at) = '{for_date}'" if for_date else ""
    with _conn(db_path) as c:
        rows = c.execute(
            f"SELECT * FROM conversations WHERE thread_id = ? {date_filter} ORDER BY created_at",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_last_report_date(db_path: str = _DB_PATH) -> str:
    """Retorna a data do último relatório enviado (YYYY-MM-DD) ou string vazia."""
    with _conn(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS report_log (
                id         INTEGER PRIMARY KEY,
                report_date TEXT NOT NULL,
                sent_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        row = c.execute("SELECT report_date FROM report_log ORDER BY id DESC LIMIT 1").fetchone()
    return row["report_date"] if row else ""


def mark_report_sent(report_date: str, db_path: str = _DB_PATH) -> None:
    """Registra que o relatório do dia foi enviado."""
    with _conn(db_path) as c:
        c.execute("CREATE TABLE IF NOT EXISTS report_log (id INTEGER PRIMARY KEY, report_date TEXT NOT NULL, sent_at TEXT NOT NULL DEFAULT (datetime('now','localtime')))")
        c.execute("INSERT INTO report_log (report_date) VALUES (?)", (report_date,))


def update_insight_feedback(
    thread_id: str,
    feedback: str,
    db_path: str = _DB_PATH,
) -> None:
    """Registra o feedback (Sim/Não) do insight na conversa mais recente da thread."""
    try:
        with _conn(db_path) as c:
            c.execute(
                """UPDATE conversations SET insight_feedback = ?
                   WHERE id = (
                       SELECT id FROM conversations
                       WHERE thread_id = ?
                         AND insight IS NOT NULL AND insight != ''
                       ORDER BY created_at DESC LIMIT 1
                   )""",
                (feedback, thread_id),
            )
        print(f"[CONV LOG] insight_feedback='{feedback}' salvo para thread={thread_id}")
    except Exception as e:
        print(f"[CONV LOG] ERRO ao salvar feedback: {e}")


def update_catalog_feedback(
    thread_id: str,
    feedback: str,
    db_path: str = _DB_PATH,
) -> None:
    """Registra a resposta (sim/não) da sugestão de catálogo na conversa out_of_scope
    mais recente da thread."""
    try:
        with _conn(db_path) as c:
            c.execute(
                """UPDATE conversations SET catalog_feedback = ?
                   WHERE id = (
                       SELECT id FROM conversations
                       WHERE thread_id = ? AND out_of_scope = 1
                       ORDER BY created_at DESC LIMIT 1
                   )""",
                (feedback, thread_id),
            )
        print(f"[CONV LOG] catalog_feedback='{feedback}' salvo para thread={thread_id}")
    except Exception as e:
        print(f"[CONV LOG] ERRO ao salvar catalog_feedback: {e}")


def training_stats(db_path: str = _DB_PATH) -> dict:
    """Painel rápido da base de aprendizado: contagem dos sinais coletados.
    Usado para acompanhar quando há volume suficiente para treinar modelos."""
    with _conn(db_path) as c:
        row = c.execute("""
            SELECT
                COUNT(*)                                         AS total,
                SUM(CASE WHEN had_error = 1 THEN 1 ELSE 0 END)   AS erros,
                SUM(CASE WHEN sql_error IS NOT NULL AND sql_error != '' THEN 1 ELSE 0 END) AS sql_erros,
                SUM(CASE WHEN out_of_scope = 1 THEN 1 ELSE 0 END) AS fora_catalogo,
                SUM(CASE WHEN is_reformulation = 1 THEN 1 ELSE 0 END) AS reformulacoes,
                SUM(CASE WHEN insight_feedback IS NOT NULL AND insight_feedback != '' THEN 1 ELSE 0 END) AS insight_fb,
                SUM(CASE WHEN catalog_feedback IS NOT NULL AND catalog_feedback != '' THEN 1 ELSE 0 END) AS catalog_fb,
                COUNT(DISTINCT intent)                           AS intents_distintos
            FROM conversations
        """).fetchone()
    return dict(row) if row else {}


def intent_breakdown(db_path: str = _DB_PATH, limit: int = 20) -> list[dict]:
    """Distribuição por intenção com taxa de erro/reformulação — insumo da etapa #1
    (identificar quais intenções o classificador/SQL mais erram)."""
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT
                COALESCE(NULLIF(intent, ''), '(sem intent)') AS intent,
                COUNT(*)                                          AS total,
                SUM(CASE WHEN had_error = 1 THEN 1 ELSE 0 END)    AS erros,
                SUM(CASE WHEN is_reformulation = 1 THEN 1 ELSE 0 END) AS reformulacoes,
                SUM(CASE WHEN insight_feedback LIKE 'N%' THEN 1 ELSE 0 END) AS insight_nao
            FROM conversations
            GROUP BY 1
            ORDER BY total DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def problem_samples(db_path: str = _DB_PATH, limit: int = 30) -> list[dict]:
    """Interações problemáticas (erro, reformulação ou fora de escopo) — os casos
    concretos que alimentam o aprendizado das etapas #1 e #2."""
    with _conn(db_path) as c:
        rows = c.execute("""
            SELECT created_at, thread_id, question, intent,
                   had_error, is_reformulation, out_of_scope, sql_error, catalog_feedback
            FROM conversations
            WHERE had_error = 1 OR is_reformulation = 1 OR out_of_scope = 1
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def truncate(db_path: str = _DB_PATH) -> None:
    """Remove todas as conversas (chamado após envio do relatório diário)."""
    with _conn(db_path) as c:
        c.execute("DELETE FROM conversations")
    print("[CONV LOG] Banco de conversas limpo ✓")


def daily_summary(for_date: str = "", db_path: str = _DB_PATH) -> dict:
    """Resumo do dia para o email."""
    if not for_date:
        for_date = date.today().isoformat()
    threads = list_threads(db_path, for_date=for_date)
    total_msgs = sum(t["total_msgs"] for t in threads)
    details = {t["thread_id"]: get_history(t["thread_id"], db_path, for_date=for_date) for t in threads}
    return {"date": for_date, "threads": threads, "total_msgs": total_msgs, "details": details}
