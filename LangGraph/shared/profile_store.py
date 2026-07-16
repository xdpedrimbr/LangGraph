"""Memória de longo prazo por CNPJ — armazena preferências e padrões de uso."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import date
from typing import Optional

DB_PATH = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")

_DDL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    cnpj            TEXT PRIMARY KEY,
    metric_counts   TEXT NOT NULL DEFAULT '{}',
    period_counts   TEXT NOT NULL DEFAULT '{}',
    last_seen       TEXT,
    total_queries   INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        con.execute(_DDL)
        con.commit()
        yield con
    finally:
        con.close()


def get_profile(cnpj: str) -> dict:
    """Retorna o perfil do CNPJ ou dict vazio se não existir."""
    if not cnpj:
        return {}
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM user_profiles WHERE cnpj = ?", (cnpj,)
            ).fetchone()
        if not row:
            return {}
        return {
            "cnpj":          row["cnpj"],
            "metric_counts": json.loads(row["metric_counts"] or "{}"),
            "period_counts": json.loads(row["period_counts"] or "{}"),
            "last_seen":     row["last_seen"],
            "total_queries": row["total_queries"],
            "notes":         row["notes"],
        }
    except Exception as e:
        print(f"[profile_store] get_profile ERRO: {e}")
        return {}


def update_profile(cnpj: str, metric: str, period_type: str) -> None:
    """Atualiza contadores de métrica e período após uma consulta bem-sucedida."""
    if not cnpj or not metric or metric == "outro":
        return
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT metric_counts, period_counts, total_queries FROM user_profiles WHERE cnpj = ?",
                (cnpj,),
            ).fetchone()

            if row:
                mc = Counter(json.loads(row["metric_counts"] or "{}"))
                pc = Counter(json.loads(row["period_counts"] or "{}"))
                total = (row["total_queries"] or 0) + 1
            else:
                mc, pc, total = Counter(), Counter(), 1

            mc[metric] += 1
            if period_type:
                pc[period_type] += 1

            con.execute(
                """INSERT INTO user_profiles (cnpj, metric_counts, period_counts, last_seen, total_queries)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(cnpj) DO UPDATE SET
                     metric_counts = excluded.metric_counts,
                     period_counts = excluded.period_counts,
                     last_seen     = excluded.last_seen,
                     total_queries = excluded.total_queries""",
                (
                    cnpj,
                    json.dumps(dict(mc)),
                    json.dumps(dict(pc)),
                    date.today().isoformat(),
                    total,
                ),
            )
            con.commit()
    except Exception as e:
        print(f"[profile_store] update_profile ERRO: {e}")


def build_profile_hint(profile: dict) -> Optional[str]:
    """Gera uma dica de contexto curta para injetar no preprocess."""
    if not profile or not profile.get("total_queries"):
        return None
    mc = profile.get("metric_counts") or {}
    if not mc:
        return None
    top = sorted(mc.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ", ".join(f"{m} ({n}x)" for m, n in top)
    total = profile.get("total_queries", 0)
    return f"[Perfil do lojista: {total} consultas anteriores. Métricas mais pedidas: {top_str}]"
