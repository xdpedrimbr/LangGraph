#!/usr/bin/env python3
"""
Script diário iMAIS — envia conversas para Databricks + relatório por email.

Configure no Windows Task Scheduler para rodar às 23:59:
  Programa : python
  Argumentos: C:\inetpub\wwwroot\LangGraph\scripts\daily_report.py
  Iniciar em: C:\inetpub\wwwroot\LangGraph
"""

import asyncio
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import shared.conversation_logger as conv_log
from shared.db_client import DBX_HOST, DBX_HEADERS, DBX_WAREHOUSE_ID

DB_PATH = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")


# ── Databricks ────────────────────────────────────────────────────────────────

async def _execute_dbx(sql: str) -> dict:
    import httpx
    payload = {
        "statement":          sql,
        "warehouse_id":       DBX_WAREHOUSE_ID,
        "result_disposition": "INLINE",
        "format":             "JSON_ARRAY",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{DBX_HOST}/api/2.0/sql/statements",
            headers=DBX_HEADERS, json=payload,
        )
        r.raise_for_status()
        sid = r.json().get("statement_id")

        for _ in range(240):
            pr = await client.get(
                f"{DBX_HOST}/api/2.0/sql/statements/{sid}",
                headers=DBX_HEADERS,
            )
            js    = pr.json()
            state = (js.get("status", {}).get("state", "")).upper()
            if state == "SUCCEEDED":
                return js
            if state in ("FAILED", "CANCELED", "CANCELLED"):
                err = js.get("status", {}).get("error", {}).get("message", state)
                raise RuntimeError(f"Databricks {state}: {err}")
            await asyncio.sleep(0.5)

    raise TimeoutError("Timeout aguardando Databricks")


def _esc(v) -> str:
    return str(v or "").replace("'", "''").replace("\n", " ").replace("\r", "")[:4000]


async def _table_is_empty() -> bool:
    """Verifica se a tabela Databricks ainda não tem dados.
    Retorna True em caso de erro (tabela não existe → força carga completa).
    """
    try:
        js    = await _execute_dbx("SELECT COUNT(*) AS n FROM gold_prod.imais_conversas_diarias")
        rows  = (js.get("result") or {}).get("data_array") or []
        count = int((rows[0][0] if rows else 0) or 0)
        return count == 0
    except Exception:
        return True  # tabela não existe ou erro → assume vazia → full load


async def send_to_databricks(date_str: str) -> int:
    """Envia conversas para Databricks.
    - Primeira execução (tabela vazia): carrega TODO o histórico do SQLite.
    - Execuções seguintes: MERGE apenas das conversas do dia (date_str).
    Retorna qtd de linhas processadas.
    """
    empty = await _table_is_empty()

    if empty:
        print(f"[DBX] Tabela vazia — carga completa do histórico")
        threads  = conv_log.list_threads(DB_PATH)          # sem filtro de data
        all_rows = []
        for t in threads:
            all_rows.extend(conv_log.get_history(t["thread_id"], DB_PATH))  # sem filtro
    else:
        threads  = conv_log.list_threads(DB_PATH, for_date=date_str)
        all_rows = []
        for t in threads:
            all_rows.extend(conv_log.get_history(t["thread_id"], DB_PATH, for_date=date_str))

    if not all_rows:
        print(f"[DBX] Nenhuma conversa em {date_str}")
        return 0

    # MERGE em lotes de 50 — insere apenas registros novos, preserva histórico
    BATCH = 50
    total = 0
    for i in range(0, len(all_rows), BATCH):
        batch  = all_rows[i : i + BATCH]
        values = []
        for r in batch:
            # data_envio: usa a data real da mensagem (full load) ou date_str (diário)
            row_date = (r.get("created_at") or date_str)[:10]  # YYYY-MM-DD
            values.append(
                f"('{_esc(r.get('thread_id'))}', '{_esc(r.get('phone'))}', "
                f"'{_esc(r.get('cnpj'))}', '{_esc(r.get('canal', 'whatsapp'))}', "
                f"'{_esc(r.get('user_name'))}', '{_esc(r.get('question'))}', "
                f"'{_esc(r.get('answer'))}', '{_esc(r.get('metric'))}', "
                f"'{_esc(r.get('insight'))}', '{_esc(r.get('insight_feedback'))}', "
                f"CAST('{_esc(r.get('created_at', ''))}' AS TIMESTAMP), "
                f"CAST('{row_date}' AS DATE))"
            )
        sql = f"""
                MERGE INTO gold_prod.imais_conversas_diarias AS target
                USING (
                SELECT thread_id, phone, cnpj, canal, user_name, question, answer,
                        metric, insight, insight_feedback, created_at, data_envio
                FROM VALUES {', '.join(values)}
                AS src(thread_id, phone, cnpj, canal, user_name, question, answer,
                        metric, insight, insight_feedback, created_at, data_envio)
                ) AS source
                ON  target.thread_id  = source.thread_id
                AND target.created_at = source.created_at
                WHEN MATCHED AND source.insight_feedback != ''
                    AND (target.insight_feedback IS NULL OR target.insight_feedback = '') THEN
                UPDATE SET target.insight_feedback = source.insight_feedback
                WHEN NOT MATCHED THEN
                INSERT (thread_id, phone, cnpj, canal, user_name, question, answer,
                        metric, insight, insight_feedback, created_at, data_envio)
                VALUES (source.thread_id, source.phone, source.cnpj, source.canal,
                        source.user_name, source.question, source.answer,
                        source.metric, source.insight, source.insight_feedback,
                        source.created_at, source.data_envio)
                """
        await _execute_dbx(sql)
        total += len(batch)
        print(f"[DBX] Lote {i // BATCH + 1}: {len(batch)} linhas processadas")

    print(f"[DBX] Total: {total} conversas enviadas ao Databricks ✓")
    return total


# ── Email ─────────────────────────────────────────────────────────────────────

def _build_html(date_str: str, summary: dict, dbx_count: int) -> str:
    dbx_status = (
        f"<p style='color:green'>✅ <b>{dbx_count} conversas enviadas ao Databricks com sucesso.</b></p>"
        if dbx_count > 0
        else "<p style='color:orange'>⚠️ Envio ao Databricks não realizado (verifique os logs).</p>"
    )

    threads_rows = "".join(
        f"<tr><td>{t['thread_id']}</td><td>{t['phone']}</td>"
        f"<td>{t.get('user_name') or '—'}</td>"
        f"<td>{t['cnpj'] or '—'}</td><td>{t['total_msgs']}</td>"
        f"<td>{t['first_at']}</td><td>{t['last_at']}</td></tr>"
        for t in summary["threads"]
    )

    uname_map = {t["thread_id"]: t.get("user_name") or "—" for t in summary["threads"]}
    detail_html = ""
    for tid, msgs in summary["details"].items():
        uname = uname_map.get(tid, "—")
        for m in msgs:
            q = (m.get("question") or "").replace("<", "&lt;")
            a = (m.get("answer") or "").replace("<", "&lt;").replace("\n", "<br>")
            detail_html += (
                f"<tr><td>{tid}</td><td>{uname}</td>"
                f"<td>{m.get('created_at', '')}</td><td>{q}</td><td>{a}</td></tr>"
            )

    return f"""
    <h2>📊 Relatório Diário iMAIS — {date_str}</h2>
    {dbx_status}
    <p><b>Threads ativas:</b> {len(summary['threads'])} &nbsp;|&nbsp;
    <b>Total de mensagens:</b> {summary['total_msgs']}</p>
    <h3>Resumo por thread</h3>
    <table border='1' cellpadding='6' style='border-collapse:collapse;font-size:13px'>
    <tr style='background:#f4f4f4'>
      <th>Thread</th><th>Telefone</th><th>Usuário</th><th>CNPJ</th>
      <th>Msgs</th><th>Início</th><th>Último</th>
    </tr>
    {threads_rows or "<tr><td colspan='7'>Nenhuma thread</td></tr>"}
    </table>
    <h3>Conversas do dia</h3>
    <table border='1' cellpadding='6' style='border-collapse:collapse;font-size:12px;width:100%'>
    <tr style='background:#f4f4f4'>
      <th>Thread</th><th>Usuário</th><th>Hora</th><th>Pergunta</th><th>Resposta</th>
    </tr>
    {detail_html or "<tr><td colspan='5'>Nenhuma mensagem</td></tr>"}
    </table>
    <p style='color:gray;font-size:11px'>Gerado automaticamente. Por favor, não responda.</p>
    """


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    today = date.today().isoformat()
    print(f"[DAILY] Iniciando relatório de {today}")

    # (guard de reenvio desativado para testes)

    # 1. Envia conversas para Databricks
    dbx_count = 0
    try:
        dbx_count = await send_to_databricks(today)
    except Exception as e:
        print(f"[DBX] ERRO: {e}")

    # 2. Monta resumo do dia
    summary = conv_log.daily_summary(today, DB_PATH)

    if not summary["threads"] and dbx_count == 0:
        print(f"[DAILY] Nenhuma conversa hoje — relatório não enviado")
        conv_log.mark_report_sent(today, DB_PATH)
        return

    # 3. Envia email
    html_body = _build_html(today, summary, dbx_count)
    try:
        from shared.email_client import _post_email_all
        await _post_email_all(f"[iMAIS] Relatório diário — {today}", html_body)
        print(f"[EMAIL] Relatório enviado ✓ ({len(summary['threads'])} threads)")
    except Exception as e:
        print(f"[EMAIL] ERRO: {e}")

    # 4. Marca como enviado (pode falhar se banco estiver somente-leitura)
    try:
        conv_log.mark_report_sent(today, DB_PATH)
    except Exception as e:
        print(f"[DAILY] AVISO: não foi possível marcar relatório como enviado: {e}")
    print(f"[DAILY] Concluído ✓")


if __name__ == "__main__":
    asyncio.run(main())
