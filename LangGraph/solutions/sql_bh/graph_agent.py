from __future__ import annotations

import os
from datetime import date
from typing import Annotated, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from solutions.sql_analytics.catalog_loader import load_schema_catalog
from shared.db_client import cleanup_sql, parse_cols, parse_rows, run_query
from solutions.sql_analytics.schema_tools import validate_sql_against_schema
from solutions.sql_bh.sql_generator import ExtractedParams, SqlGenerator

load_dotenv()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_CATALOG_PATH = os.path.join(_THIS_DIR, "schema_catalog.json")

_schema = load_schema_catalog(SCHEMA_CATALOG_PATH)
_generator = SqlGenerator(schema=_schema)


# ── Estado do grafo ────────────────────────────────────────────────────────────

class State(TypedDict, total=False):
    messages:   Annotated[list, add_messages]
    question:   str

    # curto-circuito (sem query)
    direct_reply: Optional[str]

    # extração de parâmetros
    extracted_params: Optional[dict]

    # geração de SQL
    sql:              Optional[str]
    sql_attempts:     int
    sql_error:        Optional[str]
    supervisor_retry: bool

    # resultado da query
    columns: list
    rows:    list

    # nota do sql gen
    sql_note: Optional[str]

    # saída final
    answer: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _last_user_message(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict):
            if (m.get("role") or "").lower() in ("user", "human"):
                return str(m.get("content") or "").strip()
        else:
            role = (getattr(m, "type", None) or getattr(m, "role", None) or "").lower()
            if role in ("human", "user"):
                return str(getattr(m, "content", "") or "").strip()
    return ""


def _log(node: str, **kv):
    items = " | ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[BH-{node}] {items}")


# ── Nós ────────────────────────────────────────────────────────────────────────

async def preprocess_node(state: State) -> dict:
    """Classifica a mensagem."""
    messages = state.get("messages") or []
    question = _last_user_message(messages)

    _log("preprocess", question=question[:80])

    classify = await _generator.classify(question)
    _log("preprocess", intent=classify.intent)

    if classify.intent != "data_query":
        return {
            "question":     question,
            "direct_reply": classify.direct_reply or "Como posso te ajudar?",
        }

    return {
        "question":         question,
        "direct_reply":     None,
        "extracted_params": None,
        "sql":              None,
        "sql_attempts":     0,
        "sql_error":        None,
        "supervisor_retry": False,
        "columns":          [],
        "rows":             [],
    }


async def extract_node(state: State) -> dict:
    """Extrai parâmetros estruturados da pergunta."""
    question = state.get("question") or ""
    today = date.today().isoformat()

    _log("extract", question=question[:80])

    try:
        params = await _generator.extract_params(question=question, today=today)
        params_dict = params.model_dump()
        _log("extract",
             metric=params.metric,
             grain=params.grain,
             region=params.region_filter,
             subcategory=params.subcategory_filter,
             summary=params.summary[:60])
        return {"extracted_params": params_dict}
    except Exception as e:
        _log("extract", result=f"ERRO: {e}")
        return {"extracted_params": None}


async def sql_gen_node(state: State) -> dict:
    """Gera SQL via LLM e valida."""
    question       = state.get("question") or ""
    attempts       = state.get("sql_attempts") or 0
    error_feedback = state.get("sql_error")
    today          = date.today().isoformat()

    params_dict = state.get("extracted_params")
    extracted = ExtractedParams(**params_dict) if params_dict else None

    attempts += 1
    _log("sql_gen", attempt=attempts, error_feedback=error_feedback or "nenhum")

    try:
        result = await _generator.generate_sql(
            question=question,
            today=today,
            error_feedback=error_feedback,
            extracted_params=extracted,
        )
        sql = cleanup_sql(result.sql or "")
    except Exception as e:
        _log("sql_gen", result=f"ERRO geração: {e}")
        return {
            "sql":          None,
            "sql_attempts": attempts,
            "sql_error":    f"Erro ao gerar SQL: {e}",
        }

    _log("sql_gen", sql=sql[:120])

    ok, reason = validate_sql_against_schema(sql, _schema)
    if not ok:
        _log("sql_gen", result=f"VALIDAÇÃO FALHOU: {reason}")
        return {
            "sql":          None,
            "sql_attempts": attempts,
            "sql_error":    f"SQL inválida ({reason}). Gere uma SQL diferente usando apenas a tabela gold_prod.sellout_supermercado_bh.",
        }

    _log("sql_gen", result="SQL válida")
    return {
        "sql":          sql,
        "sql_note":     result.note or None,
        "sql_attempts": attempts,
        "sql_error":    None,
        "supervisor_retry": False,
    }


async def execute_node(state: State) -> dict:
    """Executa a SQL no Databricks."""
    sql = state.get("sql") or ""
    _log("execute", sql=sql[:120])

    try:
        js      = await run_query(sql)
        columns = parse_cols(js)
        rows    = parse_rows(js)
        _log("execute", result=f"{len(rows)} rows, cols={columns}")
        return {"columns": columns, "rows": rows}
    except Exception as e:
        _log("execute", result=f"ERRO: {e}")
        return {
            "columns": [],
            "rows":    [],
            "sql_error": f"Erro ao executar SQL no Databricks: {e}",
            "supervisor_retry": False,
        }


def _rows_are_empty(rows: list) -> bool:
    if not rows:
        return True
    for row in rows:
        for val in (row or []):
            if val is not None and str(val).strip() not in ("", "0", "0.0", "0.00"):
                return False
    return True


async def supervisor_node(state: State) -> dict:
    """Verifica resultado e pede retry se necessário."""
    rows     = state.get("rows") or []
    sql_err  = state.get("sql_error")
    attempts = state.get("sql_attempts") or 0

    _log("supervisor", rows=len(rows), sql_error=bool(sql_err), attempts=attempts)

    if sql_err and attempts < 3:
        _log("supervisor", result="RETRY (erro de execução)")
        return {
            "supervisor_retry": True,
            "sql": None,
            "sql_error": sql_err,
        }

    if _rows_are_empty(rows) and attempts < 3:
        _log("supervisor", result="RETRY (vazio)")
        return {
            "supervisor_retry": True,
            "sql": None,
            "sql_error": "A query retornou zero resultados. Verifique os filtros, tente filtros mais abrangentes ou use translate() para lidar com acentos.",
        }

    if len(rows) > 200:
        _log("supervisor", result=f"TRUNCADO ({len(rows)} → 20)")
        return {
            "supervisor_retry": False,
            "rows": rows[:20],
            "answer": None,
        }

    _log("supervisor", result="OK")
    return {"supervisor_retry": False}


async def write_node(state: State) -> dict:
    """Formata a resposta final."""
    direct = (state.get("direct_reply") or "").strip()
    if direct:
        _log("write", source="direct_reply")
        return {"answer": direct}

    question = state.get("question") or ""
    columns  = state.get("columns") or []
    rows     = state.get("rows") or []
    today    = date.today().isoformat()

    _log("write", source="LLM writer", rows=len(rows), cols=len(columns))
    answer = await _generator.write_answer(
        question=question, columns=columns, rows=rows, today=today,
    )
    _log("write", answer_len=len(answer), answer_preview=answer[:100])
    return {"answer": answer}


# ── Roteadores ─────────────────────────────────────────────────────────────────

def route_preprocess(state: State) -> str:
    if (state.get("direct_reply") or "").strip():
        return "write"
    return "extract"


def route_sql_gen(state: State) -> str:
    if state.get("sql"):
        return "execute"
    if (state.get("sql_attempts") or 0) < 3:
        return "sql_gen"
    return "write"


def route_supervisor(state: State) -> str:
    if state.get("supervisor_retry") and (state.get("sql_attempts") or 0) < 3:
        return "sql_gen"
    return "write"


# ── Build ──────────────────────────────────────────────────────────────────────

def build_agent():
    g = StateGraph(State)

    g.add_node("preprocess",  preprocess_node)
    g.add_node("extract",     extract_node)
    g.add_node("sql_gen",     sql_gen_node)
    g.add_node("execute",     execute_node)
    g.add_node("supervisor",  supervisor_node)
    g.add_node("write",       write_node)

    g.add_edge(START, "preprocess")
    g.add_conditional_edges("preprocess", route_preprocess, {"write": "write", "extract": "extract"})
    g.add_edge("extract", "sql_gen")
    g.add_conditional_edges("sql_gen", route_sql_gen, {"execute": "execute", "sql_gen": "sql_gen", "write": "write"})
    g.add_edge("execute", "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor, {"sql_gen": "sql_gen", "write": "write"})
    g.add_edge("write", END)

    return g.compile(checkpointer=MemorySaver())
