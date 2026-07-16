from __future__ import annotations

import asyncio
import os
import re
from typing import Any, List

import httpx
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_SERVER_HOSTNAME = (os.getenv("DATABRICKS_SERVER_HOSTNAME") or "").strip()
DATABRICKS_HTTP_PATH       = (os.getenv("DATABRICKS_HTTP_PATH") or "").strip()
DATABRICKS_TOKEN           = (os.getenv("DATABRICKS_TOKEN") or "").strip()

TABLE_USERS = "imaiscatalog.gold_prod.whatsapp_user_permissions"

SAFE_PREFIX  = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE)\b", re.IGNORECASE)
FORBIDDEN_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|CREATE)\b", re.IGNORECASE)


def _dbx_host() -> str:
    host = DATABRICKS_SERVER_HOSTNAME
    if not host:
        return ""
    return host if host.startswith("http") else f"https://{host.rstrip('/')}"


def _warehouse_id() -> str:
    m = re.search(r"/warehouses/([^/]+)", DATABRICKS_HTTP_PATH or "")
    return m.group(1) if m else ""


DBX_HOST         = _dbx_host()
DBX_WAREHOUSE_ID = _warehouse_id()
DBX_HEADERS      = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}


def cleanup_sql(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r"^\s*```sql\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def normalize_phone(p: str) -> str:
    return re.sub(r"\D", "", p or "")


def _phone_norm_sql(col: str) -> str:
    """Expressão SQL que normaliza whatsapp_contact (DDDnumero) para 55DDDnumero.
    - 10 dígitos (DDD + 8): insere '9' após o DDD  → 55 + DDD + 9 + 8 dígitos
    - 11 dígitos (DDD + 9): já tem o 9              → 55 + 11 dígitos
    """
    c = f"cast({col} as string)"
    return (
        f"CASE "
        f"WHEN length({c}) = 10 "
        f"  THEN concat('55', substring({c}, 1, 2), '9', substring({c}, 3, 8)) "
        f"WHEN length({c}) = 11 "
        f"  THEN concat('55', {c}) "
        f"ELSE {c} END"
    )


def _cnpj14(cnpj: str) -> str:
    c = re.sub(r"\D", "", cnpj or "")
    return c.zfill(14) if c else ""


def parse_rows(js: dict) -> List[List[Any]]:
    return (((js.get("result") or {}).get("data_array")) or [])


def parse_cols(js: dict) -> List[str]:
    cols = (((js.get("manifest") or {}).get("schema") or {}).get("columns")) or []
    if cols:
        return [str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("name")]
    cols = (((js.get("result") or {}).get("schema") or {}).get("columns")) or []
    return [str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("name")]


async def run_query(sql: str, timeout: float = 60.0) -> dict:
    sql = cleanup_sql(sql)

    if FORBIDDEN_SQL.search(sql or ""):
        raise ValueError("Consulta bloqueada: comando não permitido.")
    if not SAFE_PREFIX.match(sql or ""):
        raise ValueError("Consulta bloqueada: apenas SELECT/WITH/SHOW/DESCRIBE.")
    if not (DBX_HOST and DATABRICKS_TOKEN and DBX_WAREHOUSE_ID):
        raise RuntimeError("Variáveis de ambiente Databricks não configuradas.")

    payload = {
        "statement": sql,
        "warehouse_id": DBX_WAREHOUSE_ID,
        "result_disposition": "INLINE",
        "format": "JSON_ARRAY",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{DBX_HOST}/api/2.0/sql/statements", headers=DBX_HEADERS, json=payload)
        r.raise_for_status()
        sid = (r.json() or {}).get("statement_id")
        if not sid:
            raise RuntimeError("Sem statement_id no retorno do Databricks.")

        for _ in range(120):
            pr = await client.get(f"{DBX_HOST}/api/2.0/sql/statements/{sid}", headers=DBX_HEADERS)
            pr.raise_for_status()
            js = pr.json() or {}
            state = ((js.get("status") or {}).get("state") or "").upper()

            if state == "SUCCEEDED":
                return js
            if state in ("FAILED", "CANCELED", "CANCELLED"):
                err = ((js.get("status") or {}).get("error") or {}).get("message") or state
                raise RuntimeError(f"Databricks SQL {state}: {err}")

            await asyncio.sleep(0.5)

    raise TimeoutError("Timeout esperando o Databricks finalizar a consulta.")


async def get_user_name_for_phone(phone: str, cnpj: str = "") -> str | None:
    """Retorna o user_name do usuário para o telefone+cnpj selecionado."""
    phone_norm = normalize_phone(phone)
    cnpj_norm  = re.sub(r"\D", "", cnpj or "")
    phone_expr = _phone_norm_sql("whatsapp_contact")
    where = f"{phone_expr} = '{phone_norm}'"
    if cnpj_norm:
        where += f" AND cnpj = '{cnpj_norm}'"
    sql = f"SELECT user_name FROM {TABLE_USERS} WHERE {where} LIMIT 1"
    try:
        js = await run_query(sql)
        rows = parse_rows(js)
        name = str((rows or [[None]])[0][0] or "").strip()
        print(f"[USER NAME] phone={phone_norm} cnpj={cnpj_norm or '*'} → '{name or 'None'}'")
        return name if name else None
    except Exception as e:
        print(f"[USER NAME] ERRO: {e}")
        return None


async def get_nome_fantasia_for_cnpj(cnpj: str) -> str | None:
    """Retorna o nome fantasia (razão social) de um CNPJ via nova_mvp_vendas."""
    cnpj_norm = _cnpj14(cnpj)
    if not cnpj_norm:
        return None
    sql = (
        f"SELECT MAX(RAZAO_SOCIAL) FROM imaiscatalog.gold_prod.nova_mvp_vendas "
        f"WHERE CNPJ = '{cnpj_norm}'"
    )
    try:
        js = await run_query(sql)
        rows = parse_rows(js)
        name = str((rows or [[None]])[0][0] or "").strip()
        return name if name else None
    except Exception as e:
        print(f"[FANTASIA] ERRO cnpj={cnpj_norm}: {e}")
        return None


async def get_cnpj_for_phone(phone: str) -> str:
    phone_norm = normalize_phone(phone)
    phone_expr = _phone_norm_sql("whatsapp_contact")
    sql = f"SELECT cnpj FROM {TABLE_USERS} WHERE {phone_expr} = '{phone_norm}' LIMIT 1"
    js = await run_query(sql)
    rows = parse_rows(js)
    cnpj = str((rows or [[""]])[0][0] or "").strip()
    return _cnpj14(cnpj)


async def get_cnpjs_for_phone(phone: str) -> List[dict]:
    """
    Retorna lista de CNPJs associados ao telefone, cada um com razão social.
    Formato: [{"cnpj": "05531927000171", "name": "LOJA EXEMPLO LTDA"}, ...]
    Se não encontrar razão social, usa o CNPJ formatado como nome.
    """
    phone_norm = normalize_phone(phone)
    phone_expr = _phone_norm_sql("whatsapp_contact")

    sql = f"SELECT DISTINCT cnpj FROM {TABLE_USERS} WHERE {phone_expr} = '{phone_norm}'"
    js = await run_query(sql)
    rows = parse_rows(js)
    cnpjs = [_cnpj14(str(r[0])) for r in (rows or []) if r and r[0]]
    cnpjs = [c for c in cnpjs if c]  # remove vazios

    if not cnpjs:
        return []

    # Busca razão social para cada CNPJ em nova_mvp_vendas (tabela com todos os clientes)
    cnpjs_sql = "', '".join(cnpjs)
    sql_names = (
        f"SELECT CNPJ, MAX(RAZAO_SOCIAL) AS RAZAO_SOCIAL "
        f"FROM imaiscatalog.gold_prod.nova_mvp_vendas "
        f"WHERE CNPJ IN ('{cnpjs_sql}') "
        f"GROUP BY CNPJ"
    )

    names: dict[str, str] = {}
    try:
        js2 = await run_query(sql_names)
        rows2 = parse_rows(js2)
        for r in (rows2 or []):
            if r and r[0]:
                cnpj_key = _cnpj14(str(r[0]))
                name_val = str(r[1] or "").strip() if len(r) > 1 else ""
                if cnpj_key:
                    names[cnpj_key] = name_val
    except Exception as e:
        print(f"[get_cnpjs_for_phone] Erro ao buscar razão social: {e}")

    return [
        {"cnpj": c, "name": names.get(c) or _format_cnpj(c)}
        for c in cnpjs
    ]


def _format_cnpj(cnpj: str) -> str:
    """Formata um CNPJ de 14 dígitos como XX.XXX.XXX/XXXX-XX."""
    c = re.sub(r"\D", "", cnpj or "")
    if len(c) != 14:
        return cnpj or ""
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:14]}"
