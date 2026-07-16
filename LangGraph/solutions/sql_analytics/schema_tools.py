from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


SAFE_PREFIX = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE)\b", re.IGNORECASE)
FORBIDDEN_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|CREATE)\b", re.IGNORECASE)

TABLE_REF_RE = re.compile(r"\b(from|join)\s+([a-zA-Z0-9_.]+)", re.IGNORECASE)

CTE_NAME_RE = re.compile(r"\bwith\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", re.IGNORECASE)
CTE_NEXT_RE = re.compile(r"\)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", re.IGNORECASE)


def load_schema_catalog(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("schema_catalog must be a JSON object")
    data.setdefault("tables", [])
    data.setdefault("join_hints", [])
    data.setdefault("metric_glossary", [])
    return data


def cleanup_sql(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r"^\s*```sql\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def extract_cte_names(sql: str) -> Set[str]:
    s = sql or ""
    names: Set[str] = set()
    m = CTE_NAME_RE.search(s)
    if not m:
        return names
    names.add(m.group(1).lower())
    for mm in CTE_NEXT_RE.finditer(s):
        names.add(mm.group(1).lower())
    return names


def extract_tables_from_sql(sql: str) -> List[str]:
    out: List[str] = []
    for m in TABLE_REF_RE.finditer(sql or ""):
        t = (m.group(2) or "").strip().rstrip(",").rstrip(";")
        if t:
            out.append(t)
    return out


def build_allowlists(schema: dict) -> Tuple[Set[str], Dict[str, Set[str]]]:
    allowed_tables: Set[str] = set()
    allowed_cols_by_table: Dict[str, Set[str]] = {}

    for t in (schema.get("tables") or []):
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        allowed_tables.add(name.lower())

        cols = t.get("columns") or []
        colset = set()
        if isinstance(cols, list):
            for c in cols:
                c = str(c).strip()
                if c:
                    colset.add(c.lower())
        allowed_cols_by_table[name.lower()] = colset

    return allowed_tables, allowed_cols_by_table


def validate_sql_against_schema(sql: str, schema: dict) -> Tuple[bool, str]:
    cand = cleanup_sql(sql)
    if not cand:
        return False, "empty_sql"
    if FORBIDDEN_SQL.search(cand):
        return False, "forbidden_command"
    if not SAFE_PREFIX.match(cand):
        return False, "only_select_with_show_describe"

    allowed_tables, _ = build_allowlists(schema)
    ctes = extract_cte_names(cand)

    tables = extract_tables_from_sql(cand)
    for t in tables:
        tl = t.lower()
        if tl in ctes:
            continue
        if allowed_tables and tl not in allowed_tables:
            return False, f"table_not_allowed:{t}"

    return True, "ok"


def summarize_schema_for_prompt(schema: dict, max_tables: int = 30, max_cols: int = 50) -> str:
    lines: List[str] = []
    tables = schema.get("tables") or []
    for t in tables[:max_tables]:
        name = (t.get("name") or "").strip()
        desc = (t.get("description") or "").strip()
        grain = (t.get("grain") or "").strip()
        time_cols = t.get("time_columns") or {}

        cols = t.get("columns") or []
        cols_txt = ", ".join([str(c) for c in cols[:max_cols]])

        meta = []
        if grain:
            meta.append(f"grain={grain}")
        if isinstance(time_cols, dict) and time_cols:
            meta.append(f"time_cols={time_cols}")

        meta_txt = (" | " + ", ".join(meta)) if meta else ""
        lines.append(f"- {name}{meta_txt}: cols=[{cols_txt}]")
        if desc:
            lines.append(f"  desc: {desc}")

    joins = schema.get("join_hints") or []
    if joins:
        lines.append("JOIN_HINTS:")
        for j in joins[:40]:
            lines.append(f"- {j}")

    glossary = schema.get("metric_glossary") or []
    if glossary:
        lines.append("METRIC_GLOSSARY:")
        for g in glossary[:40]:
            if isinstance(g, dict):
                term = g.get("term")
                maps_to = g.get("maps_to")
                lines.append(f"- {term}: {maps_to}")

    return "\n".join(lines).strip()