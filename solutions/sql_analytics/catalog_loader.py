from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

# -----------------------------
# Types
# -----------------------------
@dataclass
class QueryDef:
    id: str
    title: str
    description: str
    aliases: List[str]
    required_params: List[str]
    optional_params: List[str]
    param_types: Dict[str, str]
    allowed_tables: List[str]
    only_select: bool
    sql_template: str
    metric: Optional[str] = None
    grain: Optional[str] = None
    scope: Optional[str] = None
    comparison: Optional[str] = None
    tags: List[str] = None


@dataclass
class Catalog:
    schema: Dict[str, Any]
    queries: Dict[str, QueryDef]

    @staticmethod
    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, BaseModel):
            return x.model_dump()
        return getattr(x, "__dict__", {}) or {}

    def pick_by_alias(self, question: str) -> str | None:
        q = (question or "").lower().strip()

        catalog = (
            getattr(self, "index", None)
            or getattr(self, "queries_by_id", None)
            or getattr(self, "by_id", None)
            or getattr(self, "queries", None)
            or {}
        )

        if isinstance(catalog, list):
            catalog = {it.get("id"): it for it in catalog if isinstance(it, dict) and it.get("id")}

        if not isinstance(catalog, dict):
            catalog = {}

        for qid, qdef in catalog.items():
            qdef = self._to_dict(qdef)

            for a in (qdef.get("aliases") or []):
                a = (a or "").lower().strip()
                if a and a in q:
                    return qid

            title = (qdef.get("title") or "").lower()
            if title and title in q:
                return qid

        return None


# -----------------------------
# Loaders
# -----------------------------
def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schema_catalog(path: str) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError("schema_catalog.json precisa ser um objeto JSON.")
    if "tables" not in data or not isinstance(data["tables"], list):
        raise ValueError("schema_catalog.json precisa ter campo 'tables' (lista).")
    return data


def load_queries_catalog(path: str) -> Dict[str, QueryDef]:
    data = _read_json(path)

    qlist: List[dict] = []
    if isinstance(data, dict) and isinstance(data.get("queries"), list):
        qlist = data["queries"]
    elif isinstance(data, list):
        qlist = data
    elif isinstance(data, dict):
        # fallback: dict de id -> def
        for k, v in data.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("id", k)
                qlist.append(vv)
    else:
        raise ValueError("queries_catalog.json precisa ser lista, ou {queries:[...]}, ou dict id->query.")

    out: Dict[str, QueryDef] = {}
    for q in qlist:
        if not isinstance(q, dict):
            continue

        qid = str(q.get("id") or "").strip()
        if not qid:
            continue

        out[qid] = QueryDef(
            id=qid,
            title=str(q.get("title") or "").strip(),
            description=str(q.get("description") or "").strip(),
            aliases=[str(a).strip() for a in (q.get("aliases") or []) if str(a).strip()],
            required_params=[str(x).strip() for x in (q.get("required_params") or []) if str(x).strip()],
            optional_params=[str(x).strip() for x in (q.get("optional_params") or []) if str(x).strip()],
            param_types={str(k): str(v) for k, v in (q.get("param_types") or {}).items()},
            allowed_tables=[str(x).strip() for x in (q.get("allowed_tables") or []) if str(x).strip()],
            only_select=bool(q.get("only_select", True)),
            sql_template=str(q.get("sql_template") or q.get("sql") or "").strip(),
            metric=(str(q.get("metric")).strip() if q.get("metric") is not None else None),
            grain=(str(q.get("grain")).strip() if q.get("grain") is not None else None),
            scope=(str(q.get("scope")).strip() if q.get("scope") is not None else None),
            comparison=(str(q.get("comparison")).strip() if q.get("comparison") is not None else None),
            tags=[str(t).strip() for t in (q.get("tags") or []) if str(t).strip()],
        )

    if not out:
        raise ValueError("Nenhuma query carregada do queries_catalog.json.")
    return out


def load_catalog(schema_path: str, queries_path: str) -> Catalog:
    schema = load_schema_catalog(schema_path)
    queries = load_queries_catalog(queries_path)
    return Catalog(schema=schema, queries=queries)


# -----------------------------
# Prompt summaries (para LLM)
# -----------------------------
def summarize_schema(schema: Dict[str, Any], max_tables: int = 30, max_cols: int = 40) -> str:
    lines: List[str] = []
    tables = schema.get("tables") or []
    for t in tables[:max_tables]:
        name = str(t.get("name") or "").strip()
        desc = str(t.get("description") or "").strip()
        cols = t.get("columns") or []
        cols_txt = ", ".join([str(c) for c in cols[:max_cols]])
        line = f"- {name}: cols=[{cols_txt}]"
        if desc:
            line += f" | {desc}"
        lines.append(line)
    return "\n".join(lines).strip()


def summarize_queries(queries: Dict[str, QueryDef], max_items: int = 50) -> str:
    lines: List[str] = []
    for q in list(queries.values())[:max_items]:
        alias_txt = ", ".join(q.aliases[:8])
        tags_txt = ", ".join((q.tags or [])[:8])
        req_txt = ", ".join(q.required_params)
        line = f"- {q.id}: {q.title} | aliases=[{alias_txt}] | tags=[{tags_txt}] | required=[{req_txt}]"
        if q.description:
            line += f" | {q.description}"
        if len(line) > 360:
            line = line[:357] + "..."
        lines.append(line)
    if len(queries) > max_items:
        lines.append(f"... (mais {len(queries)-max_items} queries no catalogo)")
    return "\n".join(lines).strip()

# -----------------------------
# Helpers
# -----------------------------
def pick_best_comparison(self, question: str) -> str | None:
    qn = " ".join((question or "").lower().split())

    best_qid = None
    best_score = 0

    pool = getattr(self, "index", None) or getattr(self, "queries", {}) or {}

    for qid, qdef in pool.items():
        if hasattr(qdef, "model_dump"):
            qdef = qdef.model_dump()
        elif hasattr(qdef, "__dict__"):
            qdef = dict(vars(qdef))

        if not isinstance(qdef, dict):
            continue

        if (qdef.get("comparison") or "none") == "none":
            continue

        score = 0

        for a in (qdef.get("aliases") or []):
            a_clean = " ".join(str(a).lower().split())
            if a_clean and a_clean in qn:
                score += 10

        for tag in (qdef.get("tags") or []):
            tag_clean = " ".join(str(tag).lower().split())
            if tag_clean and tag_clean in qn:
                score += 3

        metric = " ".join(str(qdef.get("metric") or "").lower().split())
        if metric and metric in qn:
            score += 4

        if score > best_score:
            best_score = score
            best_qid = qid

    return best_qid

def _qdef_to_dict(self, qdef):
    if qdef is None:
        return {}
    if isinstance(qdef, dict):
        return qdef
    if hasattr(qdef, "model_dump"):
        return qdef.model_dump()
    if hasattr(qdef, "__dict__"):
        return dict(vars(qdef))
    return {}


def pick_best_by_semantics(self, sem) -> str | None:
    # aceita sem como dict ou objeto
    if isinstance(sem, dict):
        sem_metric = str(sem.get("metric") or "").strip().lower()
        sem_entity = str(sem.get("entity") or "").strip().lower()
        sem_grain = str(sem.get("grain") or "").strip().lower()
        sem_intent = str(sem.get("intent") or "").strip().lower()
        sem_ranking = str(sem.get("ranking") or "").strip().lower()
        sem_comparison = bool(sem.get("comparison"))
    else:
        sem_metric = str(getattr(sem, "metric", "") or "").strip().lower()
        sem_entity = str(getattr(sem, "entity", "") or "").strip().lower()
        sem_grain = str(getattr(sem, "grain", "") or "").strip().lower()
        sem_intent = str(getattr(sem, "intent", "") or "").strip().lower()
        sem_ranking = str(getattr(sem, "ranking", "") or "").strip().lower()
        sem_comparison = bool(getattr(sem, "comparison", False))

    # compatibiliza nome de metrica
    metric_aliases = {
        "quantidade": {"quantidade", "quantidade_vendida", "unidades", "volume"},
        "faturamento": {"faturamento", "valor", "valor_vendido", "receita", "total_vendido"},
        "ticket_medio": {"ticket_medio", "ticket", "ticket médio"},
        "transacoes": {"transacoes", "transações", "qtd_transacoes"},
        "preco": {"preco", "preço"},
        "certificado": {"certificado"},
    }

    def metric_match(q_metric: str) -> bool:
        q_metric = (q_metric or "").strip().lower()
        if not sem_metric or sem_metric == "unknown":
            return False
        if sem_metric in metric_aliases:
            return q_metric in metric_aliases[sem_metric]
        return q_metric == sem_metric

    pool = getattr(self, "queries", None) or getattr(self, "index", {}) or {}

    best_qid = None
    best_score = -999

    for qid, raw_qdef in pool.items():
        qdef = self._qdef_to_dict(raw_qdef)
        if not qdef:
            continue

        score = 0

        q_metric = str(qdef.get("metric") or "").strip().lower()
        q_grain = str(qdef.get("grain") or "").strip().lower()
        q_scope = str(qdef.get("scope") or "").strip().lower()
        q_comparison = str(qdef.get("comparison") or "none").strip().lower()
        q_tags = [str(x).strip().lower() for x in (qdef.get("tags") or [])]
        q_title = str(qdef.get("title") or "").strip().lower()
        q_aliases = [str(x).strip().lower() for x in (qdef.get("aliases") or [])]

        # metrica
        if metric_match(q_metric):
            score += 8

        # entidade / grain
        if sem_entity == "produto" and (q_grain == "produto" or "produto" in q_tags):
            score += 6
        elif sem_entity == "categoria" and (q_grain == "categoria" or "categoria" in q_tags):
            score += 6
        elif sem_entity == "geral" and q_scope == "store":
            score += 2

        # granularidade
        if sem_grain and sem_grain != "unknown" and sem_grain == q_grain:
            score += 6

        # comparacao
        if sem_comparison:
            if q_comparison != "none":
                score += 8
            else:
                score -= 8
        else:
            if q_comparison == "none":
                score += 2

        # ranking top/bottom
        if sem_ranking == "top":
            if any(t in q_tags for t in ["top", "maior", "melhor", "ranking"]):
                score += 4
            if "top" in q_title:
                score += 2
        elif sem_ranking == "bottom":
            if any(t in q_tags for t in ["bottom", "menor", "pior", "ranking"]):
                score += 4
            if "bottom" in q_title or "pior" in q_title:
                score += 2

        # intent
        if sem_intent == "comparison" and q_comparison != "none":
            score += 5
        elif sem_intent == "ranking" and any(t in q_tags for t in ["top", "bottom", "ranking"]):
            score += 3
        elif sem_intent == "aggregate" and q_grain in {"total", "mes", "dia"}:
            score += 2

        # reforcos por aliases/titulo
        if sem_grain == "dia" and ("dia" in q_title or any("dia" in a for a in q_aliases)):
            score += 3
        if sem_grain == "mes" and ("mes" in q_title or "mês" in q_title or any(("mes" in a or "mês" in a) for a in q_aliases)):
            score += 3
        if sem_metric == "faturamento" and any(t in q_tags for t in ["faturamento", "valor"]):
            score += 3

        if score > best_score:
            best_score = score
            best_qid = qid

    return best_qid