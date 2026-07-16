from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import httpx

from shared.db_client import DBX_HOST, DBX_HEADERS, DBX_WAREHOUSE_ID, parse_rows

# ── Mensagem padrão para cliente sem estoque ─────────────────────────────────

ESTOQUE_NAO_DISPONIVEL = (
    "Você não tem a funcionalidade de estoque ativa no momento. "
    "Entre em contato com nosso suporte para mais informações:\n\n"
    "- WhatsApp: (34) 99912-7261\n"
    "- Capitais e regiões metropolitanas: 3003-1266\n"
    "- Outras regiões: 0800-729-5217"
)

# Cache de disponibilidade por CNPJ (evita query Databricks a cada mensagem)
_estoque_avail_cache: dict[str, tuple[bool, float]] = {}
_AVAIL_CACHE_TTL = 3600  # 1 hora


# ── Execução crua (apenas o módulo estoque pode escrever) ─────────────────────

async def _execute_statement(sql: str, timeout: float = 60.0) -> dict:
    """
    Executa qualquer statement no Databricks (SELECT ou UPDATE).
    Uso restrito a este módulo — não recebe SQL gerado por LLM.
    """
    if not (DBX_HOST and DBX_HEADERS and DBX_WAREHOUSE_ID):
        raise RuntimeError("Variáveis Databricks não configuradas.")

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

    raise TimeoutError("Timeout esperando o Databricks finalizar.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escapa apóstrofos para evitar SQL injection em strings."""
    return (s or "").replace("'", "''")


def _normalize_term(term: str) -> str:
    """Normaliza um termo de busca: remove acentos, baixa caixa, escapa."""
    t = (term or "").lower().strip()
    t = re.sub(r"[áàâãä]", "a", t)
    t = re.sub(r"[éèêë]", "e", t)
    t = re.sub(r"[íìîï]", "i", t)
    t = re.sub(r"[óòôõö]", "o", t)
    t = re.sub(r"[úùûü]", "u", t)
    t = t.replace("ç", "c")
    return _esc(t)


# ── API pública do módulo ─────────────────────────────────────────────────────

async def get_srk_cli(cnpj: str) -> Optional[int]:
    """Busca o SRK_CLI a partir do CNPJ via dim_cli."""
    cnpj_clean = re.sub(r"\D", "", cnpj or "")
    if not cnpj_clean:
        return None

    sql = (
        f"SELECT SRK_CLI FROM imaiscatalog.gold_prod.dim_cli "
        f"WHERE lpad(cast(CNPJ_CPF as string), 14, '0') = '{_esc(cnpj_clean)}' "
        f"LIMIT 1"
    )
    js = await _execute_statement(sql)
    rows = parse_rows(js)
    if not rows or not rows[0]:
        return None
    try:
        return int(rows[0][0])
    except (ValueError, TypeError):
        return None


async def search_products(
    srk_cli: int,
    term: str,
    *,
    exclude_zero_stock: bool = False,
    limit: int = 100,
) -> list[dict]:
    """Busca produtos do estoque pelo nome (LIKE com translate p/ acentos).
    Considera apenas a snapshot mais recente (max DATA_RELATORIO) por CODIGO.
    Se exclude_zero_stock=True, ignora produtos com QNT_ESTOQUE <= 0.

    Retorna até `limit` resultados (default 100). A paginação para exibição
    é feita no router/state-machine.
    """
    term_norm = _normalize_term(term)
    if len(term_norm) < 2:
        return []

    translate_expr = (
        "translate(lower(DESCRICAO), "
        "'áàâãéèêíìîóòôõúùûüç', 'aaaaeeeiiioooouuuuc')"
    )

    # Se o termo for puramente numérico, também busca por CEAN (starts-with).
    # Isso permite digitar o código GTIN/EAN diretamente.
    digits_only = re.sub(r"\D", "", term or "")
    is_numeric  = bool(digits_only) and digits_only == re.sub(r"\s", "", term or "")
    cean_clause = (
        f"OR cast(CEAN as string) LIKE '{_esc(digits_only)}%' "
        if is_numeric else ""
    )

    where_zero = "AND QNT_ESTOQUE > 0 " if exclude_zero_stock else ""

    # Limpa caracteres especiais (barra, hífen isolado) que podem vir de OCR.
    term_clean = re.sub(r'[^a-z0-9 ]', ' ', term_norm)

    # Remove stop-words, medidas/pesos (ex: "306g", "300g", "500g", "2") e tokens curtos.
    # Medidas são os tokens mais propensos a erro de leitura — removê-los torna a busca
    # mais robusta sem perder especificidade (o nome do produto já é suficiente).
    _STOP = {"de", "do", "da", "e", "com", "kg", "un", "g", "cx"}
    _MEASURE_RE = re.compile(r'^\d+[a-z]*$')  # "306g", "300", "1kg" etc.
    tokens = [
        t for t in term_clean.split()
        if len(t) >= 2 and t not in _STOP and not _MEASURE_RE.match(t)
    ]
    if not tokens:
        tokens = [term_clean.split()[0]] if term_clean.strip() else [term_norm]

    def _build_sql(tok_list: list) -> str:
        nc = " AND ".join(f"{translate_expr} LIKE '%{t}%'" for t in tok_list)
        return (
            f"WITH latest AS ("
            f" SELECT CODIGO, CEAN, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO,"
            f"  ROW_NUMBER() OVER (PARTITION BY CODIGO ORDER BY DATA_RELATORIO DESC) AS rn"
            f" FROM imaiscatalog.silver_prod.estoque_quantum_poc"
            f" WHERE SRK_CLI = {int(srk_cli)}"
            f") "
            f"SELECT CODIGO, CEAN, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO "
            f"FROM latest "
            f"WHERE rn = 1 "
            f"AND (({nc}) {cean_clause}) "
            f"{where_zero}"
            f"ORDER BY DESCRICAO "
            f"LIMIT {int(limit)}"
        )

    js = await _execute_statement(_build_sql(tokens))
    rows = parse_rows(js)

    # Fallback: se sem resultado e token extra provavelmente errado, tenta só nome base.
    # Ex: "alface lisa tropa" → tenta "alface lisa" se "tropa" não bateu em nada.
    if not rows and len(tokens) > 2:
        js = await _execute_statement(_build_sql(tokens[:2]))
        rows = parse_rows(js)

    out: list[dict] = []
    for r in (rows or []):
        if not r:
            continue
        try:
            qnt = float(r[4]) if len(r) > 4 and r[4] is not None else 0.0
        except (ValueError, TypeError):
            qnt = 0.0
        out.append({
            "codigo":         str(r[0]) if r[0] is not None else "",
            "cean":           str(r[1]) if len(r) > 1 and r[1] is not None else "",
            "descricao":      str(r[2]) if len(r) > 2 and r[2] is not None else "",
            "unidade":        str(r[3]).strip().upper() if len(r) > 3 and r[3] else "UN",
            "qnt_estoque":    qnt,
            "data_relatorio": str(r[5]) if len(r) > 5 and r[5] is not None else "",
        })
    return out


async def get_current_stock(srk_cli: int, codigo: str) -> Optional[dict]:
    """Re-busca a snapshot mais recente de um produto específico (max DATA_RELATORIO).
    Retorna None se o produto não existir.
    """
    sql = (
        f"SELECT CODIGO, CEAN, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO "
        f"FROM imaiscatalog.silver_prod.estoque_quantum_poc "
        f"WHERE SRK_CLI = {int(srk_cli)} AND CODIGO = '{_esc(str(codigo))}' "
        f"ORDER BY DATA_RELATORIO DESC "
        f"LIMIT 1"
    )
    js = await _execute_statement(sql)
    rows = parse_rows(js)
    if not rows or not rows[0]:
        return None
    r = rows[0]
    try:
        qnt = float(r[4]) if len(r) > 4 and r[4] is not None else 0.0
    except (ValueError, TypeError):
        qnt = 0.0
    return {
        "codigo":         str(r[0]) if r[0] is not None else "",
        "cean":           str(r[1]) if len(r) > 1 and r[1] is not None else "",
        "descricao":      str(r[2]) if len(r) > 2 and r[2] is not None else "",
        "unidade":        str(r[3]).strip().upper() if len(r) > 3 and r[3] else "UN",
        "qnt_estoque":    qnt,
        "data_relatorio": str(r[5]) if len(r) > 5 and r[5] is not None else "",
    }


async def get_product_by_code(srk_cli: int, code: str) -> Optional[dict]:
    """Busca um produto por correspondência EXATA de CEAN ou CODIGO (snapshot mais recente).

    A coluna 'Barra' de relatórios de PDF (ex: Movimentação Interna de Produtos) casa
    com o CEAN do estoque — inclusive para itens a granel, que usam um CEAN curto
    (ex: '3195') em vez do código de barras completo. Compara também numericamente
    (sem zeros à esquerda) porque o CEAN pode estar armazenado como número no banco,
    o que perderia um zero à esquerda como em '0436'.
    """
    code_clean = (code or "").strip()
    if not code_clean:
        return None

    numeric_clause = ""
    if code_clean.isdigit():
        numeric_clause = f" OR try_cast(CEAN as bigint) = {int(code_clean)}"

    sql = (
        f"WITH latest AS ("
        f" SELECT CODIGO, CEAN, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO,"
        f"  ROW_NUMBER() OVER (PARTITION BY CODIGO ORDER BY DATA_RELATORIO DESC) AS rn"
        f" FROM imaiscatalog.silver_prod.estoque_quantum_poc"
        f" WHERE SRK_CLI = {int(srk_cli)}"
        f") "
        f"SELECT CODIGO, CEAN, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO "
        f"FROM latest WHERE rn = 1 "
        f"AND (cast(CEAN as string) = '{_esc(code_clean)}' OR CODIGO = '{_esc(code_clean)}'{numeric_clause}) "
        f"LIMIT 1"
    )
    js = await _execute_statement(sql)
    rows = parse_rows(js)
    if not rows or not rows[0]:
        return None
    r = rows[0]
    try:
        qnt = float(r[4]) if len(r) > 4 and r[4] is not None else 0.0
    except (ValueError, TypeError):
        qnt = 0.0
    return {
        "codigo":         str(r[0]) if r[0] is not None else "",
        "cean":           str(r[1]) if len(r) > 1 and r[1] is not None else "",
        "descricao":      str(r[2]) if len(r) > 2 and r[2] is not None else "",
        "unidade":        str(r[3]).strip().upper() if len(r) > 3 and r[3] else "UN",
        "qnt_estoque":    qnt,
        "data_relatorio": str(r[5]) if len(r) > 5 and r[5] is not None else "",
    }


async def update_stock(
    srk_cli: int,
    codigo: str,
    delta: float,
    tipo_operacao: str | None = None,
    origem: str = "WhatsApp",
) -> dict:
    """
    Registra uma movimentação inserindo uma nova linha no histórico.
    QNT_ESTOQUE = saldo atual + delta (saldo acumulado atualizado).
    tipo_operacao: se informado, usado direto; senão deriva pelo sinal do delta
    ('LANCAMENTO' se delta > 0, 'BAIXA' se delta < 0).
    origem: 'WhatsApp' ou 'Portal'.
    """
    from datetime import datetime as _datetime

    current = await get_current_stock(srk_cli, codigo)
    if current is None:
        raise RuntimeError(f"Produto {codigo} não encontrado no estoque.")

    tipo    = tipo_operacao or ("LANCAMENTO" if delta > 0 else "BAIXA")
    now     = _datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_qnt = current["qnt_estoque"] + float(delta)

    sql = (
        f"INSERT INTO imaiscatalog.silver_prod.estoque_quantum_poc "
        f"(SRK_CLI, CEAN, CODIGO, DESCRICAO, UNIDADE_MEDIDA, QNT_ESTOQUE, DATA_RELATORIO, TIPO_OPERACAO, ORIGEM) "
        f"VALUES ("
        f"  {int(srk_cli)}, "
        f"  '{_esc(current['cean'])}', "
        f"  '{_esc(str(codigo))}', "
        f"  '{_esc(current['descricao'])}', "
        f"  '{_esc(current['unidade'])}', "
        f"  {new_qnt}, "
        f"  cast('{now}' AS TIMESTAMP), "
        f"  '{_esc(tipo)}', "
        f"  '{_esc(origem)}'"
        f")"
    )
    return await _execute_statement(sql)


async def check_client_has_estoque(cnpj: str) -> bool:
    """Verifica se o cliente tem dados na tabela de estoque.
    Resultado cacheado por 1h para não bater no Databricks a cada mensagem.
    """
    cnpj_clean = re.sub(r"\D", "", cnpj or "")
    if not cnpj_clean:
        return False

    cached = _estoque_avail_cache.get(cnpj_clean)
    if cached and time.time() < cached[1]:
        return cached[0]

    sql = (
        f"SELECT 1 FROM imaiscatalog.silver_prod.estoque_quantum_poc e "
        f"JOIN imaiscatalog.gold_prod.dim_cli c ON c.SRK_CLI = e.SRK_CLI "
        f"WHERE lpad(cast(c.CNPJ_CPF as string), 14, '0') = '{_esc(cnpj_clean)}' "
        f"LIMIT 1"
    )
    try:
        js = await _execute_statement(sql)
        result = len(parse_rows(js)) > 0
    except Exception as e:
        print(f"[check_estoque_avail] ERRO: {e}")
        result = False

    _estoque_avail_cache[cnpj_clean] = (result, time.time() + _AVAIL_CACHE_TTL)
    print(f"[check_estoque_avail] cnpj={cnpj_clean} → {result}")
    return result
