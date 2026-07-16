# iMAIS — Documentação Técnica

> Chatbot de analytics de varejo para WhatsApp e portal web, baseado em LangGraph + FastAPI + Databricks.

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Stack tecnológica](#2-stack-tecnológica)
3. [Estrutura de diretórios](#3-estrutura-de-diretórios)
4. [Arquitetura do sistema](#4-arquitetura-do-sistema)
5. [Módulo shared](#5-módulo-shared)
6. [Solução sql_analytics](#6-solução-sql_analytics)
7. [Solução estoque](#7-solução-estoque)
8. [Solução sql_bh](#8-solução-sql_bh)
9. [API — endpoints FastAPI](#9-api--endpoints-fastapi)
10. [Banco de dados Databricks](#10-banco-de-dados-databricks)
11. [Persistência local (SQLite)](#11-persistência-local-sqlite)
12. [Variáveis de ambiente](#12-variáveis-de-ambiente)
13. [Scripts de automação](#13-scripts-de-automação)
14. [Fluxo completo de uma mensagem](#14-fluxo-completo-de-uma-mensagem)
15. [Modelos de linguagem e tokens](#15-modelos-de-linguagem-e-tokens)
16. [Melhorias recentes implementadas](#16-melhorias-recentes-implementadas)

---

## 1. Visão geral

O iMAIS é um assistente de inteligência artificial para lojistas do varejo brasileiro. Ele responde perguntas em linguagem natural sobre vendas, produtos, estoque, PDVs e desempenho de mercado, consultando dados reais no Databricks via SQL gerado por LLM.

**Canais de acesso:**
- **WhatsApp** — via webhook Infobip (`POST /whatsapp/webhook`)
- **Portal web** — interface interna com autenticação admin (`POST /portal/message`)
- **Debug local** — endpoint sem Infobip (`POST /debug/message`)

---

## 2. Stack tecnológica

| Camada | Tecnologia |
|---|---|
| Framework web | FastAPI + Uvicorn |
| Orquestração de agentes | LangGraph 1.0.6 |
| Checkpointing de estado | LangGraph `AsyncSqliteSaver` |
| LLM classificação/SQL | OpenAI `gpt-5.4-nano` (via `OPENAI_MODEL`) |
| LLM redação/insights | OpenAI `gpt-4o-mini` (via `OPENAI_MODEL_WRITER`) |
| Data warehouse | Databricks SQL (REST API `/api/2.0/sql/statements`) |
| Persistência local | SQLite (conversas, checkpoints, perfis) |
| Mensageria WhatsApp | Infobip REST API |
| E-mail | API externa (via `shared/email_client.py`) |
| Transcrição de áudio | `shared/audio_transcriber.py` |

---

## 3. Estrutura de diretórios

```
LangGraph/
├── main.py                          # Ponto de entrada FastAPI + roteamento global
├── .env                             # Variáveis de ambiente (não versionado)
├── conversations.db                 # SQLite: conversas, perfis, checkpoints
│
├── shared/                          # Módulos compartilhados entre soluções
│   ├── db_client.py                 # Cliente Databricks + helpers de telefone/CNPJ
│   ├── session_store.py             # Estado em memória por telefone (TTL 30min)
│   ├── profile_store.py             # Memória de longo prazo por CNPJ (SQLite)
│   ├── solution_router.py           # Registro de soluções e comandos de ativação
│   ├── conversation_logger.py       # Logging de conversas no SQLite
│   ├── infobip_client.py            # Envio de mensagens WhatsApp via Infobip
│   ├── email_client.py              # Envio de e-mails
│   └── audio_transcriber.py        # Transcrição de áudios WhatsApp
│
├── solutions/
│   ├── sql_analytics/               # Solução principal: analytics de varejo
│   │   ├── graph_agent.py           # Grafo LangGraph com todos os nós
│   │   ├── sql_generator.py         # Prompts + chamadas OpenAI (classify, extract, SQL, write, insight)
│   │   ├── router.py                # Handler WhatsApp/portal para sql_analytics
│   │   ├── schema_catalog.json      # Schema das tabelas Databricks
│   │   ├── queries_catalog.json     # Catálogo de queries pré-definidas
│   │   ├── catalog_loader.py        # Loader dos JSONs de catálogo
│   │   └── schema_tools.py          # Utilitários de schema
│   │
│   ├── estoque/                     # Módulo de movimentação de estoque
│   │   ├── router.py                # State machine de movimentação
│   │   ├── db_ops.py                # Operações de leitura/escrita no Databricks
│   │   └── quantity_parser.py       # Parser de quantidades em linguagem natural
│   │
│   └── sql_bh/                      # Solução analytics alternativa (BH)
│       ├── graph_agent.py
│       ├── sql_generator.py
│       └── router.py
│
└── scripts/
    └── daily_report.py              # Relatório diário: envia conversas ao Databricks + e-mail
```

---

## 4. Arquitetura do sistema

### Roteamento global (`main.py`)

```
Mensagem recebida (WhatsApp ou Portal)
        │
        ▼
Resolução de CNPJ (via whatsapp_user_permissions)
        │
        ▼
Verificação de solução ativa (session_store)
        │
        ├── "estoque" ativo → solutions/estoque/router.py
        └── "sql_analytics" (padrão) → solutions/sql_analytics/router.py
```

O `solution_router.py` define comandos de ativação (ex: "movimentar estoque") e saída (ex: "sair estoque") que comutam a solução ativa na sessão do usuário.

### Grafo LangGraph (sql_analytics)

```
START
  │
  ▼
preprocess_node      ← resolve CNPJ, classifica intent, carrega perfil
  │
  ├── direct_reply ──────────────────────────────────────────► write_node
  ├── insight_theme ─────────────────────────────────────────► insight_theme_node → END
  ├── relatorio_/reposicao_hortifruti ───────────────────────► hortifruti_report_node → END
  └── data_query ────────────────────────────────────────────► extract_node
                                                                    │
                                                                    ▼
                                                               cert_check_node
                                                                    │
                                                                    ▼
                                                               sql_gen_node
                                                                    │
                                                                    ▼
                                                               execute_node
                                                                    │
                                                                    ▼
                                                               supervisor_node
                                                                    │
                                                          ┌─────────┴──────────┐
                                                       retry                  OK
                                                          │                   │
                                                          └──► sql_gen_node   ▼
                                                                         write_node
                                                                              │
                                                                              ▼
                                                                         insight_node
                                                                              │
                                                                              ▼
                                                                             END
```

---

## 5. Módulo shared

### `db_client.py`

Responsável por toda comunicação com o Databricks e lookups de usuário.

**Funções principais:**

| Função | Descrição |
|---|---|
| `run_query(sql)` | Executa SELECT no Databricks com polling assíncrono. Bloqueia INSERT/UPDATE/DELETE. |
| `get_cnpj_for_phone(phone)` | Retorna o primeiro CNPJ associado ao telefone. |
| `get_cnpjs_for_phone(phone)` | Retorna todos os CNPJs do telefone com razão social. |
| `get_user_name_for_phone(phone, cnpj)` | Retorna o nome do usuário para o par telefone+CNPJ. |
| `get_nome_fantasia_for_cnpj(cnpj)` | Retorna razão social do CNPJ via `nova_mvp_vendas`. |
| `_phone_norm_sql(col)` | Expressão SQL que normaliza `whatsapp_contact` (DDDnúmero) para `55DDDnúmero`. |
| `normalize_phone(p)` | Remove caracteres não-numéricos do telefone. |
| `cleanup_sql(sql)` | Remove blocos de código markdown do SQL gerado pelo LLM. |

**Tabela de usuários:** `imaiscatalog.gold_prod.whatsapp_user_permissions`

Números armazenados no formato `DDDnúmero` (sem `55`). A função `_phone_norm_sql` normaliza em SQL:
- 10 dígitos (DDD+8): insere `9` após o DDD → `55DDDnúmero`
- 11 dígitos (DDD+9): apenas adiciona `55`

---

### `session_store.py`

Estado em memória por telefone com TTL de 30 minutos. **Não persiste entre restarts** (recomendado migrar para Redis em produção).

**Estrutura da sessão:**
```python
{
    "cnpj":                  str,      # CNPJ selecionado
    "active_solution":       str,      # "sql_analytics" | "estoque"
    "pending_options":       list,     # CNPJs aguardando seleção
    "pending_message":       str,      # Mensagem original antes da seleção
    "pending_insight":       dict,     # Insight aguardando feedback (question, answer, insight)
    "pending_insight_theme": bool,     # Aguardando tema para insight livre
    "pending_catalog":       dict,     # Sugestão de catálogo pendente
    "estoque":               dict,     # Estado da state machine de estoque
    "expires_at":            float,    # Timestamp de expiração
}
```

---

### `profile_store.py`

Memória de longo prazo por CNPJ, persistida no SQLite.

**Tabela `user_profiles`:**

| Coluna | Tipo | Descrição |
|---|---|---|
| `cnpj` | TEXT PK | CNPJ do estabelecimento |
| `metric_counts` | TEXT (JSON) | Counter de métricas consultadas |
| `period_counts` | TEXT (JSON) | Counter de períodos usados |
| `last_seen` | TEXT | Data da última consulta |
| `total_queries` | INTEGER | Total de consultas realizadas |
| `notes` | TEXT | Notas livres (reservado) |

**Uso:** após cada resposta bem-sucedida, `update_profile()` incrementa os contadores. No próximo turno, `build_profile_hint()` gera uma dica que é injetada no extrator de parâmetros para desambiguar perguntas vagas.

---

### `conversation_logger.py`

Logging de todas as interações no SQLite. Cada mensagem gera um registro com `thread_id`, `phone`, `cnpj`, `canal`, `user_name`, `question`, `answer`, `metric`, `insight` e `insight_feedback`.

---

### `solution_router.py`

Define o registro de soluções e seus comandos de ativação:

```python
SOLUTION_REGISTRY = {
    "estoque": {
        "activation": ["movimentar estoque", "baixar estoque", "lançar produto", ...]
    }
}
```

Comandos de saída retornam para `sql_analytics` (solução padrão).

---

## 6. Solução sql_analytics

### `sql_generator.py` — Modelos Pydantic

| Modelo | Campos | Uso |
|---|---|---|
| `ClassifyOut` | `intent`, `direct_reply` | Classificação de intent da mensagem |
| `ExtractedParams` | `metric`, `grain`, `period_type`, `comparison`, `product_filter`, `category_filter`, `limit`, `order`, `preferred_table`, `period_detail`, `period_detail_2`, `summary` | Parâmetros extraídos da pergunta |
| `SqlOut` | `sql`, `note` | SQL gerado + nota de truncagem |
| `RelevanceCheck` | `relevant`, `reason` | Validação semântica do resultado |

### `sql_generator.py` — Prompts do sistema

| Constante | Modelo | Tokens aprox. | Função |
|---|---|---|---|
| `CLASSIFY_SYSTEM` | `gpt-5.4-nano` | ~1.8k | Classifica intent: `data_query`, `help`, `greeting`, `out_of_scope`, `bypass`, `forecast`, `privacy`, `raw_data`, `relatorio_hortifruti`, `reposicao_hortifruti`, `reposicao_mais_hortifruti` |
| `EXTRACT_SYSTEM` | `gpt-5.4-nano` | ~5k | Extrai parâmetros estruturados da pergunta |
| `SQL_SYSTEM_TEMPLATE` | `gpt-5.4-nano` | ~18k (estático, cacheável) | Regras de geração de SQL + 30+ exemplos. **Estático** — contexto dinâmico vai no user message para ativar prompt caching da OpenAI. |
| `WRITER_SYSTEM` | `gpt-4o-mini` | ~4k | Formata resposta final em português, com regras de período, emoji, privacidade, comparações. |

### `sql_generator.py` — Métodos da classe `SqlGenerator`

| Método | Modelo | Max tokens saída | Descrição |
|---|---|---|---|
| `classify()` | `gpt-5.4-nano` | 25k | Classifica a intent da mensagem |
| `extract_params()` | `gpt-5.4-nano` | 25k | Extrai parâmetros estruturados + injeta profile_hint |
| `generate_sql()` | `gpt-5.4-nano` | 25k | Gera SQL. System=estático (cacheable); user=cnpj+today+params+schema |
| `check_relevance()` | `gpt-5.4-nano` | 25k | Valida semanticamente se o resultado responde a pergunta (~300 tokens) |
| `write_answer()` | `gpt-4o-mini` | 700 | Formata resposta final |
| `generate_insight()` | `gpt-4o-mini` | 600 | Gera insight com mix sugerido + oportunidade sazonal |
| `generate_themed_insight()` | `gpt-4o-mini` | 500 | Insight livre sobre tema escolhido pelo usuário |
| `generate_hortifruti_report()` | `gpt-4o-mini` | 2000 | Relatório completo de estoque hortifruti |
| `generate_hortifruti_reposicao()` | `gpt-4o-mini` | 1000 | Lista de reposição hortifruti com dica sazonal |

### `graph_agent.py` — Nós do grafo

#### `preprocess_node`
1. Resolve CNPJ pelo telefone (com cache in-memory de 5min)
2. Carrega perfil de longo prazo (`profile_store`)
3. Verifica flag `pending_insight_theme` → rota `insight_theme`
4. Detecta comandos de estoque (palavras-chave)
5. Detecta perguntas de validade de produto → `direct_reply`
6. Chama `classify()` → roteia por intent
7. Personaliza saudação com nome do usuário (intent `greeting`)

#### `extract_node`
- Chama `extract_params()` com contexto do turno anterior (follow-ups)
- Injeta `profile_hint` para desambiguação
- Normaliza filtros (remove acentos, converte para maiúsculas)
- Corrige `period_detail` futuro

#### `cert_check_node`
- Só executa para `metric = gastos`
- Verifica se o CNPJ tem certificado digital válido no Databricks
- Se vencido, retorna link de renovação

#### `sql_gen_node`
- Gera SQL com `generate_sql()` passando parâmetros extraídos
- Mantém histórico de tentativas (`sql_attempts`)
- Limpa o SQL de markdown antes de validar

#### `execute_node`
- Executa o SQL no Databricks via `run_query()`
- Captura erros e retorna para o supervisor
- Processa resultado em `columns` + `rows`

#### `supervisor_node`

O supervisor é o coração da qualidade. Executa em **4 fases**:

**Fase 1 — Erros e retries:**
- Erro de execução SQL → interpreta com `_interpret_sql_error()` → retry (máx 3)
- Resultado vazio → feedback hierárquico → retry
- Poucos resultados para produto → fallback hierárquico → retry
- Filtro multi-palavra exato → reescreve como AND de palavras → retry

**Fase 2 — Qualidade (sem retry):**
- Contexto de extração ausente → nota para o writer
- Estoque: analisa saldos negativos e zerados
- Colunas inconsistentes com a métrica

**Fase 2.5 — Validação semântica:**
- Chama `check_relevance()` com sample dos dados
- Se irrelevante → retry com motivo preciso

**Fase 2.6 — Datas ausentes:**
- Se não há coluna de data no resultado e a métrica precisa de período → retry solicitando `DATA_INICIO`/`DATA_FIM`

**Fase 3 — Sanitização:**
- Trunca a 20 linhas (produtos/estoque) ou 50 (séries)
- Trata certificado vencido → mensagem especial com link

#### `write_node`
1. Pré-computa `period_label` a partir das colunas de data retornadas
2. Detecta diagnostico positivo/negativo vazio → pergunta tema → seta `pending_insight_theme`
3. Chama `write_answer()` com dados + period_label
4. Atualiza `profile_store` com métrica e período usados

#### `insight_node`
- Executa apenas para métricas que têm valor de insight (exclui certificado, estoque, PDV)
- Chama `generate_insight()` com today para calendário sazonal
- Armazena resultado em `pending_insight` na sessão

#### `insight_theme_node`
- Ativado quando `pending_insight_theme` está setado na sessão
- Limpa o flag, chama `generate_themed_insight()` com o tema digitado pelo usuário

#### `hortifruti_report_node`
- Busca dados de estoque hortifruti no Databricks
- Categoriza em: crítico, comprar mais, comprar menos, bem gerenciados
- Gera relatório ou lista de reposição dependendo do intent
- Suporta paginação (reposicao_mais_hortifruti)

### Calendário sazonal (`_SEASONAL_CALENDAR`)

Usado em `generate_insight()` e `generate_themed_insight()`. Referência de eventos brasileiros com produtos relevantes para cada período. O LLM identifica quais eventos estão próximos com base em `today` e:
- Gera **alertas de risco** (⚠️) para produtos em queda que são estratégicos para o evento
- Sugere **novos itens de mix** para o período

---

## 7. Solução estoque

### Arquitetura: state machine

O módulo de estoque é uma state machine independente ativada por comandos explícitos:

```
IDLE → AGUARDANDO_PRODUTO → AGUARDANDO_QUANTIDADE → CONFIRMANDO → IDLE
```

### `db_ops.py`

Operações diretas no Databricks com permissão de escrita (não passa pelo `run_query` com bloqueio de DML):

| Função | Tabela | Operação |
|---|---|---|
| `check_client_has_estoque()` | `estoque_quantum_poc` | Verifica disponibilidade |
| `search_products()` | `estoque_quantum_poc` | Busca por nome/GTIN |
| `get_product_by_code()` | `estoque_quantum_poc` | Busca por código interno |
| `update_stock_quantity()` | `estoque_quantum_poc` | UPDATE de quantidade |
| `get_stock_movements()` | `estoque_quantum_poc` | Histórico de movimentações |

### `quantity_parser.py`

Converte linguagem natural em quantidade numérica:
- `"uma caixa"` → `1`
- `"meia dúzia"` → `6`
- `"três e meio kg"` → `3.5`

---

## 8. Solução sql_bh

Variante do `sql_analytics` para o mercado de Belo Horizonte, com configurações específicas de tabelas e prompts. Compartilha a mesma estrutura de grafo mas com schema e regras adaptadas.

**Endpoints dedicados:**
- `POST /portal/message/bh`
- `POST /whatsapp/webhook/bh`

---

## 9. API — endpoints FastAPI

### Mensagens

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/whatsapp/webhook` | Webhook Infobip — mensagens WhatsApp produção |
| `POST` | `/portal/message` | Mensagens via portal web (requer `cnpj` no body) |
| `POST` | `/debug/message` | Teste local: aceita `phone` + `message`, resolve CNPJ automaticamente |
| `POST` | `/whatsapp/webhook/bh` | Webhook BH |
| `POST` | `/portal/message/bh` | Portal BH |

**Body `/debug/message`:**
```json
{
  "phone": "349XXXXXXXX",
  "message": "quanto vendi hoje?"
}
```
Normalização automática: adiciona `55` se ausente; insere `9` para números de 8 dígitos.

### Admin

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/admin/login` | Página de login |
| `POST` | `/admin/login` | Autenticação |
| `GET` | `/admin/logout` | Logout |
| `GET` | `/admin/conversations` | Lista de conversas com filtros |
| `GET` | `/admin/conversations/{thread_id}` | Detalhes de uma conversa |
| `GET` | `/admin/daily-report` | Relatório do dia |

### Outros

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/portal/insight/feedback` | Registra feedback SIM/NÃO do insight |
| `GET` | `/health` | Health check |

---

## 10. Banco de dados Databricks

Todas as queries são executadas via `POST /api/2.0/sql/statements` com polling assíncrono.

### Tabelas principais

| Tabela | Granularidade | Uso |
|---|---|---|
| `imaiscatalog.gold_prod.mvp_dados_intermediarios` | Item por nota | Vendas diárias, produtos, categorias, faturamento detalhado |
| `imaiscatalog.gold_prod.nova_mvp_vendas` | Mensal por CNPJ | Faturamento mensal, ticket médio, comparativo de mercado |
| `imaiscatalog.gold_prod.nova_mvp_curva_abcd` | Produto por CNPJ | Curva ABCD — representatividade do produto na loja vs mercado |
| `imaiscatalog.gold_prod.pdv_daily_metrics` | Diário por PDV | Travamentos, quedas de internet, notas processadas |
| `imaiscatalog.bronze_prod.configpdv` | Por PDV | Versão e configuração do software PDV |
| `imaiscatalog.bronze_prod.cadcrfclitgv` | Por CNPJ | Certificados digitais |
| `imaiscatalog.silver_prod.estoque_quantum_poc` | Snapshot diário por produto | Saldo de estoque (JOIN obrigatório com `dim_cli`) |
| `imaiscatalog.gold_prod.dim_cli` | Por cliente | Conversão SRK_CLI ↔ CNPJ (só para estoque) |
| `imaiscatalog.gold_prod.whatsapp_user_permissions` | Por usuário | Mapeamento telefone → CNPJ + nome do usuário |
| `gold_prod.imais_conversas_diarias` | Por mensagem | Espelho das conversas (sincronizado pelo script diário) |

### Segurança de queries

O `run_query()` aplica dois filtros:
1. `FORBIDDEN_SQL` — bloqueia `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `MERGE`
2. `SAFE_PREFIX` — exige que a query comece com `SELECT`, `WITH`, `SHOW` ou `DESCRIBE`

Exceção: o módulo `estoque/db_ops.py` usa `_execute_statement()` diretamente (com permissão de escrita) pois precisa executar `UPDATE` de estoque. Esse SQL nunca é gerado por LLM.

---

## 11. Persistência local (SQLite)

Arquivo: `conversations.db` (caminho configurável via `CONVERSATIONS_DB_PATH`)

### Tabelas

| Tabela | Descrição |
|---|---|
| `conversations` | Histórico de todas as mensagens (question, answer, insight, feedback) |
| `user_profiles` | Perfil de longo prazo por CNPJ (métricas frequentes, last_seen) |
| LangGraph checkpoints | Estado do grafo por `thread_id` (gerenciado pelo `AsyncSqliteSaver`) |

### `thread_id` por canal

| Canal | Formato do thread_id |
|---|---|
| WhatsApp | `phone` (ex: `5534996658741`) |
| Portal | `portal_{cnpj}` (ex: `portal_05152108000113`) |

O mesmo `thread_id` garante continuidade de contexto (follow-ups) entre turnos.

---

## 12. Variáveis de ambiente

```env
# Databricks
DATABRICKS_SERVER_HOSTNAME=adb-xxxx.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxx
DATABRICKS_TOKEN=dapi...

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-nano              # Classificação, extração, SQL
OPENAI_MODEL_WRITER=gpt-4o-mini        # Redação, insights

# Infobip (WhatsApp)
INFOBIP_API_KEY=...
INFOBIP_BASE_URL=...
INFOBIP_SENDER=...

# App
CONVERSATIONS_DB_PATH=conversations.db
ADMIN_USERNAME=admin
ADMIN_PASSWORD=...
SECRET_KEY=...                         # Sessões admin
```

---

## 13. Scripts de automação

### `scripts/daily_report.py`

Executado diariamente às 23:59 via Windows Task Scheduler.

**Fluxo:**
1. Verifica se a tabela `gold_prod.imais_conversas_diarias` está vazia (first run → full load)
2. Faz `MERGE` das conversas do dia para o Databricks (lotes de 50)
3. Monta resumo HTML com threads ativas + mensagens do dia
4. Envia e-mail de relatório
5. Marca como enviado no SQLite

**Lógica de MERGE:**
- Chave: `(thread_id, created_at)`
- `WHEN MATCHED AND feedback != ''` → atualiza `insight_feedback`
- `WHEN NOT MATCHED` → insere novo registro

---

## 14. Fluxo completo de uma mensagem

### WhatsApp

```
1. Infobip → POST /whatsapp/webhook
2. main.py extrai phone + message
3. normalize_phone() + adiciona prefixo 55/9 se necessário
4. Verifica session_store: CNPJ já resolvido?
   └── Não → get_cnpjs_for_phone() → se >1 CNPJ → apresenta menu de seleção
5. Verifica active_solution
   ├── "estoque" → estoque/router.py (state machine)
   └── "sql_analytics" → sql_analytics/router.py
6. sql_analytics/router.py → _dispatch() → LangGraph
7. LangGraph executa os nós (preprocess → ... → write → insight)
8. Resposta retornada
9. Infobip envia texto ao usuário
10. Se insight gerado → send_insight_buttons() (botões SIM/NÃO)
11. conversation_logger.log() → SQLite
```

### Portal web

```
1. POST /portal/message com {phone, message, cnpj}
2. Direto para _dispatch() (CNPJ já resolvido)
3. Resposta JSON com {answer, insight}
```

---

## 15. Modelos de linguagem e tokens

### Custo por pergunta (estimativa)

| Etapa | Modelo | Tokens típicos |
|---|---|---|
| Classificação | `gpt-5.4-nano` | ~1.8k |
| Extração de parâmetros | `gpt-5.4-nano` | ~5k |
| Geração de SQL | `gpt-5.4-nano` | ~18.5k entrada + ~200 saída |
| Validação semântica | `gpt-5.4-nano` | ~300 |
| Redação da resposta | `gpt-4o-mini` | ~4-5k entrada + ~500 saída |
| Insight | `gpt-4o-mini` | ~1.5k |
| **Total típico** | | **~31k tokens** |

### Prompt caching

O `SQL_SYSTEM_TEMPLATE` (~18k tokens) é **100% estático** — sem variáveis de substituição. Todo o contexto dinâmico (CNPJ, data, parâmetros, schema, erros) fica no user message. A OpenAI cacheia automaticamente prefixos de system message idênticos, reduzindo custo e latência das queries SQL subsequentes.

### Fluxo de tokens acumulados

O `SqlGenerator` mantém `_tokens_by_model` acumulando tokens por modelo durante o ciclo de vida da pergunta. O log imprime o total consolidado ao final de cada `write_answer()`.

---

## 16. Melhorias recentes implementadas

### Supervisor semântico
Após cada query bem-sucedida, `check_relevance()` verifica com ~300 tokens se as colunas retornadas realmente respondem a pergunta. Ex: pediu faturamento mas voltou só quantidade → retry automático com feedback.

### Datas obrigatórias no resultado
O supervisor verifica se há colunas de período no resultado (`DATA_INICIO`, `DATA_FIM`, `ANO_MES`, etc.). Se ausentes e a métrica precisa de período → retry solicitando as colunas. Garante que o writer sempre mostra datas reais ao invés de "nos últimos 30 dias".

### Prompt caching
Separação do system message (estático) e user message (dinâmico) no `generate_sql()`. Ativa o desconto de cache da OpenAI nos ~18k tokens de regras e exemplos.

### Memória de longo prazo
Tabela `user_profiles` no SQLite registra padrão de uso por CNPJ. Um hint é injetado no extrator de parâmetros em cada turno: `[Perfil do lojista: 47 consultas. Métricas mais pedidas: diagnostico_negativo (12x), faturamento (8x)]`.

### Flow de insight temático
Quando `diagnostico_positivo/negativo` retorna vazio, o bot pergunta o tema desejado e gera um insight consultivo livre via `generate_themed_insight()`, evitando dead-ends.

### Calendário sazonal nos insights
`generate_insight()` inclui seção `*💡 Oportunidade sazonal:*` com:
- **Alertas de risco** (⚠️): produto em queda nos dados que é estratégico para evento próximo (ex: cerveja caindo com Copa chegando)
- **Sugestões de mix**: novos itens para o período (Festa Junina, Copa, Natal, etc.)
