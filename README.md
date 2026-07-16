# iMAIS Chat — Plataforma Inteligente para Varejo

---

## O Problema

```
Além das limitações do Genie...
┌──────────────────────────────────────────────────────────────────┐
│                        VAREJISTA HOJE                            │
│                                                                  │
│   "Quanto faturei esse mês?"                                     │
│   "Qual meu produto mais vendido?"                               │
│   "Como estou comparado ao mercado?"                             │
│                                                                  │
│          ↓                ↓                ↓                     │
│    Liga pro suporte   Abre dashboard   Espera relatório          │
│       (demora)         (complexo)        (atrasado)              │
└──────────────────────────────────────────────────────────────────┘
```

Informacao demorada, dependente de pessoas, e pouco acessivel.

---

## A Solucao

```
┌──────────────────────────────────────────────────────────────────┐
│                      VAREJISTA COM iMAIS                         │
│                                                                  │
│   "Quanto faturei esse mes?"                                     │
│         |                                                        │
│         v                                                        │
│   Manda mensagem no WhatsApp ou Portal                           │
│         |                                                        │
│         v                                                        │
│   Resposta imediata com dados reais                              │
│         |                                                        │
│         v                                                        │
│   "Seu faturamento em abril/2026 ate agora e R$ 45.320,00.       │
│    A estimativa para o mes e R$ 112.800,00."                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## O que o Chatbot faz HOJE

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Faturamento          "Quanto faturei em marco?"                │
│   Gastos (compras)     "Quanto gastei esse mes?"                 │
│   Ticket Medio         "Qual meu ticket medio?"                  │
│   Transacoes           "Quantos clientes tive ontem?"            │
│   Top Produtos         "Meus 5 produtos mais vendidos"           │
│   Top Categorias       "Categorias que mais faturam"             │
│   Comparacao Mercado   "Como estou vs mercado?"                  │
│   Previsao do Mes      "Quanto vou faturar esse mes?"            │
│   Diagnostico          "O que cresceu e o que caiu?"             │
│   Sugestao de Mix      "O que outras lojas vendem e eu nao?"     │
│   Curva ABCD           "Quais produtos sao curva A?"             │
│   Certificado Digital  Verifica automaticamente                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Projeto

```
LangGraph/
│
├── main.py                       <- Entrada unica da API (FastAPI)
├── .env                          
├── requirements.txt              
├── langgraph.json                <- Registro dos grafos LangGraph
│
├── shared/                       <- Codigo compartilhado entre solucoes
│   ├── __init__.py
│   └── db_client.py              <- Conexao Databricks, execucao SQL
│
└── solutions/                    <- Cada pasta = uma solucao independente
    ├── __init__.py
    └── sql_analytics/            <- SOLUCAO 1: Chatbot SQL (PRONTA)
        ├── __init__.py
        ├── router.py             <- Endpoints FastAPI desta solucao
        ├── graph_agent.py        <- Grafo LangGraph (orquestracao)
        ├── sql_generator.py      <- Prompts e geracao de SQL com IA
        ├── schema_tools.py       <- Validacao de SQL contra o schema
        ├── catalog_loader.py     <- Carrega schema e catalogo
        ├── schema_catalog.json   <- Definicao das tabelas disponiveis
        └── queries_catalog.json  <- Catalogo de queries
```

---

## Endpoints da API

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Servidor: http://localhost:8000                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  GET  /health                                              │  │
│  │  Retorna: { "status": "ok" }                               │  │
│  │  Uso: verificar se o servidor esta no ar                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  POST /sql-analytics/message                               │  │
│  │                                                            │  │
│  │  Entrada (JSON):                                           │  │
│  │  {                                                         │  │
│  │    "phone":   "5534999999999",                             │  │
│  │    "message": "quanto faturei esse mes?"                   │  │
│  │  }                                                         │  │
│  │                                                            │  │
│  │  Saida (JSON):                                             │  │
│  │  {                                                         │  │
│  │    "phone":  "5534999999999",                              │  │
│  │    "answer": "Seu faturamento em abril/2026..."            │  │
│  │  }                                                         │  │
│  │                                                            │  │
│  │  Fluxo interno: WhatsApp/Portal -> POST -> Grafo -> Resp.  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Futuras solucoes terao seus proprios endpoints:                 │
│  POST /alertas/message                                           │
│  POST /pedidos/message                                           │
│  POST /onboarding/message                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Fluxo Completo — Da mensagem a resposta

```
                   WHATSAPP / PORTAL
                           |
                           v
         POST /sql-analytics/message  { phone, message }
                           |
                           v
  ┌─────────────────────────────────────────────────────────────┐
  │                     ROUTER (router.py)                      │
  │                                                             │
  │  1. Recebe phone + message                                  │
  │  2. Busca o CNPJ vinculado ao telefone no banco             │
  │  3. Cria thread_id = phone_cnpj (memoria por loja)          │
  │  4. Invoca o grafo LangGraph                                │
  └────────────────────────┬────────────────────────────────────┘
                           |
                           v
  ┌─────────────────────────────────────────────────────────────┐
  │                  GRAFO LANGGRAPH (graph_agent.py)           │
  │                                                             │
  │  ┌─────────────┐                                            │
  │  │  PREPROCESS │ Identifica o cliente (CNPJ) pelo telefone  │
  │  │             │ Classifica a intencao da mensagem:         │
  │  │             │ - data_query -> continua o fluxo           │
  │  │             │ - help/greeting/privacy -> resposta direta │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │   EXTRACT   │  Extrai parametros da pergunta:            │
  │  │             │  - Metrica (faturamento, top produtos...)  │
  │  │             │  - Periodo (este mes, ontem, semana...)    │
  │  │             │  - Filtros (produto, categoria)            │
  │  │             │  - Tabela ideal para consultar             │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │  CERT CHECK │  Se a metrica e "gastos":                  │
  │  │             │  - Verifica se tem notas de entrada        │
  │  │             │  - Valida certificado digital              │
  │  │             │  - Se invalido, orienta o cliente          │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │   SQL GEN   │ IA gera a consulta SQL baseada nos         │
  │  │             │ parametros extraidos + schema do banco     │
  │  │             │ Valida seguranca (so SELECT permitido)     │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │   EXECUTE   │ Executa a SQL no Databricks                │
  │  │             │ Retorna colunas + linhas do resultado      │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │  SUPERVISOR │  Valida o resultado:                       │
  │  │             │  - Vazio? Tenta com filtros mais amplos    │
  │  │             │  - Erro? Reenvia pro SQL GEN (ate 3x)      │
  │  │             │  - OK? Segue pro WRITE                     │
  │  └──────┬──────┘                                            │
  │         |                                                   │
  │         v                                                   │
  │  ┌─────────────┐                                            │
  │  │    WRITE    │ IA formata a resposta em linguagem         │
  │  │             │ natural, amigavel para o varejista         │
  │  │             │ Ex: "Voce faturou R$ 45.320 em abril..."   │
  │  └─────────────┘                                            │
  │                                                             │
  └────────────────────────┬────────────────────────────────────┘
                           |
                           v
              Resposta JSON { phone, answer }
                           |
                           v
      WHATSAPP / PORTAL exibe a mensagem para o varejista
```

---

## Ciclo de Auto-correcao (Supervisor)

```
                   SQL GEN
                      |
                      v
                   EXECUTE
                      |
                      v
                  SUPERVISOR
                   /      \
                  /        \
           resultado      resultado
             ruim           bom
              |               |
              v               v
          SQL GEN           WRITE
         (tenta de         (resposta
          novo)             final)

  (Esse fluxo esta bem mais visual no langgraph)
  O supervisor tenta ate 3 vezes antes de desistir.
  Cada tentativa recebe feedback do erro anterior
  para gerar uma SQL melhor.
```

---

## Arquitetura — Plataforma Aberta

```
┌──────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│                    (entrada unica da API)                        │
│                                                                  │
│  WhatsApp ──┐                                                    │
│             ├──> FastAPI ──> roteia para a solucao certa         │
│  Portal   ──┘                                                    │
└──────┬───────────────────────────┬───────────────────────────────┘
       |                           |
       v                           v
┌──────────────────┐    ┌─────────────────────────────────────────┐
│  shared/         │    │  solutions/                             │
│  (codigo comum)  │    │                                         │
│                  │    │  ├── sql_analytics/   <- PRONTA         │
│  - db_client.py  │    │  │   router.py  (POST /sql-analytics/*) │
│    (Databricks)  │    │  │   graph_agent.py                     │
│                  │    │  │   sql_generator.py                   │
│  - futuro:       │    │  │                                      │
│    auth.py       │    │  ├── alertas/         <- FUTURA         │
│    utils.py      │    │  │   router.py  (POST /alertas/*)       │
│                  │    │  │   graph_agent.py                     │
│                  │    │  │                                      │
│                  │    │  ├── pedidos/          <- FUTURA        │
│                  │    │  │   router.py  (POST /pedidos/*)       │
│                  │    │  │   graph_agent.py                     │
│                  │    │  │                                      │
└──────────────────┘    └─────────────────────────────────────────┘

Para adicionar uma nova solucao:
  1. Criar pasta em solutions/
  2. Criar router.py com endpoints
  3. Criar graph_agent.py com o grafo
  4. Registrar no main.py (2 linhas)
  5. Registrar no langgraph.json (1 linha)
  NAO MEXER EM OUTRAS SOLUCOES QUE JA FUNCIONAM.
```

---

## Tecnologias

```
┌───────────────┬──────────────────────────────────────────────┐
│  Componente   │  Tecnologia                                  │
├───────────────┼──────────────────────────────────────────────┤
│  Orquestracao │  LangGraph (grafos de agentes IA)            │
│  API          │  FastAPI (Python)                            │
│  IA           │  OpenAI GPT-4o-mini / GPT-5.4-Nano           │
│  Banco        │  Databricks SQL (dados reais do varejista)   │
│  Observacao   │  LangSmith (monitoramento de conversas)      │
│  Canal        │  WhatsApp + Portal Web                       │
│  Servidor     │  Proprio (sem limite de grafos/solucoes)     │
└───────────────┴──────────────────────────────────────────────┘
```

---

## Possibilidades Futuras

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  SOLUCAO 1     Chatbot SQL Analytics (PRONTA)                    │
│  ----------    Perguntas sobre dados do varejista                │
│                Faturamento, previsoes, curva ABCD, mix           │
│                                                                  │
│  SOLUCAO 2     Alertas Proativos                                 │
│  ----------    "Seu faturamento caiu 20% essa semana."           │
│                "Voce tem 3 produtos sem venda ha 30 dias."       │
│                                                                  │
│  SOLUCAO 3     Recomendacao de Mix                               │
│  ----------    "Lojas parecidas vendem X e voce nao tem."        │
│                "Categoria Y esta em alta na sua regiao."         │
│                                                                  │
│  SOLUCAO 4     Assistente de Pedidos / Baixa de produtos         │
│  ----------    (Controle de estoque)                             │
│                "Quero pedir 10 caixas de cerveja."               │
│                Integracao direta com sistema de pedidos.         │
│                                                                  │
│  SOLUCAO 5     Onboarding do Varejista                           │
│  ----------    Guia o novo cliente nos primeiros passos.         │
│                Explica funcionalidades, configura certificado.   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Resumo

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  HOJE:      1 solucao funcionando (Chatbot SQL Analytics)        │
│             Vários tipos de consulta disponiveis                 │
│             Auto-correcao inteligente (ate 3 tentativas)         │
│                                                                  │
│  AMANHA:    Novas solucoes plugaveis sem retrabalho              │
│                                                                  │
│  VALOR:     Varejista com acesso imediato aos seus dados,        │
│             sem depender de suporte ou dashboards                │
│                                                                  │
│  ESCALA:    Cada nova solucao e incremental                      │
│             A plataforma ja esta pronta                          │
│                                                                  │
│  CUSTO:     Servidor proprio, sem limite de grafos               │
│             Plano gratuito do LangSmith para monitoramento       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
