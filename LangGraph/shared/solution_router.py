from __future__ import annotations

# ── Solução padrão ────────────────────────────────────────────────────────────

DEFAULT_SOLUTION = "sql_analytics"

# ── Registro de soluções ───────────────────────────────────────────────────────
# Para adicionar uma nova solução:
#   1. Inclua uma entrada aqui com as frases de ativação.
#   2. Importe o handler em main.py e adicione-o em _SOLUTION_HANDLERS.
#   3. Se a solução tiver inicialização (ex: state machine), faça o hook em main.py.

SOLUTION_REGISTRY: dict[str, dict] = {
    "estoque": {
        # Apenas frases imperativas explícitas — nunca palavras soltas como "estoque"
        # que aparecem em perguntas analíticas ("quanto tenho no estoque?").
        "activation": [
            "movimentar estoque", "modificar estoque",
            "baixar estoque", "abrir estoque",
            "modo estoque", "controle de estoque",
            "lançar produto", "lancar produto",
            "lançar produtos", "lancar produtos",
        ],
    },
}

# Comandos que voltam para a solução padrão
_EXIT_COMMANDS = [
    "sair estoque", "sair do estoque",
    "sair movimentação", "sair movimentacao",
    "sair movimentar", "sair movimentar estoque",
    "fechar estoque",
    "modo análises", "modo analises",
    "voltar análises", "voltar analises",
]


# ── Funções públicas ──────────────────────────────────────────────────────────

def match_activation(message: str) -> str | None:
    """Se a mensagem é um comando de ativação, retorna o nome da solução. Senão, None."""
    msg = (message or "").strip().lower()
    if not msg:
        return None
    for solution, cfg in SOLUTION_REGISTRY.items():
        if any(cmd in msg for cmd in cfg["activation"]):
            return solution
    return None


def is_exit_command(message: str) -> bool:
    """Retorna True se a mensagem é um comando de retorno à solução padrão."""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    return any(cmd in msg for cmd in _EXIT_COMMANDS)
