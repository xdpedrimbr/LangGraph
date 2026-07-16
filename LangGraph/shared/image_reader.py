"""Leitura de listas de produtos a partir de uma FOTO enviada no WhatsApp.

Baixa a imagem inbound do Infobip e usa um modelo de visão da OpenAI para
extrair o texto da lista (produto + quantidade), que depois é interpretado pelo
parser de movimentação do estoque.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image, ImageEnhance, ImageFilter

load_dotenv()

INFOBIP_BASE_URL = (os.getenv("INFOBIP_BASE_URL") or "l23znj.api-us.infobip.com").strip()
INFOBIP_API_KEY  = (os.getenv("INFOBIP_API_KEY")  or "").strip()
OPENAI_API_KEY   = (os.getenv("OPENAI_API_KEY")   or "").strip()
OPENAI_VISION_MODEL  = (os.getenv("OPENAI_MODEL_VISION")  or "gpt-4o").strip()
OPENAI_CORRECT_MODEL = (os.getenv("OPENAI_MODEL_ESTOQUE") or "gpt-4o-mini").strip()

_base = INFOBIP_BASE_URL if INFOBIP_BASE_URL.startswith("http") else f"https://{INFOBIP_BASE_URL}"

_openai_client: Optional[AsyncOpenAI] = None


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── Pré-processamento da imagem ───────────────────────────────────────────────

def _preprocess_image(image_bytes: bytes) -> bytes:
    """Converte para escala de cinza e aumenta contraste/nitidez.

    Manuscritos em papel ganham muito com este passo: o modelo de visão lida
    melhor com texto preto nítido sobre fundo branco do que com a foto raw.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Escala de cinza — elimina ruído de cor e melhora contraste do texto
        img = img.convert("L")
        # Boost de contraste (2.5×): torna a tinta mais escura e o papel mais branco
        img = ImageEnhance.Contrast(img).enhance(2.5)
        # Nitidez em duas passagens: bordas das letras ficam mais definidas
        img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        preprocessed = buf.getvalue()
        print(f"[IMAGE preprocess] {len(image_bytes)} → {len(preprocessed)} bytes")
        return preprocessed
    except Exception as e:
        print(f"[IMAGE preprocess] erro ({e}), usando imagem original")
        return image_bytes


# ── Prompt de OCR ─────────────────────────────────────────────────────────────

_OCR_PROMPT = (
    "Esta é uma foto de uma anotação de estoque de um supermercado/hortifruti brasileiro "
    "(manuscrita ou impressa). Pode estar girada — corrija a orientação mentalmente.\n\n"

    "INDICADORES DE AÇÃO — o lojista usa palavras OU sinais:\n"
    "  • Saída (-): palavras como 'saiu', 'vendi', 'baixa', 'retirei', ou o sinal '-' / '–'\n"
    "  • Entrada (+): palavras como 'entrou', 'chegou', 'comprei', 'recebi', ou o sinal '+'\n"
    "Um indicador de ação no início de uma seção/coluna vale para TODOS os itens abaixo "
    "até aparecer outro indicador. Propague o indicador em CADA linha.\n\n"

    "⚠️ SEM NENHUM INDICADOR DE AÇÃO (nem palavra, nem +/-): isso é comum em fichas formais "
    "de controle (ex: 'CONTROLE DE PRODUTOS', planilha com colunas Data | Código | Descrição "
    "do Produto | Qtd | Visto). Esse tipo de ficha é usado para registrar PERDA/QUEBRA de "
    "produto, então o PADRÃO é SAÍDA — prefixe TODAS as linhas com 'saiu' quando a imagem "
    "inteira não tiver nenhum sinal de ação em lugar nenhum.\n\n"

    "FORMATO DE TABELA/PLANILHA: se a foto for uma tabela com colunas como 'Data', 'Código', "
    "'Descrição do Produto', 'Qtd'/'Quant', 'Visto'/'Assin.': use a coluna de DESCRIÇÃO como "
    "nome do produto e a coluna de QUANTIDADE como quantidade. IGNORE completamente as colunas "
    "Data, Código (código de barras/EAN) e Visto/Assinatura — não são produto nem quantidade.\n\n"

    "NÚMEROS NO PADRÃO BRASILEIRO: vírgula é separador DECIMAL, não de milhar. "
    "'3,292' significa 3.292 (não 3292). '0,500' significa 0.5.\n"
    "Se a quantidade de uma linha tiver vários números separados por '+' (ex: "
    "'0,500+ 1,600+ 3,00'), são pesagens parciais do mesmo item — SOME todos para obter "
    "a quantidade total da linha (nesse exemplo: 5,1).\n\n"

    "⚠️ ATENÇÃO ESPECIAL AOS SINAIS '+' E '-' POR LINHA:\n"
    "Quando CADA linha tem seu próprio sinal individual (não um cabeçalho único de seção), "
    "leia o sinal de CADA linha separadamente, alinhado visualmente com aquela linha "
    "específica. NÃO repita ou 'herde' o sinal da linha anterior — cada linha pode ter um "
    "sinal diferente da anterior (ex: uma lista pode ir +,+,-,+,-,+ linha a linha).\n"
    "Diferencie com cuidado: '+' tem um traço VERTICAL cruzando o horizontal; "
    "'-' é APENAS um traço horizontal, sem cruzamento. Se houver qualquer indício de "
    "traço vertical (mesmo curto ou desalinhado), é '+' (entrada), não '-'.\n"
    "Antes de finalizar, releia cada linha conferindo o sinal contra a imagem — não assuma "
    "um padrão (não force todas como iguais só porque as primeiras eram iguais).\n\n"

    "COMO TRANSCREVER — UM item por linha, no formato:\n"
    "  <ação> <quantidade> <produto completo>\n"
    "  Ação = 'saiu' (para saída) ou 'entrou' (para entrada).\n"
    "  Produto = nome COMPLETO exatamente como escrito (variedade, marca, peso, embalagem).\n"
    "  NUNCA encurte para só a primeira palavra: '5 tomate cereja bandeja 300g' → "
    "product='tomate cereja bandeja 300g'.\n\n"

    "PRODUTOS COMUNS de supermercado brasileiro — se uma palavra estiver ilegível mas o "
    "contexto indicar um destes, prefira o nome correto:\n"
    "  Hortifruti: tomate (italiano, caqui, cereja, uva, mel, sweet grape, débora), "
    "alface (lisa, crespa, americana, romana, roxa, mimosa), almeirão, rúcula, agrião, couve, "
    "espinafre, acelga, cenoura, beterraba, abobrinha, pepino, chuchu, vagem, berinjela, "
    "quiabo, pimentão, pimenta, gengibre, abóbora, mandioquinha, batata (inglesa, doce, "
    "bolinha, yacon), mandioca, cebola, alho\n"
    "  Frutas: banana (prata, nanica, da terra), maçã, laranja (bahia, lima), limão, manga "
    "(tommy, palmer), abacate, mamão, melão, melancia, uva, maracujá, goiaba, abacaxi, coco, "
    "tangerina, caju, jambo, cupuaçu, morango, kiwi\n"
    "  Outros setores (mercearia/padaria/snacks): castanha, amêndoa, mix de frutas secas, "
    "chips, pasta de amendoim, geleia, granola, tapioca\n\n"

    "CUIDADO COM LETRAS PARECIDAS em letra cursiva:\n"
    "  • 'c' e 'u' são muito similares: 'cereja' pode parecer 'uereja'\n"
    "  • 'n' e 'h' são similares\n"
    "  • 'ei' e 'ai' são similares: prefira a versão que forma palavra real\n"
    "  • Números: '1' e '7', '3' e '8', '0' e '6' podem se confundir\n"
    "  Se uma leitura literal não forma um produto real, reconsidere as letras duvidosas.\n\n"

    "Exemplo de saída com lista livre (note: '+' na foto → 'entrou', '-' → 'saiu'; sinais "
    "podem alternar linha a linha, leia cada um independentemente):\n"
    "entrou 2 alface lisa\n"
    "saiu 3 almeirão\n"
    "entrou 1 tomate cereja bandeja 300g\n"
    "saiu 2 tomate italiano\n\n"

    "Exemplo de saída com tabela de 'Controle de Produtos' (sem indicador de ação → "
    "tudo 'saiu'; vírgula = decimal; código/data ignorados):\n"
    "saiu 3.292 batata monalisa\n"
    "saiu 1.788 banana prata\n"
    "saiu 0.680 chuchu\n\n"

    "Não invente itens nem quantidades; transcreva apenas o que está escrito. "
    "Se o nome de um produto estiver REALMENTE ilegível (não dá pra reconhecer nem com "
    "contexto), escreva 'ILEGIVEL' no lugar do nome em vez de chutar um produto que pareça "
    "plausível — é melhor admitir que não leu do que inventar um nome errado.\n"
    "Se a imagem não tiver nenhuma lista de produtos legível, responda exatamente: SEM_LISTA"
)

# ── Passo de correção pós-OCR ─────────────────────────────────────────────────

_CORRECTION_SYSTEM = (
    "Você é um especialista em produtos de supermercado/hortifruti brasileiro. "
    "Receberá uma lista extraída por OCR de uma nota manuscrita. O OCR pode ter errado "
    "algumas letras — sua tarefa é corrigir APENAS erros óbvios de leitura, sem inventar "
    "itens nem alterar quantidades.\n\n"
    "Regras:\n"
    "- Corrija só palavras que claramente não são nomes de produto reais.\n"
    "- Se um nome parece errado mas pode estar certo, mantenha.\n"
    "- Não mude 'saiu'/'entrou', quantidades ou unidades.\n"
    "- Retorne a lista corrigida no mesmo formato linha a linha, sem explicações.\n\n"
    "Exemplos de correção:\n"
    "  'saiu 2 tomate uva 300g' → 'saiu 2 tomate cereja 300g'  (uva ≠ variedade de tomate; cereja sim)\n"
    "  'entrou 3 alface clisa' → 'entrou 3 alface lisa'  (clisa não existe)\n"
    "  'saiu 1 almeiram' → 'saiu 1 almeirão'  (correção de acento/terminação)\n"
    "  'entrou 5 tomate itahano' → 'entrou 5 tomate italiano'  (h→l)\n"
)


async def _correct_ocr(raw_text: str) -> str:
    """Passa o texto OCR por um LLM leve para corrigir erros óbvios de leitura."""
    try:
        resp = await _get_openai().chat.completions.create(
            model=OPENAI_CORRECT_MODEL,
            messages=[
                {"role": "system", "content": _CORRECTION_SYSTEM},
                {"role": "user",   "content": raw_text},
            ],
            max_tokens=400,
            temperature=0,
        )
        corrected = (resp.choices[0].message.content or "").strip()
        if corrected and "SEM_LISTA" not in corrected.upper():
            if corrected != raw_text:
                print(f"[IMAGE correction] '{raw_text[:80]}' → '{corrected[:80]}'")
            return corrected
        return raw_text
    except Exception as e:
        print(f"[IMAGE correction] ERRO: {e}, usando OCR original")
        return raw_text


# ── Extração de info da imagem no webhook ─────────────────────────────────────

def extract_image_info(evt: dict) -> Optional[dict]:
    """Extrai url/id/mime de uma imagem do evento Infobip (busca em message e raiz)."""
    message_obj = evt.get("message") or {}
    msg_type = (message_obj.get("type") or evt.get("type") or "").strip().upper()

    if msg_type not in ("IMAGE", "PICTURE", "PHOTO"):
        return None

    media_url = (
        message_obj.get("url")
        or message_obj.get("mediaUrl")
        or evt.get("url")
        or evt.get("mediaUrl")
    )
    media_id = (
        message_obj.get("id")
        or message_obj.get("mediaId")
        or evt.get("mediaId")
    )
    mime_type = (
        message_obj.get("mimeType")
        or message_obj.get("contentType")
        or evt.get("mimeType")
        or "image/jpeg"
    )
    return {"media_url": media_url, "media_id": media_id, "mime_type": mime_type}


async def download_image(media_url: Optional[str] = None, media_id: Optional[str] = None) -> bytes:
    """Baixa a imagem do Infobip por url ou media_id."""
    headers = {"Authorization": f"App {INFOBIP_API_KEY}", "Accept": "*/*"}
    async with httpx.AsyncClient(timeout=60) as client:
        if media_url:
            print("[IMAGE] baixando por media_url")
            r = await client.get(media_url, headers=headers)
            r.raise_for_status()
            return r.content
        if media_id:
            print(f"[IMAGE] baixando por media_id={media_id}")
            url = f"{_base}/whatsapp/1/inbound/media/{media_id}"
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            return r.content
    raise RuntimeError("Nenhuma media_url ou media_id disponível para download da imagem.")


async def read_image_list(image_bytes: bytes, mime_type: str = "image/jpeg") -> Optional[str]:
    """Faz OCR da lista de produtos na imagem. Retorna o texto corrigido ou None."""
    # 1. Pré-processa para melhorar contraste/nitidez (síncrono/CPU-bound — roda em
    # thread separada para não bloquear o event loop e travar outras mensagens).
    processed = await asyncio.to_thread(_preprocess_image, image_bytes)
    b64 = base64.b64encode(processed).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    try:
        resp = await _get_openai().chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }],
            max_completion_tokens=700,
        )
        raw = (resp.choices[0].message.content or "").strip()
        print(f"[IMAGE OCR raw] {raw[:150]!r}")
        if not raw or "SEM_LISTA" in raw.upper():
            return None

        # 2. Passo de correção: LLM leve verifica nomes de produto
        corrected = await _correct_ocr(raw)
        return corrected
    except Exception as e:
        print(f"[IMAGE OCR] ERRO: {e}")
        return None


async def read_image_message(evt: dict) -> Optional[str]:
    """Entry point: dado o evento Infobip de imagem, retorna o texto da lista ou None."""
    info = extract_image_info(evt)
    if not info:
        return None
    try:
        img_bytes = await download_image(media_url=info["media_url"], media_id=info["media_id"])
        print(f"[IMAGE] {len(img_bytes)} bytes baixados")
        return await read_image_list(img_bytes, info["mime_type"])
    except Exception as e:
        print(f"[IMAGE] ERRO ao ler imagem: {e}")
        return None
