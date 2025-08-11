from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode

# ---------- Texto ----------
def strip_accents_lower(s: str) -> str:
    """Remove acentos, baixa para minúsculas e trim."""
    if s is None:
        return ""
    return unidecode(str(s)).lower().strip()

def norm_text(texto: str) -> str:
    """
    Normaliza texto para matching/regex:
    - remove acentos
    - baixa para minúsculas
    - mantém apenas [a-z0-9] e espaços simples
    """
    s = strip_accents_lower(texto)
    # troca qualquer caractere não alfanumérico por espaço
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # colapsa espaços
    return re.sub(r"\s+", " ", s).strip()


# ---------- Números pt-BR ----------
_SCI_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$')

def to_float_ptbr(x):
    """
    Converte números em pt-BR/en-US com robustez:
      - '1.234.567,89'  -> 1234567.89
      - '1234567.89'    -> 1234567.89
      - '4,7E+01'       -> 4.7     (ignora expoente quando decimal usa vírgula)
      - '63,99833333E+14' -> 63.99833333 (idem)
      - científico válido com ponto (p.ex. '6.4e+01') continua valendo.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s == "":
        return np.nan

    # normaliza espaços “estranhos”
    s = s.replace("\u00A0", "").replace("\u202F", "").replace(" ", "")

    has_comma = "," in s
    has_dot   = "." in s
    has_exp   = "e" in s.lower()

    # Caso crítico: vírgula decimal + expoente -> descartamos o expoente
    # (o Excel às vezes exporta assim e não faz sentido para esses campos)
    if has_comma and has_exp:
        s = re.sub(r'[eE][+-]?\d+$', '', s)

    # Se sobrou notação científica “pura” (com ponto), deixa o pandas cuidar
    if _SCI_RE.match(s):
        return pd.to_numeric(s, errors="coerce")

    # Decide separador decimal pela presença de vírgula/ponto
    if has_comma and has_dot:
        # decimal = o último símbolo que aparece (robusto)
        last_comma = s.rfind(",")
        last_dot   = s.rfind(".")
        if last_comma > last_dot:
            s = s.replace(".", "")   # pontos eram milhares
            s = s.replace(",", ".")  # vírgula vira decimal
        else:
            s = s.replace(",", "")   # vírgulas eram milhares
    elif has_comma and not has_dot:
        s = s.replace(".", "")       # se aparecer por engano
        s = s.replace(",", ".")
    # else: já está en-US

    return pd.to_numeric(s, errors="coerce")


def to_int_ptbr(x):
    """Converte '1.234,00' -> 1234 como inteiro (Int64 pandas, aceita NaN)."""
    v = to_float_ptbr(x)
    if pd.isna(v):
        return pd.NA
    try:
        return int(round(float(v)))
    except Exception:
        return pd.NA
    
def coerce_cols_ptbr(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Aplica conversão pt-BR para uma lista de colunas numéricas."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(to_float_ptbr)
    return df


# ---------- Datas ----------
def parse_data_ptbr(s):
    """Tenta interpretar datas com dia primeiro (pt-BR)."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


# ---------- Ativos / Campos ----------
# mapeamento explícito (com acentos removidos e minúsculas)
_ATIVO_CANON = {
    # canônicos
    "bravo": "Bravo",
    "polvo": "Bravo",     # regra PRIO: Polvo ≡ Bravo
    "tbmt": "Bravo",

    "forte": "Forte",
    "abl": "Forte",

    "frade": "Frade",

    # variações compostas (todas -> canônico)
    "fpso bravo": "Bravo",
    "poço polvo": "Bravo",
    "poco polvo": "Bravo",
    "polvo a": "Bravo",
    "produção polvo": "Bravo",
    "producao polvo": "Bravo",
    "poço tbmt": "Bravo",
    "poco tbmt": "Bravo",
    "produção tbmt": "Bravo",
    "producao tbmt": "Bravo",
    "subsea tbmt": "Bravo",

    "fpso forte": "Forte",
    "poço abl": "Forte",
    "poco abl": "Forte",
    "produção abl": "Forte",
    "producao abl": "Forte",
    "subsea abl": "Forte",

    "fpso frade": "Frade",
    "poço frade": "Frade",
    "poco frade": "Frade",
    "produção frade": "Frade",
    "producao frade": "Frade",
    "poço + produção frade": "Frade",
    "poco + producao frade": "Frade",
    "subsea frade": "Frade",
}

def normalize_ativo(x) -> str:
    """
    Normaliza o nome do ativo/campo considerando:
      - dicionário explícito (_ATIVO_CANON) com variações pt-BR
      - regras de equivalência: Polvo≡Bravo, TBMT≡Bravo, ABL≡Forte
      - fallback por substring (bravo/polvo/tbmt -> Bravo; abl/forte -> Forte; frade -> Frade)
    """
    s_raw = str(x).strip()
    s = strip_accents_lower(s_raw)

    # 1) match exato no dicionário
    if s in _ATIVO_CANON:
        return _ATIVO_CANON[s]

    # 2) fallback por palavras-chave (ordem importa)
    if any(k in s for k in ["tbmt", "polvo", "bravo"]):
        return "Bravo"
    if any(k in s for k in ["abl", "forte"]):
        return "Forte"
    if "frade" in s:
        return "Frade"

    # 3) sem mapeamento → retorna original (para auditoria posterior)
    return s_raw


# ---------- IO helpers ----------
def salvar_parquet(df: pd.DataFrame, path: Path):
    """Salva DataFrame em parquet garantindo a pasta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
