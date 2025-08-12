# src/utils.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from unidecode import unidecode

# =========================
# Texto / Normalização
# =========================
def strip_accents_lower(s: str) -> str:
    """Remove acentos, minúsculas e trim. Tolera None."""
    if s is None:
        return ""
    return unidecode(str(s)).lower().strip()

def norm_text(texto: str) -> str:
    """
    Normaliza para matching/regex:
    - remove acentos
    - minúsculas
    - mantém apenas [a-z0-9] e espaço simples
    """
    s = strip_accents_lower(texto)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# =========================
# Números pt-BR
# =========================
_SCI_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$')

def to_float_ptbr(x):
    """
    Converte números pt-BR/en-US de forma robusta:
      - '1.234.567,89' -> 1234567.89
      - '1234567.89'   -> 1234567.89
      - preserva notação científica ('1.2e+05')
      - escolhe separador decimal pela ÚLTIMA vírgula/ponto
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s == "":
        return np.nan

    # espaços não‑quebrantes
    s = s.replace("\u00A0", "").replace("\u202F", "").replace(" ", "")

    if _SCI_RE.match(s):
        return pd.to_numeric(s, errors="coerce")

    has_comma = "," in s
    has_dot   = "." in s

    if has_comma and has_dot:
        # o último símbolo define o decimal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    # else: já está en-US

    return pd.to_numeric(s, errors="coerce")

def to_int_ptbr(x):
    """Converte para inteiro (Int64 Pandas), aceitando NaN."""
    v = to_float_ptbr(x)
    if pd.isna(v):
        return pd.NA
    try:
        return int(round(float(v)))
    except Exception:
        return pd.NA

def coerce_cols_ptbr(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Aplica to_float_ptbr em uma lista de colunas do DataFrame."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(to_float_ptbr)
    return df

# =========================
# Datas
# =========================
def parse_data_ptbr(s):
    """Interpreta datas no padrão dia/mês/ano (tolerante)."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

# =========================
# Ativos / Campos
# =========================
_ATIVO_CANON = {
    "bravo": "Bravo",
    "polvo": "Bravo",   # regra PRIO
    "tbmt":  "Bravo",

    "forte": "Forte",
    "abl":   "Forte",

    "frade": "Frade",

    # variações compostas comuns
    "fpso bravo": "Bravo",
    "poço polvo": "Bravo", "poco polvo": "Bravo",
    "polvo a": "Bravo",
    "produção polvo": "Bravo", "producao polvo": "Bravo",
    "poço tbmt": "Bravo", "poco tbmt": "Bravo",
    "produção tbmt": "Bravo", "producao tbmt": "Bravo",
    "subsea tbmt": "Bravo",

    "fpso forte": "Forte",
    "poço abl": "Forte", "poco abl": "Forte",
    "produção abl": "Forte", "producao abl": "Forte",
    "subsea abl": "Forte",

    "fpso frade": "Frade",
    "poço frade": "Frade", "poco frade": "Frade",
    "produção frade": "Frade", "producao frade": "Frade",
    "poço + produção frade": "Frade", "poco + producao frade": "Frade",
    "subsea frade": "Frade",
}

def normalize_ativo(x) -> str:
    """
    Normaliza o nome do ativo/campo considerando regras PRIO:
    Polvo≡TBMT≡Bravo; ABL≡Forte.
    """
    s_raw = "" if x is None else str(x).strip()
    s = strip_accents_lower(s_raw)

    if s in _ATIVO_CANON:
        return _ATIVO_CANON[s]
    if any(k in s for k in ("tbmt", "polvo", "bravo")):
        return "Bravo"
    if any(k in s for k in ("abl", "forte")):
        return "Forte"
    if "frade" in s:
        return "Frade"
    return s_raw

# =========================
# IO Helpers
# =========================
def salvar_parquet(df: pd.DataFrame, path: Path):
    """Salva DataFrame em parquet garantindo a pasta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
