# src/agregacao.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import os
import pandas as pd
import numpy as np

from .utils import yyyymm_key  # gera "YYYYMM" a partir de datas

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Carga
# =============================
def carregar_eventos_qualificados() -> pd.DataFrame:
    fp = OUT_DIR / "eventos_qualificados.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Não encontrei {fp}. Rode antes: python -m src.matching")
    df = pd.read_parquet(fp)

    # colunas mínimas esperadas
    for c in ["ativo", "data_evento", "equipamento", "periodo_h", "bbl"]:
        if c not in df.columns:
            df[c] = pd.NA

    # tipos
    df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    for c in ("periodo_h", "bbl", "perda_financeira_usd", "brent_usd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =============================
# Brent
# =============================
def _normalize_brent_dict(brent_mensal: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza chaves do dicionário mensal para 'YYYYMM', aceitando 'YYYYMM' ou 'YYYY-MM'.
    """
    out: Dict[str, float] = {}
    for k, v in brent_mensal.items():
        ks = str(k).strip()
        if not ks:
            continue
        if "-" in ks:
            ks = ks.replace("-", "")
        out[ks] = float(v) if v is not None and not pd.isna(v) else np.nan
    return out

def aplicar_brent(
    df: pd.DataFrame,
    brent_unico_usd: Optional[float] = None,
    brent_mensal: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Aplica Brent:
      - Se brent_mensal for passado (dict 'YYYYMM' ou 'YYYY-MM' -> USD), usa por mês de data_evento.
      - Senão usa brent_unico_usd (float). Se nada for passado, usa env BRENT_USD_DEFAULT (padrão 80.0).
    """
    df = df.copy()
    df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    for c in ("bbl",):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if brent_mensal:
        bm = _normalize_brent_dict(brent_mensal)
        df["_yyyymm"] = yyyymm_key(df["data_evento"])  # "YYYYMM"
        df["brent_usd"] = df["_yyyymm"].map(bm).astype(float)
    else:
        bm = None
        df["brent_usd"] = np.nan

    # fallback para lacunas da série mensal
    if brent_unico_usd is None:
        brent_unico_usd = float(os.getenv("BRENT_USD_DEFAULT", "80"))
    df["brent_usd"] = df["brent_usd"].fillna(float(brent_unico_usd))

    # perda financeira = bbl * brent
    if "bbl" not in df.columns:
        df["bbl"] = 0.0
    df["perda_financeira_usd"] = df["bbl"].fillna(0) * df["brent_usd"].fillna(0)

    # limpeza
    if "_yyyymm" in df.columns:
        df.drop(columns=["_yyyymm"], inplace=True, errors="ignore")

    return df

# =============================
# Agregação
# =============================
def agregar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # garantia de tipos
    for c in ("periodo_h", "bbl", "perda_financeira_usd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby(["ativo", "equipamento"], as_index=False)
          .agg(
              eventos=("equipamento", "count"),
              horas_paradas_total=("periodo_h", "sum"),
              bbl_perdidos_total=("bbl", "sum"),
              perda_financeira_total_USD=("perda_financeira_usd", "sum"),
          )
    )
    # ordenar por USD desc dentro do ativo
    agg = agg.sort_values(["ativo", "perda_financeira_total_USD"], ascending=[True, False])
    return agg

# =============================
# Runner
# =============================
def run_agregacao(
    brent_unico_usd: Optional[float] = None,
    brent_mensal: Optional[Dict[str, float]] = None
):
    df = carregar_eventos_qualificados()
    df = aplicar_brent(df, brent_unico_usd=brent_unico_usd, brent_mensal=brent_mensal)
    agg = agregar(df)

    # persistir
    agg.to_parquet(OUT_DIR / "agregado_ativo_equip.parquet", index=False)
    agg.to_csv(OUT_DIR / "agregado_ativo_equip.csv", index=False, encoding="utf-8-sig")
    print(f"Agregações geradas: {len(agg)} linhas -> outputs/agregado_ativo_equip.*")

if __name__ == "__main__":
    # exemplo: valor único de Brent (ajuste depois no app)
    run_agregacao(brent_unico_usd=80.0)
