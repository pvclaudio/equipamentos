# src/ranking.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Optional
import os
import pandas as pd
import numpy as np

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PESOS: Dict[str, float] = {"horas": 0.4, "bbl": 0.3, "usd": 0.3}

# =============================
# Carga e validação
# =============================
def carregar_agregado() -> pd.DataFrame:
    fp = OUT_DIR / "agregado_ativo_equip.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Não encontrei {fp}. Rode antes: python -m src.agregacao")
    df = pd.read_parquet(fp)

    required = ["ativo","equipamento","horas_paradas_total","bbl_perdidos_total","perda_financeira_total_USD","eventos"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Coluna ausente no agregado: {col}")

    # Garantir tipos numéricos
    num_cols = ["horas_paradas_total","bbl_perdidos_total","perda_financeira_total_USD","eventos"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strings “limpas”
    df["ativo"] = df["ativo"].astype(str).str.strip()
    df["equipamento"] = df["equipamento"].astype(str).str.strip()

    return df

# =============================
# Utils
# =============================
def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        # série constante -> tudo zero para não distorcer o composto
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mn) / (mx - mn)

def _normalize_pesos(pesos: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not pesos:
        pesos = DEFAULT_PESOS.copy()
    # valores não numéricos -> 0
    for k in ("horas","bbl","usd"):
        try:
            pesos[k] = float(pesos.get(k, DEFAULT_PESOS[k]))
        except Exception:
            pesos[k] = DEFAULT_PESOS[k]
    total = pesos["horas"] + pesos["bbl"] + pesos["usd"]
    if total <= 0:
        # fallback seguro
        return DEFAULT_PESOS.copy()
    # reescala para somar 1
    return {k: v/total for k, v in pesos.items()}

# =============================
# Geração de rankings
# =============================
def gerar_rankings(
    df: pd.DataFrame,
    pesos: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    w = _normalize_pesos(pesos)

    # normalizações para o composto
    n_horas = _minmax(df["horas_paradas_total"])
    n_bbl   = _minmax(df["bbl_perdidos_total"])
    n_usd   = _minmax(df["perda_financeira_total_USD"])

    df_comp = df.copy()
    df_comp["score_composto"] = (
        w["horas"] * n_horas +
        w["bbl"]   * n_bbl   +
        w["usd"]   * n_usd
    )

    rank_horas = df.sort_values(["ativo", "horas_paradas_total"], ascending=[True, False])
    rank_bbl   = df.sort_values(["ativo", "bbl_perdidos_total"], ascending=[True, False])
    rank_usd   = df.sort_values(["ativo", "perda_financeira_total_USD"], ascending=[True, False])
    rank_comp  = df_comp.sort_values(["ativo", "score_composto"], ascending=[True, False])

    return rank_horas, rank_bbl, rank_usd, rank_comp

# =============================
# Runner (CLI)
# =============================
def run_ranking(pesos: Optional[Dict[str, float]] = None):
    df = carregar_agregado()
    r_horas, r_bbl, r_usd, r_comp = gerar_rankings(df, pesos=pesos)

    r_horas.to_csv(OUT_DIR / "rank_por_horas.csv", index=False, encoding="utf-8-sig")
    r_bbl.to_csv(OUT_DIR / "rank_por_bbl.csv", index=False, encoding="utf-8-sig")
    r_usd.to_csv(OUT_DIR / "rank_por_usd.csv", index=False, encoding="utf-8-sig")
    r_comp.to_csv(OUT_DIR / "rank_composto.csv", index=False, encoding="utf-8-sig")

    r_horas.to_parquet(OUT_DIR / "rank_por_horas.parquet", index=False)
    r_bbl.to_parquet(OUT_DIR / "rank_por_bbl.parquet", index=False)
    r_usd.to_parquet(OUT_DIR / "rank_por_usd.parquet", index=False)
    r_comp.to_parquet(OUT_DIR / "rank_composto.parquet", index=False)

    print("Rankings gerados em outputs/: rank_por_horas/bbl/usd/composto")

if __name__ == "__main__":
    # Permite passar pesos via env, ex: RANK_W_HORAS=0.5 RANK_W_BBL=0.2 RANK_W_USD=0.3
    env_pesos = {
        "horas": float(os.getenv("RANK_W_HORAS", DEFAULT_PESOS["horas"])),
        "bbl":   float(os.getenv("RANK_W_BBL",   DEFAULT_PESOS["bbl"])),
        "usd":   float(os.getenv("RANK_W_USD",   DEFAULT_PESOS["usd"])),
    }
    run_ranking(pesos=env_pesos)
