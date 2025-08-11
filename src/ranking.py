from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PESOS = {"horas": 0.4, "bbl": 0.3, "usd": 0.3}

def carregar_agregado() -> pd.DataFrame:
    fp = OUT_DIR / "agregado_ativo_equip.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Não encontrei {fp}. Rode antes: python -m src.agregacao")
    df = pd.read_parquet(fp)
    for col in ["ativo","equipamento","horas_paradas_total","bbl_perdidos_total","perda_financeira_total_USD","eventos"]:
        if col not in df.columns:
            raise ValueError(f"Coluna ausente no agregado: {col}")
    return df

def _minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def gerar_rankings(df: pd.DataFrame, pesos: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pesos is None:
        pesos = DEFAULT_PESOS

    # normalizações para o composto
    n_horas = _minmax(df["horas_paradas_total"])
    n_bbl   = _minmax(df["bbl_perdidos_total"])
    n_usd   = _minmax(df["perda_financeira_total_USD"])

    df_comp = df.copy()
    df_comp["score_composto"] = (
        pesos.get("horas", 0.4) * n_horas +
        pesos.get("bbl",   0.3) * n_bbl   +
        pesos.get("usd",   0.3) * n_usd
    )

    rank_horas = df.sort_values(["ativo", "horas_paradas_total"], ascending=[True, False])
    rank_bbl   = df.sort_values(["ativo", "bbl_perdidos_total"], ascending=[True, False])
    rank_usd   = df.sort_values(["ativo", "perda_financeira_total_USD"], ascending=[True, False])
    rank_comp  = df_comp.sort_values(["ativo", "score_composto"], ascending=[True, False])

    return rank_horas, rank_bbl, rank_usd, rank_comp

def run_ranking(pesos: dict = None):
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
    run_ranking()
# -*- coding: utf-8 -*-

