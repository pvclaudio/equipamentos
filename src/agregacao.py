from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import os
import pandas as pd

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def carregar_eventos_qualificados() -> pd.DataFrame:
    fp = OUT_DIR / "eventos_qualificados.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Não encontrei {fp}. Rode antes: python -m src.matching")
    df = pd.read_parquet(fp)
    # colunas mínimas esperadas
    for c in ["ativo", "data_evento", "equipamento", "periodo_h", "bbl"]:
        if c not in df.columns:
            df[c] = None
    return df

def aplicar_brent(df: pd.DataFrame, brent_unico_usd: Optional[float]=None,
                  brent_mensal: Optional[Dict[str, float]]=None) -> pd.DataFrame:
    """
    Aplica Brent:
      - Se brent_mensal for passado (dict AAAA-MM -> USD), usa por mês de data_evento.
      - Senão usa brent_unico_usd (float). Se nada for passado, usa placeholder 80.0.
    """
    if brent_mensal:
        df["_yyyymm"] = pd.to_datetime(df["data_evento"]).dt.strftime("%Y-%m")
        df["brent_usd"] = df["_yyyymm"].map(brent_mensal)
    else:
        if brent_unico_usd is None:
            # fallback: variável de ambiente ou placeholder
            brent_unico_usd = float(os.getenv("BRENT_USD_DEFAULT", "80"))
        df["brent_usd"] = float(brent_unico_usd)

    # perda financeira = bbl * brent
    df["perda_financeira_usd"] = df["bbl"].fillna(0) * df["brent_usd"].fillna(0)
    return df

def agregar(df: pd.DataFrame) -> pd.DataFrame:
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

def run_agregacao(brent_unico_usd: Optional[float]=None,
                  brent_mensal: Optional[Dict[str, float]]=None):
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


