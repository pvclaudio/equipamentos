# src/agregacao.py (revisado)
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
    for c, default in (("ativo", None), ("data_evento", pd.NaT),
                       ("equipamento", "NAO_CLASSIFICADO"), ("periodo_h", np.nan),
                       ("bbl", pd.NA)):
        if c not in df.columns:
            df[c] = default

    # tipos
    df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")

    # período e bbl em numérico; bbl pode vir como Int64 → convertemos para float na sequência
    df["periodo_h"] = pd.to_numeric(df["periodo_h"], errors="coerce")
    df["bbl"] = pd.to_numeric(df["bbl"], errors="coerce")

    # sanity logs (não quebram execução)
    try:
        total = len(df)
        ok_dt = df["data_evento"].notna().sum()
        print(f"[agregacao] data_evento válidas: {ok_dt}/{total}")
    except Exception:
        pass

    return df

# =============================
# Brent
# =============================
def _normalize_brent_dict(brent_mensal: Dict[str, float]) -> Dict[str, float]:
    """Normaliza chaves do dicionário mensal para 'YYYYMM', aceitando 'YYYYMM' ou 'YYYY-MM'."""
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
    df["bbl"] = pd.to_numeric(df.get("bbl", np.nan), errors="coerce")

    # Série mensal (opcional)
    if brent_mensal:
        bm = _normalize_brent_dict(brent_mensal)
        # yyyymm_key deve tolerar NaT (retornar NaN/None sem quebrar)
        df["_yyyymm"] = yyyymm_key(df["data_evento"])
        df["brent_usd"] = df["_yyyymm"].map(bm).astype(float)
    else:
        df["brent_usd"] = np.nan

    # Fallback único
    if brent_unico_usd is None:
        brent_unico_usd = float(os.getenv("BRENT_USD_DEFAULT", "80"))
    df["brent_usd"] = df["brent_usd"].fillna(float(brent_unico_usd))

    # perda financeira = bbl * brent (assegura float)
    df["perda_financeira_usd"] = df["bbl"].fillna(0).astype(float) * df["brent_usd"].fillna(0).astype(float)

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

    # usa size() para contar eventos independentemente de nulos (equipamento/ativo já vêm preenchidos)
    agg = (
        df.groupby(["ativo", "equipamento"], dropna=False)
          .agg(
              eventos=("equipamento", "size"),
              horas_paradas_total=("periodo_h", "sum"),
              bbl_perdidos_total=("bbl", "sum"),
              perda_financeira_total_USD=("perda_financeira_usd", "sum"),
          )
          .reset_index()
          .sort_values(["ativo", "perda_financeira_total_USD"], ascending=[True, False])
    )
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
    run_agregacao(brent_unico_usd=80.0)
