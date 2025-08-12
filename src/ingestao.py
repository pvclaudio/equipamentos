# src/ingestao.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from .utils import (
    normalize_ativo,
    to_float_ptbr,
    to_int_ptbr,          # ok manter, mesmo que não use em todas as bases
    salvar_parquet,
    strip_accents_lower,
)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")

def _col(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    normmap = {strip_accents_lower(c): c for c in df.columns}
    for want in candidatos:
        k = strip_accents_lower(want)
        if k in normmap:
            return normmap[k]
    return None

def run_ingestao() -> pd.DataFrame:
    fp_evt = DATA_DIR / "Eventos 2024 e 2025.xlsx"
    # Lê TUDO como texto p/ preservar vírgula e não deixar o pandas “interpretar”
    df_evt = pd.read_excel(fp_evt, sheet_name="eventos", dtype=str)

    col_campo = _col(df_evt, ["Campo", "ATIVO", "ativo"])
    col_dia   = _col(df_evt, ["Dia", "Data", "data"])
    col_h     = _col(df_evt, ["Período parado (h)", "Periodo parado (h)", "periodo_h", "horas paradas", "Tempo parado (h)"])
    col_bbl   = _col(df_evt, [
        "Perda de Produção (bbl)", "Perda de Producao (bbl)",
        "Perda de Produção", "Perda de Producao", "bbl", "barris perdidos"
    ])
    col_just  = _col(df_evt, ["Justificativa", "justificativa", "descricao", "descrição"])

    df_evt_out = pd.DataFrame()

    # Ativo e data
    if col_campo:
        df_evt_out["ativo"] = df_evt[col_campo].map(normalize_ativo)
    else:
        df_evt_out["ativo"] = None

    if col_dia:
        # ✅ substitui parse_data_ptbr por to_datetime com dayfirst
        df_evt_out["data_evento"] = pd.to_datetime(df_evt[col_dia], dayfirst=True, errors="coerce")
    else:
        df_evt_out["data_evento"] = pd.NaT

    # --- AUDITORIA: capture as strings cruas que vieram do Excel ---
    raw_h   = df_evt[col_h].astype(str)  if col_h   else pd.Series([], dtype=str)
    raw_bbl = df_evt[col_bbl].astype(str) if col_bbl else pd.Series([], dtype=str)

    # Converte com nossas ferramentas pt-BR (sem NENHUM replace de vírgula!)
    if col_h:
        horas = raw_h.apply(to_float_ptbr)
        df_evt_out["periodo_h"] = pd.to_numeric(horas, errors="coerce")
    else:
        df_evt_out["periodo_h"] = np.nan

    if col_bbl:
        bblf = raw_bbl.apply(to_float_ptbr)           # parse decimal corretamente
        df_evt_out["bbl"] = pd.to_numeric(bblf, errors="coerce").round().astype("Int64")  # inteiro (nullable)
    else:
        df_evt_out["bbl"] = pd.Series(pd.NA, index=df_evt_out.index, dtype="Int64")

    if col_just:
        df_evt_out["justificativa"] = df_evt[col_just].fillna("").astype(str)
    else:
        df_evt_out["justificativa"] = ""

    # --- AUDITORIA: salvar 20 primeiras linhas cruas vs. convertidas ---
    try:
        audit = pd.DataFrame({
            "raw_periodo_h": raw_h.head(20) if len(raw_h) else pd.Series([], dtype=str),
            "parsed_periodo_h": df_evt_out["periodo_h"].head(20),
            "raw_bbl": raw_bbl.head(20) if len(raw_bbl) else pd.Series([], dtype=str),
            "parsed_bbl": df_evt_out["bbl"].head(20),
        })
        audit.to_csv(OUT_DIR / "audit_ingestao_primeiras_linhas.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass

    salvar_parquet(df_evt_out, OUT_DIR / "eventos_base.parquet")

    # Sanidade: prints que você já usou
    print(df_evt_out.dtypes)
    print(df_evt_out[["periodo_h","bbl"]].head(12))
    print("MAX periodo_h:", df_evt_out["periodo_h"].max(), " | MAX bbl:", df_evt_out["bbl"].max())

    return df_evt_out

if __name__ == "__main__":
    run_ingestao()
