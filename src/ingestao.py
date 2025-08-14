from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from .utils import (
    normalize_ativo,
    to_float_ptbr,
    to_int_ptbr,          # ok manter
    salvar_parquet,
    strip_accents_lower,
)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============== Helpers ===============

def _col(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    """Retorna o nome real da coluna que corresponde a algum dos `candidatos` (case/acento-insensitive)."""
    normmap = {strip_accents_lower(c): c for c in df.columns}
    for want in candidatos:
        k = strip_accents_lower(want)
        if k in normmap:
            return normmap[k]
    return None

def _read_excel_eventos(fp: Path) -> pd.DataFrame:
    """
    Lê Excel como texto (dtype=str) para preservar vírgulas e formatações.
    Tenta a planilha 'eventos' e, se não existir, usa a primeira planilha.
    """
    if not fp.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fp}")
    try:
        return pd.read_excel(fp, sheet_name="eventos", dtype=str)
    except Exception:
        xls = pd.ExcelFile(fp)
        if not xls.sheet_names:
            raise ValueError(f"Nenhuma planilha encontrada em {fp}")
        return pd.read_excel(fp, sheet_name=xls.sheet_names[0], dtype=str)

def _parse_date_universal(series: pd.Series) -> pd.Series:
    """
    Parser universal para datas:
      1) to_datetime(dayfirst=False) → ISO (YYYY-MM-DD[ HH:MM[:SS]])
      2) fallback to_datetime(dayfirst=True) → dd/mm/yyyy[ HH:MM[:SS]]
      3) fallback número serial do Excel (origin 1899-12-30)
    Retorna datetime64[ns] (naive) com NaT quando não conseguir parsear.
    """
    s = series.astype(str).str.strip()

    # Passo 1: ISO / formatos ano-mês-dia
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False)

    # Passo 2: dd/mm/yyyy (só onde falhou)
    mask2 = dt1.isna()
    dt2 = pd.to_datetime(s.where(mask2), errors="coerce", dayfirst=True)

    # Passo 3: serial do Excel (só onde ainda falhou)
    mask3 = dt1.combine_first(dt2).isna()
    num = pd.to_numeric(s.where(mask3), errors="coerce")
    dt3 = pd.to_datetime(num, unit="d", origin="1899-12-30", errors="coerce")

    out = dt1.combine_first(dt2).combine_first(dt3)

    # Remove timezone se vier algo com tz por acidente
    try:
        out = out.dt.tz_localize(None)
    except Exception:
        pass

    return out

# =============== Pipeline ===============

def run_ingestao() -> pd.DataFrame:
    """
    Lê 'data/Eventos 2024 e 2025.xlsx', normaliza e salva 'outputs/eventos_base.parquet'.
    Retorna o DataFrame pronto para os próximos estágios.
    """
    fp_evt = DATA_DIR / "Eventos 2024 e 2025.xlsx"
    df_evt = _read_excel_eventos(fp_evt)

    # Mapeamento tolerante de colunas
    col_campo = _col(df_evt, ["Campo", "ATIVO", "ativo"])
    col_dia   = _col(df_evt, ["Dia", "Data", "data", "data do evento", "data_ocorrencia"])
    col_h     = _col(df_evt, ["Período parado (h)", "Periodo parado (h)", "periodo_h", "horas paradas", "Tempo parado (h)"])
    col_bbl   = _col(df_evt, [
        "Perda de Produção (bbl)", "Perda de Producao (bbl)",
        "Perda de Produção", "Perda de Producao", "bbl", "barris perdidos"
    ])
    col_just  = _col(df_evt, ["Justificativa", "justificativa", "descricao", "descrição", "descricao do evento"])

    # DataFrame de saída
    df_evt_out = pd.DataFrame(index=df_evt.index)

    # Ativo
    df_evt_out["ativo"] = df_evt[col_campo].map(normalize_ativo) if col_campo else None

    # Datas (universal)
    if col_dia:
        df_evt_out["data_evento"] = _parse_date_universal(df_evt[col_dia])
    else:
        df_evt_out["data_evento"] = pd.NaT

    # Números: horas e bbl
    raw_h   = df_evt[col_h].astype(str)   if col_h   else pd.Series([], dtype=str)
    raw_bbl = df_evt[col_bbl].astype(str) if col_bbl else pd.Series([], dtype=str)

    if col_h:
        df_evt_out["periodo_h"] = pd.to_numeric(raw_h.apply(to_float_ptbr), errors="coerce")
    else:
        df_evt_out["periodo_h"] = np.nan

    if col_bbl:
        bblf = pd.to_numeric(raw_bbl.apply(to_float_ptbr), errors="coerce").round()
        df_evt_out["bbl"] = bblf.astype("Int64")
    else:
        df_evt_out["bbl"] = pd.Series(pd.NA, index=df_evt_out.index, dtype="Int64")

    # Justificativa
    df_evt_out["justificativa"] = df_evt[col_just].fillna("").astype(str) if col_just else ""

    # Auditoria rápida
    try:
        audit_cols = {}
        if col_dia:
            audit_cols["raw_data"] = df_evt[col_dia].head(30)
            audit_cols["parsed_data_evento"] = df_evt_out["data_evento"].head(30)
        if len(raw_h):
            audit_cols["raw_periodo_h"] = raw_h.head(20)
            audit_cols["parsed_periodo_h"] = df_evt_out["periodo_h"].head(20)
        if len(raw_bbl):
            audit_cols["raw_bbl"] = raw_bbl.head(20)
            audit_cols["parsed_bbl"] = df_evt_out["bbl"].head(20)
        if audit_cols:
            pd.DataFrame(audit_cols).to_csv(
                OUT_DIR / "audit_ingestao_primeiras_linhas.csv",
                index=False,
                encoding="utf-8-sig",
            )
    except Exception:
        pass

    # Snapshot de linhas com NaT (se houver)
    try:
        mask_nat = df_evt_out["data_evento"].isna()
        if mask_nat.any() and col_dia:
            pd.DataFrame({"raw_data": df_evt[col_dia][mask_nat]}).head(200).to_csv(
                OUT_DIR / "audit_ingestao_datas_invalidas.csv",
                index=False,
                encoding="utf-8-sig",
            )
    except Exception:
        pass

    # Persistência
    salvar_parquet(df_evt_out, OUT_DIR / "eventos_base.parquet")

    # Prints de sanidade (log)
    try:
        print(df_evt_out.dtypes)
        print(df_evt_out[["data_evento","periodo_h","bbl"]].head(10))
        print("Datas válidas:", df_evt_out["data_evento"].notna().sum())
        print("MAX periodo_h:", df_evt_out["periodo_h"].max(), " | MAX bbl:", df_evt_out["bbl"].max())
    except Exception:
        pass

    return df_evt_out

if __name__ == "__main__":
    run_ingestao()
