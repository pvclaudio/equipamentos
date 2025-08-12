# src/equipamentos.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List
import re
import pandas as pd

from .utils import normalize_ativo, strip_accents_lower

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _split_top5(text: str) -> List[str]:
    """
    Quebra o campo 'top5' em itens usando separadores comuns.
    ⚠️ Não quebra por hífen para preservar tags do tipo 'Tratador-XYZ'.
    Ex.: "Bomba A; Compressores / Tratador-XYZ" -> ["Bomba A", "Compressores", "Tratador-XYZ"]
    """
    if not isinstance(text, str):
        return []
    # separadores: ; , / | (e quebras de linha)
    parts = [p.strip() for p in re.split(r"[;,\n/|]+", text) if p and p.strip()]
    # colapsa múltiplos espaços
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    return parts

def _read_excel_safe(fp: Path, sheet: str | None = None, dtype=str) -> pd.DataFrame:
    """Tenta ler a planilha 'sheet'; em caso de falha, usa a primeira disponível."""
    if not fp.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {fp}")
    try:
        return pd.read_excel(fp, sheet_name=sheet if sheet else 0, dtype=dtype)
    except Exception:
        xls = pd.ExcelFile(fp)
        if not xls.sheet_names:
            raise ValueError(f"Nenhuma planilha encontrada em {fp}")
        return pd.read_excel(fp, sheet_name=xls.sheet_names[0], dtype=dtype)

def _ensure_eventos_base() -> pd.DataFrame:
    """
    Garante a existência de outputs/eventos_base.parquet (gerado pela ingestão).
    Se não existir, tenta rodar a ingestão automaticamente.
    """
    fp_evt = OUT_DIR / "eventos_base.parquet"
    if not fp_evt.exists():
        # Import tardio para evitar import circular quando chamado via app.py
        try:
            from .ingestao import run_ingestao
            run_ingestao()
        except Exception as e:
            raise RuntimeError(f"Falha ao executar ingestão automaticamente: {e}") from e

    if not fp_evt.exists():
        raise FileNotFoundError("outputs/eventos_base.parquet não encontrado após tentar rodar a ingestão.")

    df = pd.read_parquet(fp_evt)
    return df

def _ensure_bdo_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera/Carrega os parquets limpos de BDO:
      - base_bdos_clean.parquet (a partir de data/base_bdos.xlsx, aba 'query')
      - lista_bdo_clean.parquet  (a partir de data/Lista de Equipamentos - BDO.xlsx, abas Bravo/Polvo/Forte/Frade)
    Retorna (base_bdos_clean, lista_bdo_clean)
    """
    fp_bdo_clean = OUT_DIR / "base_bdos_clean.parquet"
    fp_lista_clean = OUT_DIR / "lista_bdo_clean.parquet"

    # -------- base_bdos (aba 'query') --------
    if fp_bdo_clean.exists():
        base_bdos = pd.read_parquet(fp_bdo_clean)
    else:
        xls_bdo = DATA_DIR / "base_bdos.xlsx"
        base_bdos = _read_excel_safe(xls_bdo, sheet="query", dtype=str)

        # normaliza ativo (Bravo≡Polvo≡TBMT; Forte≡ABL)
        if "ativo" in base_bdos.columns:
            base_bdos["ativo"] = base_bdos["ativo"].map(normalize_ativo)
        else:
            base_bdos["ativo"] = None

        # localizar coluna do top5 com tolerância a nomes
        # exemplos: top_5_itens_emergenciais / top5 / top_5 / emergenciais
        def _norm(s: str) -> str: return strip_accents_lower(s)
        cols_norm = {_norm(c): c for c in base_bdos.columns}
        want_key = None
        for k in ("top_5_itens_emergenciais", "top5", "top_5", "emergenciais", "top_itens_emergenciais"):
            if k in cols_norm:
                want_key = cols_norm[k]
                break

        base_bdos["top5"] = base_bdos[want_key].astype(str) if want_key else ""
        base_bdos["top5"] = base_bdos["top5"].fillna("").astype(str)

        base_bdos.to_parquet(fp_bdo_clean, index=False)

    # -------- Lista de Equipamentos – BDO (todas as abas) --------
    if fp_lista_clean.exists():
        lista_bdo = pd.read_parquet(fp_lista_clean)
    else:
        xls_lista = DATA_DIR / "Lista de Equipamentos - BDO.xlsx"
        sheets_want = ["Bravo", "Polvo", "Forte", "Frade"]
        frames: List[pd.DataFrame] = []
        xls = pd.ExcelFile(xls_lista)
        for sn in sheets_want:
            if sn not in xls.sheet_names:
                continue
            df = pd.read_excel(xls, sheet_name=sn, dtype=str)

            # mapear colunas com tolerância a nomes
            cols = {strip_accents_lower(c): c for c in df.columns}

            col_ativo = None
            for k in ("ativo", "campo", "campo/ativo", "asset"):
                if k in cols:
                    col_ativo = cols[k]
                    break

            col_equip = None
            for k in ("equipamento", "equipamentos", "tag", "descricao", "descrição", "equip"):
                if k in cols:
                    col_equip = cols[k]
                    break
            if col_equip is None:
                # fallback: primeira coluna não totalmente vazia
                non_empty_cols = [c for c in df.columns if df[c].notna().any()]
                if non_empty_cols:
                    col_equip = non_empty_cols[0]
            if col_equip is None:
                continue

            tmp = pd.DataFrame()
            tmp["equipamento"] = df[col_equip].fillna("").astype(str).str.strip()
            # colapsa espaços internos
            tmp["equipamento"] = tmp["equipamento"].str.replace(r"\s+", " ", regex=True)
            # remove linhas vazias
            tmp = tmp[tmp["equipamento"] != ""]

            tmp["ativo"] = df[col_ativo].map(normalize_ativo) if col_ativo else normalize_ativo(sn)
            frames.append(tmp)

        if frames:
            lista_bdo = pd.concat(frames, ignore_index=True)
        else:
            lista_bdo = pd.DataFrame(columns=["ativo", "equipamento"])

        # normaliza ativo e dedup case-insensitive preservando a forma original
        lista_bdo["ativo"] = lista_bdo["ativo"].map(normalize_ativo)
        if not lista_bdo.empty:
            # chave para dedupe: (ativo_norm, equip_norm)
            key = (
                lista_bdo["ativo"].astype(str).str.strip().str.lower()
                + "||"
                + lista_bdo["equipamento"].astype(str).str.strip().str.lower()
            )
            lista_bdo = lista_bdo.loc[~key.duplicated()].reset_index(drop=True)

        lista_bdo.to_parquet(fp_lista_clean, index=False)

    return base_bdos, lista_bdo

# ------------------------------------------------------------
# API pública consumida pelo matching
# ------------------------------------------------------------
def carregar_bases() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega:
      - outputs/eventos_base.parquet (garante a existência; roda ingestão se faltar)
      - outputs/base_bdos_clean.parquet (gera se não existir)
      - outputs/lista_bdo_clean.parquet (gera se não existir)
    """
    eventos = _ensure_eventos_base()
    base_bdos, lista_bdo = _ensure_bdo_clean()
    return eventos, base_bdos, lista_bdo

def construir_dicionarios() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Retorna:
      - dict_equipamentos[ativo] = {equipamentos da Lista BDO}
      - dict_top5[ativo]        = {itens do top_5_itens_emergenciais (coluna 'top5')}
    Observações:
      - Deduplicação case-insensitive, preservando a grafia original como canônica.
      - Itens vazios ou com 1 char são descartados.
    """
    _, base_bdos, lista_bdo = carregar_bases()

    # ---------- dicionário de equipamentos ----------
    dict_equip: Dict[str, Set[str]] = {}
    if not lista_bdo.empty and {"ativo", "equipamento"}.issubset(lista_bdo.columns):
        for ativo, sub in lista_bdo.groupby("ativo", dropna=True):
            pool_raw = [str(x).strip() for x in sub["equipamento"].dropna().tolist()]
            pool_raw = [re.sub(r"\s+", " ", x) for x in pool_raw if x]
            # dedupe case-insensitive preservando a 1ª ocorrência
            seen_ci = set()
            pool: List[str] = []
            for x in pool_raw:
                k = x.lower()
                if k not in seen_ci and len(x) >= 2:
                    pool.append(x)
                    seen_ci.add(k)
            if pool:
                dict_equip[ativo] = set(pool)

    # ---------- dicionário do top5 ----------
    dict_top5: Dict[str, Set[str]] = {}
    if "top5" in base_bdos.columns:
        for ativo, sub in base_bdos.groupby("ativo", dropna=True):
            bucket: List[str] = []
            for item in sub["top5"].dropna().astype(str):
                for tok in _split_top5(item):
                    if tok:
                        bucket.append(tok)
            # limpeza + dedupe case-insensitive
            seen_ci = set()
            clean_bucket: List[str] = []
            for b in (re.sub(r"\s+", " ", x).strip() for x in bucket if x):
                if len(b) < 2:
                    continue
                k = b.lower()
                if k not in seen_ci:
                    clean_bucket.append(b)
                    seen_ci.add(k)
            if clean_bucket:
                dict_top5[ativo] = set(clean_bucket)

    # ---------- snapshots para auditoria ----------
    try:
        snap_equip = pd.DataFrame([{"ativo": a, "equipamento": e} for a, ss in dict_equip.items() for e in sorted(ss)])
        snap_top5  = pd.DataFrame([{"ativo": a, "item": e}         for a, ss in dict_top5.items() for e in sorted(ss)])
        if not snap_equip.empty:
            snap_equip.to_csv(OUT_DIR / "snapshot_dict_equipamentos.csv", index=False, encoding="utf-8-sig")
        if not snap_top5.empty:
            snap_top5.to_csv(OUT_DIR / "snapshot_dict_top5.csv", index=False, encoding="utf-8-sig")
    except Exception:
        # snapshots são best-effort
        pass

    return dict_equip, dict_top5
