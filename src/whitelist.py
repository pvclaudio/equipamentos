# src/whitelist.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Iterable, List

import pandas as pd

# Pastas
OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Fallback estático (não canibalize nomes; use como “rede de segurança”)
# ------------------------------------------------------------
_STATIC_EQUIPS = [
    "TURBINA D","TURBINA C","TURBINA B","TURBINA A","TURBINA 4","TURBINA 3","TURBINA 2","TURBINA 1",
    "MC C","MC B","MC A",
    "LP/IP B (MOD 4)","LP/IP A (MOD 3)","IGG B","IGG A","HP B (MOD 4)","HP A (MOD 3)",
    "HAWSER BE","HAWSER BB",
    "GUINDASTE SUL","GUINDASTE POPA","GUINDASTE NORTE","GUINDASTE BE","GUINDASTE BB",
    "GUINDASTE 03","GUINDASTE 02","GUINDASTE 01",
    "GERADOR DE EMERGÊNCIA","DG 6 (Agrekko)","DG 5 (Agrekko)","DG 4 (Agrekko)","DG 3 (Agrekko)","DG 2 (Agrekko)","DG 1 (Agrekko)",
    "COMPRESSOR GÁS A","COMPRESSOR DE AR D","COMPRESSOR DE AR C","COMPRESSOR DE AR B","COMPRESSOR DE AR A",
    "CALDEIRA BE","CALDEIRA BB",
    "BOTE RESGATE",
    "BOMBA INJEÇÃO B","BOMBA INJEÇÃO A",
    "BOMBA INCÊNDIO SUL","BOMBA INCÊNDIO PROA","BOMBA INCÊNDIO POPA","BOMBA INCÊNDIO NORTE",
    "BOMBA INCENDIO C","BOMBA INCENDIO B","BOMBA INCENDIO A",
    "BOMBA DE INJEÇÃO D","BOMBA DE INJEÇÃO C","BOMBA DE INJEÇÃO B","BOMBA DE INJEÇÃO A",
    "BOMBA BOOSTER C","BOMBA BOOSTER B","BOMBA BOOSTER A",
    "BB MULTIFÁSICA C","BB MULTIFÁSICA B","BB MULTIFÁSICA A",
    "BALEEIRA F","BALEEIRA E","BALEEIRA D","BALEEIRA C","BALEEIRA B","BALEEIRA A",
    "AR-CONDICIONADO VSD #6","AR-CONDICIONADO VSD #5","AR-CONDICIONADO VSD #4",
    "AR-CONDICIONADO VSD #3","AR-CONDICIONADO VSD #2","AR-CONDICIONADO VSD #1",
    "BCS","VFD","PSD","LGL","LPO","mangueira de acionamento","válvula"
]

_OVERRIDES_CSV = DATA_DIR / "whitelist_overrides.csv"
_LISTA_BDO_PARQUET = OUT_DIR / "lista_bdo_clean.parquet"
_BASE_BDOS_PARQUET = OUT_DIR / "base_bdos_clean.parquet"  # só para eventualmente extrair Top5

def _load_lista_bdo() -> pd.DataFrame:
    """Carrega a lista canônica gerada em equipamentos.py (ativo, equipamento)."""
    if _LISTA_BDO_PARQUET.exists():
        df = pd.read_parquet(_LISTA_BDO_PARQUET)
        # blindagem
        if not {"ativo", "equipamento"}.issubset(df.columns):
            return pd.DataFrame(columns=["ativo","equipamento"])
        df = df.dropna(subset=["equipamento"]).copy()
        df["equipamento"] = df["equipamento"].astype(str).str.strip()
        df["ativo"] = df["ativo"].astype(str).str.strip()
        return df[df["equipamento"] != ""]
    return pd.DataFrame(columns=["ativo","equipamento"])

def _load_overrides() -> pd.DataFrame:
    """Carrega overrides manuais/aceitas (ativo, equipamento)."""
    if _OVERRIDES_CSV.exists():
        try:
            df = pd.read_csv(_OVERRIDES_CSV, dtype=str).fillna("")
            if not {"ativo","equipamento"}.issubset(df.columns):
                return pd.DataFrame(columns=["ativo","equipamento"])
            df["ativo"] = df["ativo"].astype(str).str.strip()
            df["equipamento"] = df["equipamento"].astype(str).str.strip()
            df = df[(df["ativo"]!="") & (df["equipamento"]!="")]
            df = df.drop_duplicates(subset=["ativo","equipamento"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["ativo","equipamento"])

def _load_top5() -> pd.DataFrame:
    """Opcional: colhe itens 'top5' já limpos (se quiser uni-los ao whitelist)."""
    if _BASE_BDOS_PARQUET.exists():
        try:
            df = pd.read_parquet(_BASE_BDOS_PARQUET)
            if {"ativo", "top5"}.issubset(df.columns):
                out = []
                for _, r in df[["ativo","top5"]].fillna("").iterrows():
                    ativo = str(r["ativo"]).strip()
                    for token in re_split_top5(str(r["top5"])):
                        out.append((ativo, token))
                if out:
                    dft = pd.DataFrame(out, columns=["ativo","equipamento"])
                    dft = dft[(dft["ativo"]!="") & (dft["equipamento"]!="")]
                    dft = dft.drop_duplicates(subset=["ativo","equipamento"])
                    return dft
        except Exception:
            pass
    return pd.DataFrame(columns=["ativo","equipamento"])

def re_split_top5(text: str) -> List[str]:
    import re
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"[;,\n/|\-]+", text) if p and p.strip()]
    return parts

def _dedupe_keep(seq: Iterable[str]) -> List[str]:
    seen = set(); out = []
    for s in seq:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def build_whitelist_map(
    *,
    union_top5: bool = True,
    include_overrides: bool = True,
    include_static_backfill: bool = True
) -> Dict[str, Set[str]]:
    """
    Constrói {ativo -> set(equipamentos)} combinando:
      - lista_bdo_clean.parquet (base principal)
      - (opcional) top5 (base_bdos_clean.parquet)
      - (opcional) overrides manuais (whitelist_overrides.csv)
      - (opcional) fallback estático (_STATIC_EQUIPS) para ativos vazios
    """
    wl: Dict[str, Set[str]] = {}

    lista_bdo = _load_lista_bdo()
    if not lista_bdo.empty:
        for a, sub in lista_bdo.groupby("ativo", dropna=True):
            pool = [str(x).strip() for x in sub["equipamento"].dropna().tolist() if str(x).strip()]
            pool = _dedupe_keep(pool)
            if pool:
                wl[a] = set(pool)

    if union_top5:
        top5 = _load_top5()
        if not top5.empty:
            for a, sub in top5.groupby("ativo", dropna=True):
                pool = [str(x).strip() for x in sub["equipamento"].dropna().tolist() if str(x).strip()]
                if not pool:
                    continue
                wl.setdefault(a, set()).update(pool)

    if include_overrides:
        ovr = _load_overrides()
        if not ovr.empty:
            for a, sub in ovr.groupby("ativo", dropna=True):
                pool = [str(x).strip() for x in sub["equipamento"].dropna().tolist() if str(x).strip()]
                if not pool:
                    continue
                wl.setdefault(a, set()).update(pool)

    # reforço para ativos vazios
    if include_static_backfill:
        ativos_vistos = set(wl.keys())
        if not ativos_vistos:
            # tentar inferir ativos de eventos base/qualificados
            for fp in [OUT_DIR/"eventos_base.parquet", OUT_DIR/"eventos_qualificados.parquet"]:
                if fp.exists():
                    try:
                        d = pd.read_parquet(fp)
                        if "ativo" in d.columns:
                            ativos_vistos.update([str(x) for x in d["ativo"].dropna().unique()])
                    except Exception:
                        pass
            if not ativos_vistos:
                ativos_vistos = {"Bravo","Forte","Frade"}
        for a in ativos_vistos:
            if not wl.get(a):
                wl[a] = set(_STATIC_EQUIPS)

    return wl

def add_override(ativo: str, equipamento: str) -> None:
    """
    Adiciona (ativo,equipamento) ao CSV de overrides e deduplica.
    Use quando aceitar manualmente ou por script um novo nome fora do whitelist.
    """
    ativo = (ativo or "").strip()
    equipamento = (equipamento or "").strip()
    if not ativo or not equipamento:
        return
    df = pd.read_csv(_OVERRIDES_CSV) if _OVERRIDES_CSV.exists() else pd.DataFrame(columns=["ativo","equipamento"])
    df.loc[len(df)] = [ativo, equipamento]
    df = df.drop_duplicates(subset=["ativo","equipamento"])
    _OVERRIDES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OVERRIDES_CSV, index=False, encoding="utf-8-sig")
