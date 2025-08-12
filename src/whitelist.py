# src/whitelist.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs")
DATA_DIR = Path("data")

# ------------------------------------------------------------
# Fallback estático (pode editar livremente)
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
]

# ------------------------------------------------------------
# Constrói {ativo -> set(equipamentos canônicos)}
# ------------------------------------------------------------
def build_whitelist_map() -> dict[str, set[str]]:
    wl: dict[str, set[str]] = {}

    # 1) Tenta parquet limpo gerado pelo pipeline
    fp_lista = OUT_DIR / "lista_bdo_clean.parquet"
    if fp_lista.exists():
        try:
            df = pd.read_parquet(fp_lista)
            if not df.empty and {"ativo", "equipamento"}.issubset(df.columns):
                for a, sub in df.groupby("ativo", dropna=True):
                    pool = set(str(x).strip() for x in sub["equipamento"].dropna().unique() if str(x).strip())
                    if pool:
                        wl[str(a)] = pool
        except Exception:
            pass

    # 2) Se não achou, tenta diretamente o Excel original
    if not wl:
        xls = DATA_DIR / "Lista de Equipamentos - BDO.xlsx"
        if xls.exists():
            try:
                xf = pd.ExcelFile(xls)
                for sn in ["Bravo", "Polvo", "Forte", "Frade"]:
                    if sn not in xf.sheet_names:
                        continue
                    d = pd.read_excel(xls, sheet_name=sn)
                    if d.empty:
                        continue
                    # tenta encontrar uma coluna com descrição/equipamento
                    non_empty_cols = [c for c in d.columns if d[c].notna().any()]
                    col_e = non_empty_cols[0] if non_empty_cols else d.columns[0]
                    pool = set(str(x).strip() for x in d[col_e].dropna().unique() if str(x).strip())
                    if pool:
                        ativo = "Bravo" if sn in ("Bravo", "Polvo", "TBMT") else sn
                        wl.setdefault(ativo, set()).update(pool)
            except Exception:
                pass

    # 3) Fallback estático — replica a lista para os ativos visíveis
    if not wl:
        ativos_vistos: set[str] = set()
        for fp in [OUT_DIR / "eventos_base.parquet", OUT_DIR / "eventos_qualificados.parquet"]:
            if fp.exists():
                try:
                    d = pd.read_parquet(fp)
                    if "ativo" in d.columns:
                        ativos_vistos.update([str(x) for x in d["ativo"].dropna().unique()])
                except Exception:
                    pass
        if not ativos_vistos:
            ativos_vistos = {"Bravo", "Forte", "Frade"}
        for a in ativos_vistos:
            wl[a] = set(_STATIC_EQUIPS)

    return wl
