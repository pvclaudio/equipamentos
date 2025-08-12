# src/whitelist.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Iterable
import pandas as pd

OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "BCS","VFD","PSD","LGL","LPO","mangueira de acionamento","válvula"
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ci_set_preservando_primeiro(it: Iterable[str]) -> Set[str]:
    """
    Dedup case-insensitive preservando a primeira grafia original.
    """
    seen = set()
    out = []
    for x in it:
        if not x:
            continue
        k = str(x).strip()
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
    return set(out)

def _read_overrides_csv() -> Dict[str, Set[str]]:
    """
    Lê overrides opcionais em data/whitelist_overrides.csv com colunas: ativo,equipamento.
    """
    fp = DATA_DIR / "whitelist_overrides.csv"
    if not fp.exists():
        return {}
    try:
        df = pd.read_csv(fp, dtype=str)
        if not {"ativo", "equipamento"}.issubset(df.columns):
            return {}
        df["ativo"] = df["ativo"].fillna("").astype(str).str.strip()
        df["equipamento"] = df["equipamento"].fillna("").astype(str).str.strip()
        out: Dict[str, Set[str]] = {}
        for a, sub in df.groupby("ativo"):
            items = [e for e in sub["equipamento"].tolist() if e]
            if items:
                out[a] = _ci_set_preservando_primeiro(items)
        return out
    except Exception:
        return {}

# ------------------------------------------------------------
# Fonte preferencial: equipamentos + (opcional) top5 via equipamentos.py
# ------------------------------------------------------------
def _from_equipamentos(union_top5: bool = True) -> Dict[str, Set[str]]:
    try:
        from .equipamentos import construir_dicionarios
        dict_equip, dict_top5 = construir_dicionarios()
        wl: Dict[str, Set[str]] = {}
        for ativo, pool in dict_equip.items():
            wl[ativo] = _ci_set_preservando_primeiro(pool)
        if union_top5:
            for ativo, pool in dict_top5.items():
                wl.setdefault(ativo, set())
                wl[ativo] = _ci_set_preservando_primeiro(list(wl[ativo]) + list(pool))
        return wl
    except Exception:
        return {}

# ------------------------------------------------------------
# Fallback secundário: outputs/lista_bdo_clean.parquet
# ------------------------------------------------------------
def _from_parquet_lista() -> Dict[str, Set[str]]:
    fp_lista = OUT_DIR / "lista_bdo_clean.parquet"
    wl: Dict[str, Set[str]] = {}
    if not fp_lista.exists():
        return wl
    try:
        df = pd.read_parquet(fp_lista)
        if not df.empty and {"ativo", "equipamento"}.issubset(df.columns):
            for a, sub in df.groupby("ativo"):
                items = [str(x).strip() for x in sub["equipamento"].dropna().unique().tolist() if str(x).strip()]
                if items:
                    wl[a] = _ci_set_preservando_primeiro(items)
    except Exception:
        pass
    return wl

# ------------------------------------------------------------
# Constrói {ativo -> set(equipamentos canônicos)}
# ------------------------------------------------------------
def build_whitelist_map(
    *,
    union_top5: bool = True,
    include_overrides: bool = True,
    include_static_backfill: bool = True
) -> dict[str, set[str]]:
    """
    Constrói o whitelist por ativo com múltiplas fontes (prioridade):
      1) equipamentos.construir_dicionarios()  [preferencial]
         - se union_top5=True, une também os itens do Top-5 emergenciais
      2) outputs/lista_bdo_clean.parquet       [fallback]
      3) overrides de data/whitelist_overrides.csv (sempre adiciona se include_overrides=True)
      4) backfill estático _STATIC_EQUIPS por ativo visto (ou {'Bravo','Forte','Frade'})

    Parâmetros:
      - union_top5: incluir itens do Top-5 no whitelist (recomendado)
      - include_overrides: ler e aplicar overrides do CSV
      - include_static_backfill: usar lista estática quando algum ativo ficar vazio
    """
    # 1) fonte preferencial
    wl = _from_equipamentos(union_top5=union_top5)

    # 2) fallback parquet se a fonte preferencial falhar
    if not wl:
        wl = _from_parquet_lista()

    # 3) overrides manuais
    if include_overrides:
        overrides = _read_overrides_csv()
        for a, items in overrides.items():
            wl.setdefault(a, set())
            wl[a] = _ci_set_preservando_primeiro(list(wl[a]) + list(items))

    # 4) backfill estático por ativo visto nos outputs; último recurso usa ativos padrão
    if include_static_backfill:
        # quais ativos temos nos dados?
        ativos_vistos = set()
        for fp in [OUT_DIR / "eventos_base.parquet", OUT_DIR / "eventos_qualificados.parquet"]:
            if fp.exists():
                try:
                    d = pd.read_parquet(fp)
                    if "ativo" in d.columns:
                        ativos_vistos.update([str(x) for x in d["ativo"].dropna().unique().tolist()])
                except Exception:
                    pass
        if not ativos_vistos:
            ativos_vistos = {"Bravo", "Forte", "Frade"}

        # aplica estático para ativos não presentes ou vazios
        for a in ativos_vistos:
            if a not in wl or not wl[a]:
                wl[a] = _ci_set_preservando_primeiro(_STATIC_EQUIPS)

    # sanity: garante que todo set seja de strings limpas
    for a in list(wl.keys()):
        wl[a] = _ci_set_preservando_primeiro([str(x).strip() for x in wl[a] if str(x).strip()])

    return wl

# ------------------------------------------------------------
# CLI/Snapshot
# ------------------------------------------------------------
if __name__ == "__main__":
    wl = build_whitelist_map()
    snap = pd.DataFrame([{"ativo": a, "equipamento": e} for a, ss in wl.items() for e in sorted(ss)])
    snap.to_csv(OUT_DIR / "whitelist_snapshot.csv", index=False, encoding="utf-8-sig")
    print(f"Whitelist gerado para {len(wl)} ativos; snapshot em outputs/whitelist_snapshot.csv")
