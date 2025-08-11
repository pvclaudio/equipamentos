# src/matching.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from .utils import norm_text
from .equipamentos import carregar_bases, construir_dicionarios

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH_ACCEPT = 90
THRESH_AMBIG  = 85


def _match_exact_pool(mencao: str, pool: Set[str]) -> Optional[str]:
    """Match exato tolerante (normalizando)."""
    m_norm = norm_text(mencao)
    for cand in pool:
        if norm_text(cand) == m_norm:
            return cand
    return None


def _match_contains(texto_longo: str, pool: Set[str]) -> Optional[str]:
    """Verifica se algum termo do pool aparece no texto normalizado."""
    t = norm_text(texto_longo)
    for cand in sorted(pool, key=lambda s: len(s), reverse=True):
        cn = norm_text(cand)
        if cn and cn in t:
            return cand
    return None


def _match_fuzzy(mencao: str, pool: Set[str]) -> Tuple[Optional[str], float]:
    if not pool:
        return None, 0.0
    best = process.extractOne(
        query=norm_text(mencao),
        choices=[norm_text(x) for x in pool],
        scorer=fuzz.WRatio,
    )
    if best is None:
        return None, 0.0
    choice_norm, score, _ = best
    for original in pool:
        if norm_text(original) == choice_norm:
            return original, float(score)
    return None, float(score)


def _primeira_mencao_na_justificativa(justificativa: str) -> Optional[str]:
    if not isinstance(justificativa, str):
        return None
    txt = justificativa.strip()
    return txt if txt else None


def aplicar_matching() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Bases (eventos_base + BDOs) – a função já roda ingestão se faltar eventos_base
    eventos, base_bdos, lista_bdo = carregar_bases()
    dict_equip, dict_top5 = construir_dicionarios()

    # 🔒 Blindagem: garantir colunas mínimas para evitar KeyError
    required = {
        "ativo": None,
        "data_evento": pd.NaT,
        "justificativa": "",
        "periodo_h": np.nan,
        "bbl": np.nan,
    }
    for col, default in required.items():
        if col not in eventos.columns:
            eventos[col] = default
    eventos = eventos.copy()

    logs = []
    linhas_ok = []

    for idx, row in eventos.reset_index(drop=True).iterrows():
        ativo = row.get("ativo", None)
        just  = row.get("justificativa", "")

        mencao_bruta = _primeira_mencao_na_justificativa(just)
        if not ativo or not mencao_bruta:
            logs.append({
                "id_evento": idx,
                "ativo": ativo, "data_evento": row.get("data_evento", pd.NaT),
                "menção_bruta": mencao_bruta, "equipamento_candidato": None,
                "equipamento_canonizado": None, "fonte": None, "score": None,
                "status": "descartado", "regra_aplicada": "faltando_contexto",
            })
            continue

        pool_top5 = dict_top5.get(ativo, set())
        pool_dict = dict_equip.get(ativo, set())

        chosen = None
        fonte = None
        score = None
        status = None
        regra = None

        # 1) top5 emergenciais
        cand = _match_exact_pool(mencao_bruta, pool_top5)
        if cand is None:
            cand = _match_contains(mencao_bruta, pool_top5)
        if cand is not None:
            chosen = cand; fonte = "top5"; score = 100.0; status = "aceito"; regra = "top5_exato_ou_contido"
        else:
            # 2) dicionário por ativo
            cand = _match_exact_pool(mencao_bruta, pool_dict)
            if cand is None:
                cand = _match_contains(mencao_bruta, pool_dict)
            if cand is not None:
                chosen = cand; fonte = "dicionario"; score = 100.0; status = "aceito"; regra = "dict_exato_ou_contido"
            else:
                # 3) fuzzy dentro do pool
                pool_for_fuzzy = pool_dict if pool_dict else pool_top5
                cand, sc = _match_fuzzy(mencao_bruta, pool_for_fuzzy)
                if cand is not None:
                    if sc >= THRESH_ACCEPT:
                        chosen = cand; fonte = "fuzzy"; score = sc; status = "aceito"; regra = f"fuzzy>={THRESH_ACCEPT}"
                    elif THRESH_AMBIG <= sc < THRESH_ACCEPT:
                        chosen = None; fonte = "fuzzy"; score = sc; status = "ambíguo"; regra = f"{THRESH_AMBIG}<=fuzzy<{THRESH_ACCEPT}"
                    else:
                        chosen = None; fonte = "fuzzy"; score = sc; status = "descartado"; regra = f"fuzzy<{THRESH_AMBIG}"
                else:
                    chosen = None; fonte = None; score = None; status = "descartado"; regra = "sem_candidato"

        logs.append({
            "id_evento": idx,
            "ativo": ativo, "data_evento": row.get("data_evento", pd.NaT),
            "menção_bruta": mencao_bruta,
            "equipamento_candidato": chosen,
            "equipamento_canonizado": chosen,
            "fonte": fonte, "score": score, "status": status, "regra_aplicada": regra,
        })

        if status == "aceito" and chosen:
            linhas_ok.append({
                "id_evento": idx,
                "ativo": ativo,
                "data_evento": row.get("data_evento", pd.NaT),
                "equipamento": chosen,
                "periodo_h": row.get("periodo_h", np.nan),
                "bbl": row.get("bbl", np.nan),
                "justificativa": just,
            })

    log_df = pd.DataFrame(logs)
    qual_df = pd.DataFrame(linhas_ok)
    
    if not qual_df.empty:
        qual_df["periodo_h"] = pd.to_numeric(qual_df["periodo_h"], errors="coerce")
        # bbl inteiro
        qual_df["bbl"] = pd.to_numeric(qual_df["bbl"], errors="coerce").round().astype("Int64")

    # Tipos: garantir numéricos corretos
    if not qual_df.empty:
        if pd.api.types.is_timedelta64_dtype(qual_df.get("periodo_h", pd.Series([], dtype="float64"))):
            qual_df["periodo_h"] = qual_df["periodo_h"].dt.total_seconds() / 3600.0
        else:
            qual_df["periodo_h"] = pd.to_numeric(qual_df["periodo_h"], errors="coerce")
        qual_df["bbl"] = pd.to_numeric(qual_df["bbl"], errors="coerce")

    # Persistir artefatos
    log_df.to_parquet(OUT_DIR / "log_matching.parquet", index=False)
    log_df.to_csv(OUT_DIR / "log_matching.csv", index=False, encoding="utf-8-sig")
    qual_df.to_parquet(OUT_DIR / "eventos_qualificados.parquet", index=False)
    qual_df.to_csv(OUT_DIR / "eventos_qualificados.csv", index=False, encoding="utf-8-sig")

    # Auxiliares
    amb  = log_df[log_df["status"] == "ambíguo"]
    desc = log_df[log_df["status"] == "descartado"]
    amb.to_csv(OUT_DIR / "linhas_ambiguas.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUT_DIR / "linhas_descartadas.csv", index=False, encoding="utf-8-sig")

    print(f"Matching concluído. {len(qual_df)} eventos qualificados; LOG em outputs/.")
    return log_df, qual_df


if __name__ == "__main__":
    aplicar_matching()
