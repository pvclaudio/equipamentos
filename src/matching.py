# src/matching.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
import os
import json
import re

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from .utils import norm_text
from .equipamentos import carregar_bases, construir_dicionarios
from .whitelist import build_whitelist_map  # whitelist unificado

# Integração opcional com interpretador + revisor (continua compatível)
try:
    from .agent_utils import classificar_com_revisao, NAO_CLASSIFICADO
    _REVIEW_AVAILABLE = True
except Exception:
    classificar_com_revisao = None  # type: ignore
    NAO_CLASSIFICADO = "NAO_CLASSIFICADO"
    _REVIEW_AVAILABLE = False

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH_ACCEPT = 90
THRESH_AMBIG  = 85

# =============================
# Regras (data/rules_map.csv)
# =============================
def _load_rules_map(path: Path = Path("data/rules_map.csv")) -> list[tuple[str, re.Pattern, str]]:
    rules: list[tuple[str, re.Pattern, str]] = []
    if not path.exists():
        return rules
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, r in df.iterrows():
            ativo = (r.get("ativo", "") or "*").strip()
            rx    = (r.get("regex", "") or "").strip()
            equip = (r.get("equipamento", "") or "").strip()
            if rx and equip:
                try:
                    rules.append((ativo, re.compile(rx, re.IGNORECASE), equip))
                except re.error:
                    continue
    except Exception:
        pass
    return rules

def _apply_rules(ativo: str, justificativa: str, rules: list[tuple[str, re.Pattern, str]]) -> Optional[str]:
    a = (ativo or "").strip()
    txt = (justificativa or "").strip()
    if not txt:
        return None
    for alvo, rx, equip in rules:
        if alvo in ("*", a):
            if rx.search(txt):
                return equip
    return None

# =============================
# Helpers de normalização
# =============================
def _build_norm_maps(pool: Set[str]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    map_norm2orig: Dict[str, str] = {}
    list_orig_norm: List[Tuple[str, str]] = []
    for orig in pool:
        n = norm_text(orig)
        if n and n not in map_norm2orig:
            map_norm2orig[n] = orig
            list_orig_norm.append((orig, n))
    list_orig_norm.sort(key=lambda t: len(t[1]), reverse=True)
    return map_norm2orig, list_orig_norm

def _match_exact_pool(mencao: str, map_norm2orig: Dict[str, str]) -> Optional[str]:
    m_norm = norm_text(mencao)
    return map_norm2orig.get(m_norm)

def _match_contains(texto_longo: str, list_orig_norm_sorted: List[Tuple[str, str]]) -> Optional[str]:
    t = norm_text(texto_longo)
    if not t:
        return None
    for orig, cn in list_orig_norm_sorted:
        if cn and cn in t:
            return orig
    return None

def _match_fuzzy(mencao: str, map_norm2orig: Dict[str, str]) -> Tuple[Optional[str], float]:
    if not map_norm2orig:
        return None, 0.0
    query_n = norm_text(mencao)
    if not query_n:
        return None, 0.0
    choices_norm = list(map_norm2orig.keys())
    best = process.extractOne(query=query_n, choices=choices_norm, scorer=fuzz.WRatio)
    if best is None:
        return None, 0.0
    choice_norm, score, _ = best
    return map_norm2orig.get(choice_norm), float(score)

def _primeira_mencao_na_justificativa(justificativa: str) -> Optional[str]:
    if not isinstance(justificativa, str):
        return None
    txt = justificativa.strip()
    return txt if txt else None

# =============================
# Pipeline de matching
# =============================
def aplicar_matching(
    *,
    use_review_pipeline: bool = False,
    keep_unclassified: bool = True,      # <<< padrão: manter NAO_CLASSIFICADO para garantir presença do ativo
    limiar_conf: float = 0.60,
    agent_model: str = "gpt-4o",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ordem:
      0) rules_map.csv (regras específicas)
      1) Top5 (exato/contido)
      2) Dicionário por ativo (exato/contido)
      3) Fuzzy (aceito se score >= THRESH_ACCEPT; entre THRESH_AMBIG e THRESH_ACCEPT => ambíguo)
      4) (opcional) AGENTE + REVISOR como fallback
      5) Se ainda assim nada: grava NAO_CLASSIFICADO quando keep_unclassified=True
    """
    if not use_review_pipeline:
        use_review_pipeline = os.getenv("USE_REVIEW_PIPELINE", "").strip().lower() in ("1","true","yes","on")

    eventos, base_bdos, lista_bdo = carregar_bases()
    dict_equip, dict_top5 = construir_dicionarios()

    required = {"ativo": None, "data_evento": pd.NaT, "justificativa": "", "periodo_h": np.nan, "bbl": np.nan}
    for col, default in required.items():
        if col not in eventos.columns:
            eventos[col] = default
    eventos = eventos.copy()

    logs: List[dict] = []
    linhas_ok: List[dict] = []

    rules = _load_rules_map()
    whitelist_map: Dict[str, Set[str]] = build_whitelist_map(
        union_top5=True, include_overrides=True, include_static_backfill=True,
    )

    cache_top5_norm: Dict[str, Tuple[Dict[str, str], List[Tuple[str, str]]]] = {}
    cache_dict_norm: Dict[str, Tuple[Dict[str, str], List[Tuple[str, str]]]] = {}

    def _get_norm_structs(ativo: str):
        if ativo not in cache_top5_norm:
            pool_top5 = dict_top5.get(ativo, set())
            cache_top5_norm[ativo] = _build_norm_maps(pool_top5)
        if ativo not in cache_dict_norm:
            pool_dict = dict_equip.get(ativo, set())
            cache_dict_norm[ativo] = _build_norm_maps(pool_dict)
        return cache_top5_norm[ativo], cache_dict_norm[ativo]

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
            if keep_unclassified:
                linhas_ok.append({
                    "id_evento": idx, "ativo": ativo, "data_evento": row.get("data_evento", pd.NaT),
                    "equipamento": NAO_CLASSIFICADO,
                    "periodo_h": row.get("periodo_h", np.nan),
                    "bbl": row.get("bbl", np.nan),
                    "justificativa": just,
                })
            continue

        (top5_map, top5_list), (dict_map, dict_list) = _get_norm_structs(ativo)

        chosen = None
        fonte = None
        score = None
        status = None
        regra = None

        # 0) Regras
        rule_hit = _apply_rules(ativo, just, rules)
        if rule_hit:
            chosen = rule_hit; fonte = "rule_map"; score = 100.0; status = "aceito"; regra = "rule_map"

        # 1) Top‑5
        if chosen is None:
            cand = _match_exact_pool(mencao_bruta, top5_map)
            if cand is None:
                cand = _match_contains(mencao_bruta, top5_list)
            if cand is not None:
                chosen = cand; fonte = "top5"; score = 100.0; status = "aceito"; regra = "top5_exato_ou_contido"

        # 2) Dicionário
        if chosen is None:
            cand = _match_exact_pool(mencao_bruta, dict_map)
            if cand is None:
                cand = _match_contains(mencao_bruta, dict_list)
            if cand is not None:
                chosen = cand; fonte = "dicionario"; score = 100.0; status = "aceito"; regra = "dict_exato_ou_contido"

        # 3) Fuzzy
        if chosen is None:
            cand, sc = _match_fuzzy(mencao_bruta, dict_map if dict_map else top5_map)
            if cand is not None:
                if sc >= THRESH_ACCEPT:
                    chosen = cand; fonte = "fuzzy"; score = sc; status = "aceito"; regra = f"fuzzy>={THRESH_ACCEPT}"
                elif THRESH_AMBIG <= sc < THRESH_ACCEPT:
                    chosen = None; fonte = "fuzzy"; score = sc; status = "ambíguo"; regra = f"{THRESH_AMBIG}<=fuzzy<{THRESH_ACCEPT}"
                else:
                    chosen = None; fonte = "fuzzy"; score = sc; status = "descartado"; regra = f"fuzzy<{THRESH_AMBIG}"
            else:
                chosen = None; fonte = None; score = None; status = "descartado"; regra = "sem_candidato"

        # 4) Agente + revisor (opcional)
        meta_review = {}
        if chosen is None and use_review_pipeline:
            if not _REVIEW_AVAILABLE:
                meta_review = {
                    "origem_classificacao": "interpretador+revisor:NOK",
                    "confianca": 0.0,
                    "motivo": "pipeline indisponível",
                    "proposta_bruta": "",
                }
            else:
                try:
                    found_list, meta = classificar_com_revisao(
                        justificativa=just,
                        ativo=ativo,
                        whitelist_map=whitelist_map,
                        model_interpretador=agent_model,
                        model_revisor=agent_model,
                        limiar_conf=limiar_conf,
                    )
                    conf = float(meta.get("confianca", 0.0))
                    meta_review = {
                        "origem_classificacao": meta.get("origem_classificacao", ""),
                        "confianca": conf,
                        "motivo": meta.get("motivo", ""),
                        "proposta_bruta": meta.get("proposta_bruta", ""),
                    }
                    if found_list and found_list[0] and found_list[0] != NAO_CLASSIFICADO and conf >= float(limiar_conf):
                        chosen = found_list[0]
                        fonte = "interpretador+revisor"
                        score = conf * 100.0
                        status = "aceito"
                        regra = f"revisor_conf>={limiar_conf}"
                except Exception as e:
                    meta_review = {
                        "origem_classificacao": "interpretador+revisor:erro",
                        "confianca": 0.0,
                        "motivo": f"exceção: {e}",
                        "proposta_bruta": "",
                    }

        # LOG
        log_rec = {
            "id_evento": idx,
            "ativo": ativo, "data_evento": row.get("data_evento", pd.NaT),
            "menção_bruta": mencao_bruta,
            "equipamento_candidato": chosen,
            "equipamento_canonizado": chosen,
            "fonte": fonte, "score": score, "status": status, "regra_aplicada": regra,
        }
        if meta_review:
            log_rec.update({
                "origem_classificacao": meta_review.get("origem_classificacao", ""),
                "confianca": meta_review.get("confianca", np.nan),
                "motivo": meta_review.get("motivo", ""),
                "proposta_bruta": json.dumps(meta_review.get("proposta_bruta", ""), ensure_ascii=False)
                    if isinstance(meta_review.get("proposta_bruta"), (dict, list)) else meta_review.get("proposta_bruta", ""),
            })
        logs.append(log_rec)

        # Saída (aceitos) OU NAO_CLASSIFICADO quando habilitado
        if status == "aceito" and chosen:
            payload = {
                "id_evento": idx,
                "ativo": ativo,
                "data_evento": row.get("data_evento", pd.NaT),
                "equipamento": chosen,
                "periodo_h": row.get("periodo_h", np.nan),
                "bbl": row.get("bbl", np.nan),
                "justificativa": just,
            }
            if meta_review:
                payload.update({k: log_rec.get(k) for k in ("origem_classificacao","confianca","motivo","proposta_bruta")})
            linhas_ok.append(payload)
        else:
            if keep_unclassified:
                payload = {
                    "id_evento": idx,
                    "ativo": ativo,
                    "data_evento": row.get("data_evento", pd.NaT),
                    "equipamento": NAO_CLASSIFICADO,
                    "periodo_h": row.get("periodo_h", np.nan),
                    "bbl": row.get("bbl", np.nan),
                    "justificativa": just,
                }
                if meta_review:
                    payload.update({k: log_rec.get(k) for k in ("origem_classificacao","confianca","motivo","proposta_bruta")})
                linhas_ok.append(payload)

    log_df = pd.DataFrame(logs)
    qual_df = pd.DataFrame(linhas_ok)

    if not qual_df.empty:
        if pd.api.types.is_timedelta64_dtype(qual_df.get("periodo_h", pd.Series([], dtype="float64"))):
            qual_df["periodo_h"] = qual_df["periodo_h"].dt.total_seconds() / 3600.0
        else:
            qual_df["periodo_h"] = pd.to_numeric(qual_df["periodo_h"], errors="coerce")
        if "bbl" in qual_df.columns:
            qual_df["bbl"] = pd.to_numeric(qual_df["bbl"], errors="coerce").round().astype("Int64")

    log_df.to_parquet(OUT_DIR / "log_matching.parquet", index=False)
    log_df.to_csv(OUT_DIR / "log_matching.csv", index=False, encoding="utf-8-sig")
    qual_df.to_parquet(OUT_DIR / "eventos_qualificados.parquet", index=False)
    qual_df.to_csv(OUT_DIR / "eventos_qualificados.csv", index=False, encoding="utf-8-sig")

    log_df[log_df["status"] == "ambíguo"].to_csv(OUT_DIR / "linhas_ambiguas.csv", index=False, encoding="utf-8-sig")
    log_df[log_df["status"] == "descartado"].to_csv(OUT_DIR / "linhas_descartadas.csv", index=False, encoding="utf-8-sig")

    print(f"Matching concluído. {len(qual_df)} linhas gravadas (inclui NAO_CLASSIFICADO={keep_unclassified}). "
          f"Revisor: {'ON' if use_review_pipeline else 'OFF'}")
    return log_df, qual_df


if __name__ == "__main__":
    aplicar_matching()
