# src/matching.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
import os
import json

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from .utils import norm_text
from .equipamentos import carregar_bases, construir_dicionarios

# Integração opcional com o pipeline interpretador+revisor
# (mantém compat: se não existir, segue o comportamento antigo)
try:
    from .agent_utils import classificar_com_revisao, NAO_CLASSIFICADO
    _REVIEW_AVAILABLE = True
except Exception:
    classificar_com_revisao = None  # type: ignore
    NAO_CLASSIFICADO = "NAO_CLASSIFICADO"  # fallback
    _REVIEW_AVAILABLE = False

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limiar de aceitação/ambiguidade para fuzzy
THRESH_ACCEPT = 90
THRESH_AMBIG  = 85


# =========================
# Helpers de normalização
# =========================
def _build_norm_maps(pool: Set[str]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    A partir do pool original, retorna:
      - map_norm2orig: {texto_normalizado: original}
      - list_orig_norm_sorted: [(orig, norm)] ordenado por len(norm) desc (para 'contains')
    Se houver colisões de normalização, preserva a 1ª ocorrência.
    """
    map_norm2orig: Dict[str, str] = {}
    list_orig_norm: List[Tuple[str, str]] = []
    for orig in pool:
        n = norm_text(orig)
        if n and n not in map_norm2orig:
            map_norm2orig[n] = orig
            list_orig_norm.append((orig, n))
    # ordenar por tamanho do normalizado (desc) para 'contains' priorizar termos mais específicos
    list_orig_norm.sort(key=lambda t: len(t[1]), reverse=True)
    return map_norm2orig, list_orig_norm


def _match_exact_pool(mencao: str, map_norm2orig: Dict[str, str]) -> Optional[str]:
    """Match exato tolerante (normalizando)."""
    m_norm = norm_text(mencao)
    if not m_norm:
        return None
    return map_norm2orig.get(m_norm)


def _match_contains(texto_longo: str, list_orig_norm_sorted: List[Tuple[str, str]]) -> Optional[str]:
    """Verifica se algum termo do pool aparece contido no texto normalizado (prioriza termos mais longos)."""
    t = norm_text(texto_longo)
    if not t:
        return None
    for orig, cn in list_orig_norm_sorted:
        if cn and cn in t:
            return orig
    return None


def _match_fuzzy(mencao: str, map_norm2orig: Dict[str, str]) -> Tuple[Optional[str], float]:
    """Fuzzy match usando RapidFuzz sobre os NORMALIZADOS, mapeando de volta ao original."""
    if not map_norm2orig:
        return None, 0.0
    query_n = norm_text(mencao)
    if not query_n:
        return None, 0.0
    choices_norm = list(map_norm2orig.keys())
    best = process.extractOne(
        query=query_n,
        choices=choices_norm,
        scorer=fuzz.WRatio,
    )
    if best is None:
        return None, 0.0
    choice_norm, score, _ = best
    return map_norm2orig.get(choice_norm), float(score)


def _primeira_mencao_na_justificativa(justificativa: str) -> Optional[str]:
    """Por ora, retorna a justificativa inteira (gancho para evolução futura de extração de entidade)."""
    if not isinstance(justificativa, str):
        return None
    txt = justificativa.strip()
    return txt if txt else None


# =========================
# Pipeline de matching
# =========================
def aplicar_matching(
    *,
    use_review_pipeline: bool = False,
    keep_unclassified: bool = False,
    limiar_conf: float = 0.60,
    agent_model: str = "gpt-4o",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o matching clássico e, opcionalmente, cai no pipeline interpretador+revisor
    quando não houver match por whitelist/top5/dicionário.

    Args:
        use_review_pipeline: ativa interpretador+revisor quando não houver match clássico.
                             Também pode ser habilitado por env var USE_REVIEW_PIPELINE=1.
        keep_unclassified: inclui NAO_CLASSIFICADO quando revisor não mapear com confiança.
        limiar_conf: confiança mínima do revisor para aceitar um canônico.
        agent_model: modelo a ser usado pelo agente (ex.: 'gpt-4o', 'gpt-4o-mini').

    Returns:
        (log_df, qual_df)
    """
    # Permite habilitar por variável de ambiente (sem alterar chamadas existentes)
    if not use_review_pipeline:
        use_review_pipeline = os.getenv("USE_REVIEW_PIPELINE", "").strip() in ("1", "true", "TRUE", "yes", "on")

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

    # Pré-construção dos mapas normalizados por ativo (evita recomputar a cada linha)
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

    # whitelist_map para o revisor (usa dicionário de equipamentos; se quiser, una com top5)
    whitelist_map: Dict[str, Set[str]] = {a: set(s) for a, s in dict_equip.items()}

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

        (top5_map, top5_list), (dict_map, dict_list) = _get_norm_structs(ativo)

        chosen = None
        fonte = None
        score = None
        status = None
        regra = None

        # 1) top5 emergenciais
        cand = _match_exact_pool(mencao_bruta, top5_map)
        if cand is None:
            cand = _match_contains(mencao_bruta, top5_list)
        if cand is not None:
            chosen = cand; fonte = "top5"; score = 100.0; status = "aceito"; regra = "top5_exato_ou_contido"
        else:
            # 2) dicionário por ativo
            cand = _match_exact_pool(mencao_bruta, dict_map)
            if cand is None:
                cand = _match_contains(mencao_bruta, dict_list)
            if cand is not None:
                chosen = cand; fonte = "dicionario"; score = 100.0; status = "aceito"; regra = "dict_exato_ou_contido"
            else:
                # 3) fuzzy dentro do pool (prioriza dicionário; se vazio, usa top5)
                pool_map = dict_map if dict_map else top5_map
                cand, sc = _match_fuzzy(mencao_bruta, pool_map)
                if cand is not None:
                    if sc >= THRESH_ACCEPT:
                        chosen = cand; fonte = "fuzzy"; score = sc; status = "aceito"; regra = f"fuzzy>={THRESH_ACCEPT}"
                    elif THRESH_AMBIG <= sc < THRESH_ACCEPT:
                        chosen = None; fonte = "fuzzy"; score = sc; status = "ambíguo"; regra = f"{THRESH_AMBIG}<=fuzzy<{THRESH_ACCEPT}"
                    else:
                        chosen = None; fonte = "fuzzy"; score = sc; status = "descartado"; regra = f"fuzzy<{THRESH_AMBIG}"
                else:
                    chosen = None; fonte = None; score = None; status = "descartado"; regra = "sem_candidato"

        # =========================================================
        # NOVO: fallback para interpretador+revisor (opção compat)
        # =========================================================
        meta_review = {}
        if chosen is None and use_review_pipeline:
            if not _REVIEW_AVAILABLE:
                # modo compat: apenas loga que não está disponível
                meta_review = {
                    "origem_classificacao": "interpretador+revisor:NOK",
                    "confianca": 0.0,
                    "motivo": "pipeline de revisão indisponível",
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
                    meta_review = {
                        "origem_classificacao": meta.get("origem_classificacao", ""),
                        "confianca": meta.get("confianca", 0.0),
                        "motivo": meta.get("motivo", ""),
                        "proposta_bruta": meta.get("proposta_bruta", ""),
                    }
                    # aceita se veio canônico (≠ NAO_CLASSIFICADO)
                    if found_list and found_list[0] and found_list[0] != NAO_CLASSIFICADO:
                        chosen = found_list[0]
                        fonte = "interpretador+revisor"
                        score = float(meta_review.get("confianca", 0.0)) * 100.0  # apenas indicativo no LOG
                        status = "aceito"
                        regra = f"revisor_conf>={limiar_conf}"
                    else:
                        # mantém sem candidato; decide incluir NAO_CLASSIFICADO abaixo
                        pass
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
        # anexa metadados do revisor quando houver
        if meta_review:
            log_rec.update({
                "origem_classificacao": meta_review.get("origem_classificacao", ""),
                "confianca": meta_review.get("confianca", np.nan),
                "motivo": meta_review.get("motivo", ""),
                "proposta_bruta": json.dumps(meta_review.get("proposta_bruta", ""), ensure_ascii=False)
                    if isinstance(meta_review.get("proposta_bruta"), (dict, list)) else meta_review.get("proposta_bruta", ""),
            })
        logs.append(log_rec)

        # Saída qualificada
        if status == "aceito" and chosen:
            linhas_ok.append({
                "id_evento": idx,
                "ativo": ativo,
                "data_evento": row.get("data_evento", pd.NaT),
                "equipamento": chosen,
                "periodo_h": row.get("periodo_h", np.nan),
                "bbl": row.get("bbl", np.nan),
                "justificativa": just,
                # se veio do revisor, preserva metadados úteis
                **({k: log_rec.get(k) for k in ("origem_classificacao","confianca","motivo","proposta_bruta")} if meta_review else {})
            })
        else:
            # incluir NAO_CLASSIFICADO se solicitado
            if use_review_pipeline and keep_unclassified and (chosen is None):
                linhas_ok.append({
                    "id_evento": idx,
                    "ativo": ativo,
                    "data_evento": row.get("data_evento", pd.NaT),
                    "equipamento": NAO_CLASSIFICADO,
                    "periodo_h": row.get("periodo_h", np.nan),
                    "bbl": row.get("bbl", np.nan),
                    "justificativa": just,
                    **({k: log_rec.get(k) for k in ("origem_classificacao","confianca","motivo","proposta_bruta")} if meta_review else {})
                })

    log_df = pd.DataFrame(logs)
    qual_df = pd.DataFrame(linhas_ok)

    # Tipagem/numéricos
    if not qual_df.empty:
        if pd.api.types.is_timedelta64_dtype(qual_df.get("periodo_h", pd.Series([], dtype="float64"))):
            qual_df["periodo_h"] = qual_df["periodo_h"].dt.total_seconds() / 3600.0
        else:
            qual_df["periodo_h"] = pd.to_numeric(qual_df["periodo_h"], errors="coerce")
        # bbl inteiro (nullable)
        if "bbl" in qual_df.columns:
            qual_df["bbl"] = pd.to_numeric(qual_df["bbl"], errors="coerce").round().astype("Int64")

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

    print(f"Matching concluído. {len(qual_df)} eventos qualificados; LOG em outputs/. "
          f"{'Revisor ON' if use_review_pipeline else 'Revisor OFF'}")
    return log_df, qual_df


if __name__ == "__main__":
    # Você pode ligar pelo env sem mudar código:
    #   USE_REVIEW_PIPELINE=1 python -m src.matching
    aplicar_matching()
