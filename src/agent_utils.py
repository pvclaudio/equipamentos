# src/agent_utils.py
from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3

# Ambiente corporativo com inspeção TLS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# Normalização e utilitários
# =========================
_WS = re.compile(r"\s+", re.UNICODE)

def _norm(s: str) -> str:
    """remove acento, baixa, tira pontuação leve e colapsa espaços (para matching)."""
    import unicodedata
    if s is None:
        return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _WS.sub(" ", s).strip()

def _is_na(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def _safe_float(x, default: float = np.nan) -> float:
    if _is_na(x):
        return default
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "nat", "<na>"):
            return default
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        return default

def _safe_int(x, default: int = 0) -> int:
    if _is_na(x):
        return default
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return default
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "nat", "<na>"):
            return default
        return int(float(s))
    except Exception:
        return default

# ================  Prompt & parsing  ================
_SCHEMA_HINT = (
    'Responda APENAS JSON válido. Formato aceito: '
    '{"equipamentos": ["MC A","BOMBA DE INJEÇÃO C"]} '
    'ou simplesmente ["MC A","BOMBA DE INJEÇÃO C"].'
)

def _build_prompt(just: str, ativo: str, whitelist: Sequence[str]) -> str:
    examples = "\n".join(f"- {w}" for w in whitelist[:80])
    return f"""
Você é um especialista em manutenção/ops de FPSO. Leia a justificativa e retorne TODOS os equipamentos citados,
EXCLUSIVAMENTE entre os itens da whitelist do ativo indicado. { _SCHEMA_HINT }

Regras:
- Use exatamente a grafia da whitelist.
- Nada de comentários, títulos ou texto fora do JSON.
- Se não houver equipamento, retorne {{"equipamentos":[]}}.

Ativo: {ativo}

Whitelist (amostra):
{examples}

Justificativa:
\"\"\"{just.strip()}\"\"\"""".strip()

def _clean_json_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    return t

def _parse_llm_output(content: str) -> List[str]:
    if not content:
        return []
    txt = _clean_json_text(content)
    m_list = re.search(r"\[[\s\S]*\]", txt)
    m_obj  = re.search(r"\{[\s\S]*\}", txt)
    cand = m_obj.group(0) if m_obj else (m_list.group(0) if m_list else txt)
    try:
        data = json.loads(cand)
        eqs = data.get("equipamentos", []) if isinstance(data, dict) else data
        if not isinstance(eqs, list):
            return []
        out = [str(x).strip() for x in eqs if str(x).strip()]
        seen, uniq = set(), []
        for e in out:
            if e not in seen:
                uniq.append(e); seen.add(e)
        return uniq
    except Exception:
        return []

# ===================================================
# Utilitário: extrair primeiro JSON-objeto
# ===================================================
def _extract_first_json_obj(text: str) -> dict:
    if not text:
        return {}
    t = _clean_json_text(text)
    m_obj = re.search(r"\{[\s\S]*\}", t)
    if not m_obj:
        return {}
    try:
        return json.loads(m_obj.group(0))
    except Exception:
        return {}

# ===========================  OpenAI  ===========================
def _post_openai(payload: dict, api_key: str, timeout: int = 60) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/chat/completions"
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}

def _call_openai_json(prompt: str, model: str = "gpt-4o", api_key: Optional[str] = None) -> List[str]:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        return []
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Você responde exclusivamente em JSON válido."},
            {"role": "user", "content": prompt}
        ],
    }
    data = _post_openai(payload, key)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _parse_llm_output(content)

def _call_openai_json_obj(prompt: str, model: str = "gpt-4o", api_key: Optional[str] = None) -> dict:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        return {}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Você responde exclusivamente em JSON válido."},
            {"role": "user", "content": prompt}
        ],
    }
    data = _post_openai(payload, key)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _extract_first_json_obj(content)

# ===========================  Fallback léxico  ===========================
def _to_regex_from_name(name: str) -> re.Pattern:
    base = _norm(name)
    esc = r"\s+".join(map(re.escape, base.split()))
    return re.compile(rf"\b{esc}\b", re.IGNORECASE)

def _lexical_candidates(just_text: str, wl: Set[str]) -> List[str]:
    if not wl:
        return []
    txt = _norm(just_text)
    if not txt:
        return []
    specials: List[Tuple[re.Pattern, str]] = []
    specials += [(re.compile(r"\bmc\s*a\b"), "MC A"),
                 (re.compile(r"\bmc\s*b\b"), "MC B"),
                 (re.compile(r"\bmc\s*c\b"), "MC C")]
    specials += [(re.compile(r"\bbomba\s+de\s+inj[eé]cao\s*a\b"), "BOMBA DE INJEÇÃO A"),
                 (re.compile(r"\bbomba\s+de\s+inj[eé]cao\s*b\b"), "BOMBA DE INJEÇÃO B"),
                 (re.compile(r"\bbomba\s+de\s+inj[eé]cao\s*c\b"), "BOMBA DE INJEÇÃO C"),
                 (re.compile(r"\bbomba\s+de\s+inj[eé]cao\s*d\b"), "BOMBA DE INJEÇÃO D")]
    patterns = [( _to_regex_from_name(w), w) for w in wl]

    found: List[str] = []
    seen: Set[str] = set()
    for rx, label in specials:
        if rx.search(txt) and label in wl and label not in seen:
            found.append(label); seen.add(label)
    for rx, label in patterns:
        if label in seen:
            continue
        if rx.search(txt):
            found.append(label); seen.add(label)
    return found

# ===========================  API pública  ===========================
def detect_equips_for_event(
    justificativa: str,
    ativo: str | None,
    whitelist_map: Dict[str, Set[str]],
    model: str = "gpt-4o",
    use_lex_fallback: bool = True,
) -> List[str]:
    if not isinstance(justificativa, str) or not justificativa.strip():
        return []
    wl_for_asset = whitelist_map.get(str(ativo), set())
    if not wl_for_asset:
        all_union: Set[str] = set()
        for s in whitelist_map.values():
            all_union.update(s)
        wl_for_asset = all_union
    if not wl_for_asset:
        return []
    prompt = _build_prompt(justificativa, str(ativo), sorted(list(wl_for_asset))[:160])
    llm_out = _call_openai_json(prompt, model=model)

    wl_norm = {_norm(x): x for x in wl_for_asset}
    picked: List[str] = []
    seen = set()
    for e in llm_out:
        k = _norm(e)
        if k in wl_norm and wl_norm[k] not in seen:
            picked.append(wl_norm[k]); seen.add(wl_norm[k])

    if not picked and use_lex_fallback:
        picked = _lexical_candidates(justificativa, wl_for_asset)
    return picked

# =====================================================
# Interpretador livre + revisor HÍBRIDO (classe aberta)
# =====================================================
NAO_CLASSIFICADO = "NAO_CLASSIFICADO"

_PROMPT_INTERPRETADOR_LIVRE = """
Você é um especialista em ativos industriais de O&G.
Tarefa: a partir do texto abaixo, IDENTIFIQUE qual é o equipamento citado/principal afetado.
- Se não houver menção explícita, deduza pelo contexto (processo, sintomas, verbos, variáveis).
- Responda em JSON:
{
  "equipamento_inferido": "<string>",
  "confianca": <0..1>,
  "justificativa": "<máx 200 caracteres>"
}
Texto:
\"\"\"{texto}\"\"\"""".strip()

_PROMPT_REVISOR_HIBRIDO = """
Você é um revisor técnico. Recebe:
- Uma proposta de equipamento inferido por outro agente.
- Uma whitelist (nomes canônicos) do ativo (se disponível).

Regras:
1) Se a proposta corresponder claramente a um item da whitelist (igual/sinônimo), normalize para o NOME CANÔNICO.
2) Se NÃO corresponder, mas ainda assim a proposta for um NOME DE EQUIPAMENTO plausível (classe aberta),
   aceite a proposta como "novo_equipamento" (fora do whitelist).
3) Se o texto não sustentar um equipamento, use "NAO_CLASSIFICADO".
4) Confiança (0..1) conforme clareza e aderência.

Responda em JSON:
{
  "equipamento_final": "<NOME_CANONICO|novo_equipamento|NAO_CLASSIFICADO>",
  "nome_novo": "<preencher caso seja classe aberta, senão vazio>",
  "confianca_revisao": <0..1>,
  "motivo": "<máx 200 caracteres>"
}

Whitelist:
{whitelist}

Proposta:
{proposta_json}

Texto:
\"\"\"{texto}\"\"\"""".strip()

def _free_interpret_equipment(justificativa: str, model: str = "gpt-4o") -> dict:
    prompt = _PROMPT_INTERPRETADOR_LIVRE.format(texto=justificativa.strip())
    out = _call_openai_json_obj(prompt, model=model) or {}
    e = str(out.get("equipamento_inferido", "")).strip() or ""
    c = _safe_float(out.get("confianca"), default=np.nan)
    j = str(out.get("justificativa", "")).strip()
    if math.isnan(c):
        c = 0.4
    return {"equipamento_inferido": e[:120], "confianca": float(max(0.0, min(1.0, c))), "justificativa": j[:200]}

def _review_equipment_hibrido(
    proposta: dict, justificativa: str, whitelist: Sequence[str], *, model: str = "gpt-4o"
) -> dict:
    wl_txt = "\n".join(f"- {w}" for w in whitelist[:200]) or "- (vazio)"
    prompt = _PROMPT_REVISOR_HIBRIDO.format(
        whitelist=wl_txt,
        proposta_json=json.dumps(proposta, ensure_ascii=False),
        texto=justificativa.strip()
    )
    out = _call_openai_json_obj(prompt, model=model) or {}
    eq_fin = str(out.get("equipamento_final", "")).strip() or NAO_CLASSIFICADO
    nome_novo = str(out.get("nome_novo", "")).strip()
    conf   = _safe_float(out.get("confianca_revisao"), default=0.0)
    mot    = str(out.get("motivo", "")).strip()

    wl_set = set(map(str, whitelist))
    if eq_fin not in ("NAO_CLASSIFICADO","novo_equipamento") and eq_fin not in wl_set:
        wl_norm = {_norm(x): x for x in wl_set}
        k = _norm(eq_fin)
        mapped = wl_norm.get(k)
        if mapped:
            eq_fin = mapped
            nome_novo = ""
        else:
            eq_fin = "novo_equipamento"

    return {
        "equipamento_final": eq_fin,
        "nome_novo": nome_novo if eq_fin == "novo_equipamento" else "",
        "confianca_revisao": float(max(0.0, min(1.0, conf))),
        "motivo": mot[:200]
    }

def _append_csv(path: str, row: dict, fieldnames: list[str]):
    import os, csv
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def classificar_com_revisao(
    justificativa: str,
    ativo: str | None,
    whitelist_map: Dict[str, Set[str]],
    model_interpretador: str = "gpt-4o",
    model_revisor: str = "gpt-4o",
    limiar_conf: float = 0.55
) -> Tuple[List[str], dict]:
    wl_for_asset = whitelist_map.get(str(ativo), set())

    found = detect_equips_for_event(
        justificativa, ativo, whitelist_map, model=model_interpretador, use_lex_fallback=True
    )
    if found:
        return found, {
            "origem_classificacao": "whitelist|lex",
            "confianca": 1.0,
            "motivo": "Correspondência por whitelist/lexical"
        }

    proposta = _free_interpret_equipment(justificativa, model=model_interpretador)
    revisao  = _review_equipment_hibrido(
        proposta, justificativa, sorted(list(wl_for_asset)), model=model_revisor
    )

    eq_fin = revisao.get("equipamento_final","NAO_CLASSIFICADO")
    conf   = _safe_float(revisao.get("confianca_revisao"), default=0.0)
    nome_novo = revisao.get("nome_novo","")

    if eq_fin == "novo_equipamento" and nome_novo:
        _append_csv(
            "outputs/novos_equip_sugeridos.csv",
            {
                "ativo": str(ativo or ""),
                "sugestao": nome_novo,
                "confianca": conf,
                "justificativa": justificativa,
                "proposta_bruta": json.dumps(proposta, ensure_ascii=False)
            },
            ["ativo","sugestao","confianca","justificativa","proposta_bruta"]
        )
        return [nome_novo], {
            "origem_classificacao": "interpretador+revisor(classe_aberta)",
            "confianca": float(conf),
            "motivo": revisao.get("motivo",""),
            "proposta_bruta": proposta
        }

    if eq_fin != NAO_CLASSIFICADO and conf < limiar_conf:
        eq_fin = NAO_CLASSIFICADO
        revisao["motivo"] = (revisao.get("motivo") or "") + " | abaixo do limiar"

    return [eq_fin], {
        "origem_classificacao": "interpretador+revisor",
        "confianca": float(conf),
        "motivo": revisao.get("motivo", ""),
        "proposta_bruta": proposta
    }

# ===========================  Persistência p/ monitoramento  ===========================
def salvar_monitoramento_csv_factory(caminho_csv: str):
    import os, csv
    campos = [
        "id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa",
        "perda_financeira_usd","origem_classificacao","confianca","motivo","proposta_bruta"
    ]
    def _save(row: dict):
        row = row.copy()
        if isinstance(row.get("proposta_bruta"), (dict, list)):
            row["proposta_bruta"] = json.dumps(row["proposta_bruta"], ensure_ascii=False)
        file_exists = os.path.exists(caminho_csv)
        with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            if not file_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in campos})
    return _save

# ===========================  explode_with_agent (compat)  ===========================
def explode_with_agent(
    df_in: pd.DataFrame,
    whitelist_map: Dict[str, Set[str]],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    model: str = "gpt-4o",
    *,
    use_review_pipeline: bool = True,
    keep_unclassified: bool = False,
    salvar_fn: Optional[Callable[[dict], None]] = None,
    limiar_conf: float = 0.55
) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=[
            "id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"
        ])

    df = df_in.copy()
    total = len(df)

    if "id_evento" in df.columns:
        tmp = pd.to_numeric(df["id_evento"], errors="coerce")
    else:
        tmp = pd.Series(index=df.index, dtype="float64")
    df["id_evento"] = tmp.fillna(pd.Series(df.index, index=df.index, dtype="int64")).astype(int)

    df["data_evento"] = pd.to_datetime(df.get("data_evento"), errors="coerce")
    for c in ("periodo_h", "bbl"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    use_fallback_series = df["_lex_fallback"] if "_lex_fallback" in df.columns else True

    rows: List[dict] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        if progress_cb:
            try:
                progress_cb(i, total)
            except Exception:
                pass

        ativo = getattr(row, "ativo", None)
        just  = getattr(row, "justificativa", None)
        if not isinstance(just, str) or not just.strip():
            continue

        base_h   = _safe_float(getattr(row, "periodo_h", np.nan), default=np.nan)
        base_bbl = _safe_float(getattr(row, "bbl",       np.nan), default=np.nan)
        base_usd = _safe_float(getattr(row, "perda_financeira_usd", np.nan), default=np.nan)

        use_fb = bool(use_fallback_series if isinstance(use_fallback_series, (bool, np.bool_)) else
                      getattr(row, "_lex_fallback", True))

        try:
            if use_review_pipeline:
                found_list, meta = classificar_com_revisao(
                    just, ativo, whitelist_map,
                    model_interpretador=model, model_revisor=model, limiar_conf=limiar_conf
                )
            else:
                found_list = detect_equips_for_event(
                    just, ativo, whitelist_map, model=model, use_lex_fallback=use_fb
                )
                meta = {"origem_classificacao": "whitelist|lex" if found_list else "whitelist|lex:sem_match",
                        "confianca": 1.0 if found_list else 0.0,
                        "motivo": ""}
        except Exception:
            found_list, meta = [], {"origem_classificacao":"erro","confianca":0.0,"motivo":"exceção na classificação"}

        if not found_list:
            if keep_unclassified:
                found_list = [NAO_CLASSIFICADO]
            else:
                continue

        is_unclassified = (len(found_list) == 1 and found_list[0] == NAO_CLASSIFICADO)
        n = 1 if is_unclassified else max(1, len(found_list))

        share_h   = (base_h   / n) if not math.isnan(base_h)   else np.nan
        share_bbl = (base_bbl / n) if not math.isnan(base_bbl) else np.nan
        share_usd = (base_usd / n) if not math.isnan(base_usd) else np.nan

        for eq in (found_list if not is_unclassified else [NAO_CLASSIFICADO]):
            rec = {
                "id_evento": getattr(row, "id_evento"),
                "ativo": ativo,
                "data_evento": getattr(row, "data_evento", pd.NaT),
                "equipamento": eq,
                "periodo_h": share_h,
                "bbl": share_bbl,
                "justificativa": just,
                "perda_financeira_usd": share_usd if not math.isnan(share_usd) else np.nan,
                "origem_classificacao": meta.get("origem_classificacao",""),
                "confianca": meta.get("confianca", np.nan),
                "motivo": meta.get("motivo",""),
                "proposta_bruta": meta.get("proposta_bruta",""),
            }
            rows.append(rec)
            if salvar_fn:
                try:
                    salvar_fn(rec)
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=[
            "id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"
        ])

    base_cols = ["id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa","perda_financeira_usd"]
    extra_cols = ["origem_classificacao","confianca","motivo","proposta_bruta"]
    cols = base_cols + [c for c in extra_cols if any(c in r for r in rows)]
    out = pd.DataFrame(rows, columns=cols)

    out["data_evento"] = pd.to_datetime(out.get("data_evento"), errors="coerce")
    for c in ("periodo_h","bbl","perda_financeira_usd"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    keep = ["id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"]
    if "perda_financeira_usd" in out.columns:
        keep.append("perda_financeira_usd")
    for c in ("origem_classificacao","confianca","motivo","proposta_bruta"):
        if c in out.columns:
            keep.append(c)
    return out[keep]
