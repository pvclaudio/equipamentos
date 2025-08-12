# src/agent_utils.py
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Set
import json
import os
import re

import numpy as np
import pandas as pd
import requests
import urllib3

# suprime avisos de SSL quando verify=False (ambiente corporativo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# Helpers de robustez
# ============================================================
def _is_na(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return x is None

def _safe_int(x, default: int = 0) -> int:
    """
    Converte para int tratando pd.NA/NaN/None/strings.
    Nunca levanta exceção – retorna default no erro.
    """
    if _is_na(x):
        return default
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return default if np.isnan(x) else int(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "nat", "<na>"):
            return default
        return int(float(s))
    except Exception:
        return default

def _safe_float(x, default: float = np.nan) -> float:
    """
    Converte para float tratando NA/strings pt‑BR/en‑US.
    Nunca levanta exceção – retorna default no erro.
    """
    if _is_na(x):
        return default
    try:
        if isinstance(x, (int, np.integer, float, np.floating)):
            return float(x)
        s = str(x).strip().replace("\u00A0", "").replace("\u202F", "")
        if s == "" or s.lower() in ("nan", "none", "nat", "<na>"):
            return default
        # pt‑BR -> en‑US
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

def _norm(s: str) -> str:
    """minúsculas, sem acento, espaço único (não depende de src.utils)."""
    import unicodedata
    if s is None:
        s = ""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s))
                if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ============================================================
# Prompt / parsing
# ============================================================
def _build_prompt(justificativa: str, ativo: str, whitelist: List[str]) -> str:
    exemplos = "\n".join(f"- {w}" for w in whitelist[:120])
    return f"""
Você é um especialista em manutenção/oper operação de FPSO.
Leia a JUSTIFICATIVA do evento e retorne APENAS um JSON válido na forma:

{{ "equipamentos": ["NOME EXATO 1", "NOME EXATO 2", ...] }}

Regras:
- Só use nomes que estejam na whitelist abaixo (mesma grafia).
- Se não houver nenhum, retorne {{ "equipamentos": [] }}.
- Não escreva comentários, explicações ou campos extras.

Whitelist (amostra — a lista completa está disponível internamente):
{exemplos}

Ativo: {ativo}
Justificativa:
\"\"\"{justificativa}\"\"\"
""".strip()

def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text

def _parse_equip_list(content: str) -> List[str]:
    """
    Tenta ler:
      { "equipamentos": ["MC A", "MC C"] }
    Se vier texto misturado, tenta extrair o primeiro objeto { ... } válido.
    """
    txt = _clean_json_text(content)
    # caminho feliz
    try:
        data = json.loads(txt)
        eqs = data.get("equipamentos", [])
        if isinstance(eqs, list):
            return [str(x).strip() for x in eqs if str(x).strip()]
    except Exception:
        pass
    # fallback: procura objetos { ... }
    try:
        for m in re.findall(r"\{.*?\}", txt, flags=re.DOTALL):
            try:
                data = json.loads(m)
                eqs = data.get("equipamentos", [])
                if isinstance(eqs, list):
                    return [str(x).strip() for x in eqs if str(x).strip()]
            except Exception:
                continue
    except Exception:
        pass
    return []

# ============================================================
# Chamada OpenAI via requests (verify=False)
# ============================================================
def _call_openai_json(prompt: str, model: str = "gpt-4o", timeout: int = 60) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada (.env).")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Você responde estritamente JSON válido."},
            {"role": "user", "content": prompt},
        ],
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
        verify=False,  # ambiente com interceptação TLS
    )
    resp.raise_for_status()
    return resp.json()

# ============================================================
# API: detectar equipamentos para uma justificativa
# ============================================================
def detect_equips_for_event(
    justificativa: str,
    ativo: str,
    whitelist_map: Dict[str, Set[str]],
    model: str = "gpt-4o",
) -> List[str]:
    """
    Lê SOMENTE a justificativa e devolve lista de equipamentos (grafia canônica).
    Usa whitelist por ativo; se vazio, retorna [].
    """
    if not isinstance(justificativa, str) or not justificativa.strip():
        return []

    wl = whitelist_map.get(ativo, set())
    if not wl:
        return []

    prompt = _build_prompt(justificativa.strip(), ativo, sorted(list(wl))[:120])
    data = _call_openai_json(prompt, model=model)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    equips = _parse_equip_list(content)

    # normaliza pela whitelist (case/acento)
    wl_norm = {_norm(x): x for x in wl}
    out: List[str] = []
    seen = set()
    for e in equips:
        k = _norm(e)
        if k in wl_norm and wl_norm[k] not in seen:
            out.append(wl_norm[k])
            seen.add(wl_norm[k])

    # Fallback simples: interseção textual
    if not out:
        jn = _norm(justificativa)
        for k, canon in wl_norm.items():
            if k and k in jn and canon not in out:
                out.append(canon)

    return out

# ============================================================
# Executor: explode e divide valores
# ============================================================
def explode_with_agent(
    df_base: pd.DataFrame,
    whitelist_map: Dict[str, Set[str]],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    model: str = "gpt-4o",
) -> pd.DataFrame:
    """
    Para cada linha:
      - detecta 0..N equipamentos
      - N==0: descarta a linha (não entra no qualificado)
      - N>=1: cria N linhas e divide igualmente periodo_h e bbl
    Retorna colunas:
      ["id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"]
    """
    if df_base is None or df_base.empty:
        return pd.DataFrame(columns=[
            "id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"
        ])

    # cópia tipada e tolerante
    df = df_base.copy()

    has_id = "id_evento" in df.columns

    # datas tolerantes
    if "data_evento" in df.columns:
        df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    else:
        df["data_evento"] = pd.NaT

    # numéricos tolerantes
    df["periodo_h"] = pd.to_numeric(df.get("periodo_h", np.nan), errors="coerce")
    df["bbl"]       = pd.to_numeric(df.get("bbl",       np.nan), errors="coerce")

    total = len(df)
    out_rows: List[dict] = []

    for pos, (idx, row) in enumerate(df.iterrows(), start=1):
        if progress_cb:
            try:
                progress_cb(pos, total)
            except Exception:
                pass

        ativo = str(row.get("ativo", "") or "").strip()
        just  = str(row.get("justificativa", "") or "").strip()
        if not ativo or not just:
            continue

        event_id = _safe_int(row["id_evento"], default=int(idx)) if has_id else int(idx)
        base_h   = _safe_float(row.get("periodo_h", np.nan), default=np.nan)
        base_bbl = _safe_float(row.get("bbl",       np.nan), default=np.nan)

        # chama agente
        try:
            found = detect_equips_for_event(just, ativo, whitelist_map, model=model)
        except Exception:
            # se falhar a chamada, apenas pula este evento
            continue

        if not found:
            continue

        n = max(1, len(found))
        share_h   = (base_h   / n) if not _is_na(base_h)   else np.nan
        share_bbl = (base_bbl / n) if not _is_na(base_bbl) else np.nan

        for eq in found:
            out_rows.append({
                "id_evento": event_id,
                "ativo": ativo,
                "data_evento": row.get("data_evento", pd.NaT),
                "equipamento": eq,
                "periodo_h": share_h,
                "bbl": share_bbl,
                "justificativa": just,
            })

    out = pd.DataFrame(out_rows, columns=[
        "id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"
    ])

    if not out.empty:
        out["data_evento"] = pd.to_datetime(out["data_evento"], errors="coerce")
        out["periodo_h"]   = pd.to_numeric(out["periodo_h"], errors="coerce")
        out["bbl"]         = pd.to_numeric(out["bbl"], errors="coerce")

    return out
