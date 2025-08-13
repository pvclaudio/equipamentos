# app/components/review_panel.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

# caminhos
ROOT    = Path(__file__).resolve().parents[1]
DATADIR = ROOT.parent / "data"
OUTDIR  = ROOT.parent / "outputs"

OVERRIDES_CSV  = DATADIR / "whitelist_overrides.csv"
SUG_CSV        = OUTDIR / "novos_equip_sugeridos.csv"   # gerado pelo agente (classe aberta)
EVT_PARQUET    = OUTDIR / "eventos_qualificados.parquet"
CORRECOES_CSV  = DATADIR / "correcoes_manuais.csv"

# utilidades de IO
def _read_csv_safe(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()

def _read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def _save_overrides(rows: list[tuple[str, str]]):
    """Acrescenta (ativo,equipamento) ao overrides CSV (dedup)."""
    DATADIR.mkdir(parents=True, exist_ok=True)
    base = _read_csv_safe(OVERRIDES_CSV, dtype=str).fillna("")
    if base.empty:
        base = pd.DataFrame(columns=["ativo","equipamento"])
    add = pd.DataFrame(rows, columns=["ativo","equipamento"])
    add["ativo"] = add["ativo"].astype(str).str.strip()
    add["equipamento"] = add["equipamento"].astype(str).str.strip()
    base = pd.concat([base, add], ignore_index=True)
    base = base[(base["ativo"]!="") & (base["equipamento"]!="")]
    base = base.drop_duplicates(subset=["ativo","equipamento"])
    base.to_csv(OVERRIDES_CSV, index=False, encoding="utf-8-sig")

def _append_correcoes(rows: list[dict]):
    """Acrescenta correções manuais em trilha data/correcoes_manuais.csv"""
    DATADIR.mkdir(parents=True, exist_ok=True)
    base = _read_csv_safe(CORRECOES_CSV)
    if base.empty:
        base = pd.DataFrame(columns=[
            "ts","id_evento","ativo","equipamento_antigo","equipamento_novo","justificativa","autor"
        ])
    add = pd.DataFrame(rows)
    base = pd.concat([base, add], ignore_index=True)
    base.to_csv(CORRECOES_CSV, index=False, encoding="utf-8-sig")

def _apply_event_corrections(changes: list[tuple[int, str]]):
    """
    Aplica correções diretamente no outputs/eventos_qualificados.parquet
    changes = [(id_evento, equipamento_novo), ...]
    """
    if not changes:
        return 0
    df = _read_parquet_safe(EVT_PARQUET)
    if df.empty or "id_evento" not in df.columns:
        return 0
    df["id_evento"] = pd.to_numeric(df["id_evento"], errors="coerce").astype("Int64")
    map_new = {int(i): str(e) for i, e in changes}
    mask = df["id_evento"].isin(map_new.keys())
    if not mask.any():
        return 0
    # aplicar
    df.loc[mask, "equipamento"] = df.loc[mask, "id_evento"].map(map_new)
    # persistir
    df.to_parquet(EVT_PARQUET, index=False)
    df.to_csv(EVT_PARQUET.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    return int(mask.sum())

# ---------------------------
# Seção A — Sugestões do agente
# ---------------------------
def section_suggestions():
    st.subheader("Promover sugestões do agente → whitelist")
    df = _read_csv_safe(SUG_CSV, dtype=str)
    if df.empty:
        st.info("Não há `outputs/novos_equip_sugeridos.csv` para revisar.")
        return

    # normalização leve
    for c in ("ativo","sugestao","confianca","justificativa"):
        if c not in df.columns:
            df[c] = ""
    df["confianca"] = pd.to_numeric(df["confianca"], errors="coerce").fillna(0.0)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        ativos = ["(todos)"] + sorted([a for a in df["ativo"].dropna().unique() if str(a).strip()])
        ativo_sel = st.selectbox("Ativo", ativos, index=0)
    with col2:
        min_conf = st.slider("Confiança mínima", 0.0, 1.0, 0.60, 0.01)
    with col3:
        filtro_txt = st.text_input("Filtro de texto (contém)", "")

    view = df.copy()
    if ativo_sel != "(todos)":
        view = view[view["ativo"] == ativo_sel]
    view = view[view["confianca"] >= float(min_conf)]
    if filtro_txt.strip():
        t = filtro_txt.strip().lower()
        view = view[
            view.astype(str).apply(lambda r: t in " ".join(r.values).lower(), axis=1)
        ]
    if view.empty:
        st.warning("Nenhuma sugestão atende aos filtros.")
        return

    # linha de edição em massa
    st.caption("Edite a sugestão se quiser normalizar o nome antes de promover.")
    view = view[["ativo","sugestao","confianca","justificativa"]].reset_index(drop=True)
    edited = st.data_editor(
        view,
        key="edit_sug",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "ativo": {"editable": False},
            "confianca": {"editable": False},
            "justificativa": {"editable": False},
            "sugestao": {"label": "equipamento (editar se necessário)"}
        }
    )

    to_add = []
    for _, r in edited.iterrows():
        a = str(r["ativo"]).strip()
        e = str(r["sugestao"]).strip()
        if a and e:
            to_add.append((a, e))

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("➕ Promover selecionadas para whitelist_overrides.csv", type="primary"):
            _save_overrides(to_add)
            st.success(f"{len(to_add)} entradas promovidas. Reabra/atualize o app para refletir o novo whitelist.")

    with c2:
        st.download_button(
            "Baixar CSV das sugestões filtradas",
            data=edited.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="sugestoes_filtradas.csv",
            mime="text/csv"
        )

# ---------------------------
# Seção B — Correção manual de eventos
# ---------------------------
def section_manual_fix(usuario: str = "usuario"):
    st.subheader("Correção manual de eventos NAO_CLASSIFICADO")
    df = _read_parquet_safe(EVT_PARQUET)
    if df.empty:
        st.info("Arquivo outputs/eventos_qualificados.parquet não encontrado ou vazio.")
        return
    # garantir tipos
    df["id_evento"] = pd.to_numeric(df.get("id_evento"), errors="coerce").astype("Int64")
    df["equipamento"] = df["equipamento"].astype(str)
    df["ativo"] = df["ativo"].astype(str)
    nao = df[df["equipamento"].str.upper().eq("NAO_CLASSIFICADO")].copy()

    if nao.empty:
        st.success("Não há eventos NAO_CLASSIFICADO no momento. ✅")
        return

    col1, col2 = st.columns([1,1])
    with col1:
        ativos = ["(todos)"] + sorted([a for a in nao["ativo"].dropna().unique() if str(a).strip()])
        ativo_sel = st.selectbox("Ativo", ativos, index=0, key="fix_ativo")
    with col2:
        filtro_txt = st.text_input("Filtro de justificativa (contém)", "", key="fix_filtro")

    view = nao
    if ativo_sel != "(todos)":
        view = view[view["ativo"] == ativo_sel]
    if filtro_txt.strip():
        t = filtro_txt.strip().lower()
        view = view[view["justificativa"].astype(str).str.lower().str.contains(t, na=False)]

    if view.empty:
        st.warning("Nenhum NAO_CLASSIFICADO com os filtros atuais.")
        return

    st.caption("Selecione as linhas e informe o equipamento canônico desejado.")
    # selecionar por id_evento
    ids = view["id_evento"].dropna().astype(int).tolist()
    sel = st.multiselect("IDs para corrigir", ids, max_selections=min(300, len(ids)))
    equip_novo = st.text_input("Equipamento novo (aplicado para todos os IDs selecionados)")

    colx, coly, colz = st.columns([1,1,1])
    with colx:
        add_to_whitelist = st.checkbox("Também promover ao whitelist_overrides.csv", value=True)
    with coly:
        autor = st.text_input("Autor (trilha)", value=usuario)
    with colz:
        aplicar = st.button("✅ Aplicar correções", type="primary", use_container_width=True)

    st.dataframe(
        view[["id_evento","ativo","data_evento","equipamento","periodo_h","bbl","justificativa"]].reset_index(drop=True),
        use_container_width=True, height=300
    )

    if aplicar:
        equip_novo = equip_novo.strip()
        if not sel or not equip_novo:
            st.error("Selecione pelo menos um ID e informe o equipamento novo.")
            return

        # trilha
        rows_trilha = []
        sub = df[df["id_evento"].isin(sel)]
        now = datetime.utcnow().isoformat()
        for _, r in sub.iterrows():
            rows_trilha.append({
                "ts": now,
                "id_evento": int(r["id_evento"]),
                "ativo": str(r["ativo"]),
                "equipamento_antigo": str(r["equipamento"]),
                "equipamento_novo": equip_novo,
                "justificativa": str(r.get("justificativa","")),
                "autor": autor
            })
        _append_correcoes(rows_trilha)

        # aplicar na base
        n_aplicados = _apply_event_corrections([(int(i), equip_novo) for i in sel])

        # opcional: promover ao overrides
        if add_to_whitelist:
            pares = [(str(a), equip_novo) for a in sub["ativo"].astype(str).unique()]
            _save_overrides(pares)

        st.success(f"Correções aplicadas em {n_aplicados} evento(s). "
                   f"{'Promovido ao whitelist.' if add_to_whitelist else ''} "
                   "Atualize os rankings para refletir.")


