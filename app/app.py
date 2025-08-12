# app/app.py
import os
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import urllib3

# =============================
# Configurações gerais
# =============================
st.set_page_config(page_title="Criticidade de Equipamentos PRIO", layout="wide")

ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Integração com o pipeline
# =============================
from src.ingestao import run_ingestao
from src.matching import aplicar_matching
from src.whitelist import build_whitelist_map

# Tenta importar o executor do agente
try:
    from src.agent_utils import explode_with_agent, salvar_monitoramento_csv_factory
    _AGENT_AVAILABLE = True
except Exception:
    explode_with_agent = None  # type: ignore
    salvar_monitoramento_csv_factory = None  # type: ignore
    _AGENT_AVAILABLE = False

# =============================
# Garantir artefatos do pipeline
# =============================
def ensure_pipeline() -> bool:
    """Gera outputs/eventos_qualificados.parquet e log_matching.parquet se ainda não existirem."""
    parquet_evt = OUT_DIR / "eventos_qualificados.parquet"
    parquet_log = OUT_DIR / "log_matching.parquet"
    if parquet_evt.exists() and parquet_log.exists():
        return True
    with st.spinner("Rodando ingestão e matching para preparar os dados..."):
        run_ingestao()       # (a) ingestão/limpeza
        aplicar_matching()   # (b) matching + logs
    ok = parquet_evt.exists() and parquet_log.exists()
    if not ok:
        st.error("Não consegui gerar os artefatos do pipeline. Verifique os arquivos em data/ e tente novamente.")
    return ok

# =============================
# Utilitários
# =============================
def _bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def _bytes_excel(sheets: dict[str, pd.DataFrame], file_name: str = "export.xlsx") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            sheet_name = str(name)[:31] if name else "Sheet1"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    bio.seek(0)
    return bio.read()

def _to_float_ptbr(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s, errors="coerce")

# =============================
# Garantir dados
# =============================
ensure_pipeline()

# =============================
# Sidebar (filtros e parâmetros)
# =============================
st.title("Equipamentos — Eventos Qualificados")

@st.cache_data(show_spinner=False)
def load_eventos_qualificados() -> pd.DataFrame:
    fp = OUT_DIR / "eventos_qualificados.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=["ativo","data_evento","equipamento","periodo_h","bbl","justificativa"])
    df = pd.read_parquet(fp)
    if "data_evento" in df.columns:
        df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    for col in ("periodo_h","bbl"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_log_matching() -> pd.DataFrame:
    fp = OUT_DIR / "log_matching.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    if "data_evento" in df.columns:
        df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")
    return df

evt = load_eventos_qualificados()
log = load_log_matching()

st.sidebar.header("Parâmetros")
# Reprocessamento manual
if st.sidebar.button("🔄 Reprocessar (ingestão + matching)"):
    if ensure_pipeline():
        load_eventos_qualificados.clear()
        load_log_matching.clear()
        st.rerun()

# Entrada de Brent
brent_value = st.sidebar.number_input(
    "Brent (USD/bbl)", min_value=0.0, max_value=300.0, value=80.0, step=1.0,
    help="Valor aplicado globalmente quando não houver série mensal para o mês do evento."
)

# Série mensal opcional (CSV: yyyymm, brent_usd)
brent_series = None
with st.sidebar.expander("Série mensal (opcional)"):
    csv = st.file_uploader("CSV com colunas: yyyymm, brent_usd", type=["csv"], key="brent_csv")
    if csv is not None:
        try:
            tmp = pd.read_csv(csv)
            col_ym = [c for c in tmp.columns if c.strip().lower() in ("yyyymm","mes")]
            col_px = [c for c in tmp.columns if c.strip().lower() in ("brent_usd","preco","preco_usd")]
            if col_ym and col_px:
                tmp["yyyymm"] = tmp[col_ym[0]].astype(str).str.strip()
                tmp["brent_usd"] = pd.to_numeric(tmp[col_px[0]], errors="coerce")
                brent_series = {k: v for k, v in tmp.dropna(subset=["yyyymm","brent_usd"]).values}
                st.success(f"Série mensal carregada: {len(brent_series)} meses")
            else:
                st.error("CSV inválido. Esperado: colunas 'yyyymm' e 'brent_usd'.")
        except Exception as e:
            st.error(f"Falha ao ler série mensal: {e}")

# =============================
# Seção IA (opcional)
# =============================
st.header("Identificação avançada por IA (opcional)")
use_agent = st.checkbox("Habilitar agente para ler justificativas e identificar equipamentos (whitelist)", value=False)
agent_model = st.selectbox("Modelo", options=["gpt-4o", "gpt-4o-mini"], index=0)
use_lexical_fallback = st.checkbox("Usar fallback léxico/fuzzy quando IA não retornar nada", value=True)

# NOVO: parâmetros do pipeline interpretador + revisor (opção 1: compatível)
with st.expander("Opções avançadas do agente"):
    use_review_pipeline = st.checkbox(
        "Ativar interpretador + revisor quando a whitelist não encontrar",
        value=False,
        help="Compatível: quando desligado, o comportamento é idêntico ao atual."
    )
    keep_unclassified = st.checkbox(
        "Manter NAO_CLASSIFICADO em vez de descartar",
        value=True,
        help="Quando ligado, eventos sem match entram como NAO_CLASSIFICADO (não são descartados)."
    )
    limiar_conf = st.slider(
        "Limiar de confiança do revisor",
        min_value=0.0, max_value=1.0, value=0.60, step=0.05,
        help="Se a confiança do revisor ficar abaixo do limiar, classifica como NAO_CLASSIFICADO."
    )
    salvar_monitoramento = st.checkbox(
        "Salvar cada linha na base de monitoramento (CSV em outputs/base_monitoramento.csv)",
        value=True
    )

if use_agent:
    # Desarma proxies/CA do ambiente (evita herdar proxy corporativo) + suprime warning
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","no_proxy","NO_PROXY"]:
        os.environ.pop(k, None)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    st.info("⚠️ Modo rede ‘sem proxy’ + SSL relaxado ativo para o agente. Se ainda falhar com CERTIFICATE_VERIFY_FAILED, é bloqueio de saída na rede.")

    if not _AGENT_AVAILABLE:
        st.warning("Agente indisponível nesta build (explode_with_agent não encontrado). Dashboard segue normal.")
    else:
        @st.cache_data(show_spinner=False)
        def _load_eventos_base():
            fp = OUT_DIR / "eventos_base.parquet"
            if not fp.exists():
                run_ingestao()
            dfb = pd.read_parquet(fp)

            # id_evento seguro (preenche com índice quando faltante)
            if "id_evento" in dfb.columns:
                tmp = pd.to_numeric(dfb["id_evento"], errors="coerce")
            else:
                tmp = pd.Series(index=dfb.index, dtype="float64")
            idx_series = pd.Series(dfb.index, index=dfb.index, dtype="int64")
            dfb["id_evento"] = tmp.fillna(idx_series).astype(int)

            # data_evento e numéricos
            dfb["data_evento"] = pd.to_datetime(dfb.get("data_evento"), errors="coerce")
            for c in ["periodo_h", "bbl", "perda_financeira_usd"]:
                if c in dfb.columns:
                    dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
            # justificativa sempre str
            if "justificativa" in dfb.columns:
                dfb["justificativa"] = dfb["justificativa"].astype(str).fillna("")

            return dfb

        df_base = _load_eventos_base().copy()
        st.caption(f"Base de eventos para IA: {len(df_base)} linhas")

        if st.button("🔍 Rodar agente (IA)"):
            # monta whitelist localmente
            wl_map = build_whitelist_map()

            prog = st.progress(0.0, text="Iniciando...")
            def _cb(done, total):
                frac = done / max(total, 1)
                prog.progress(frac, text=f"Evento {done}/{total}")

            try:
                df_scope = df_base.copy()
                df_scope["_lex_fallback"] = bool(use_lexical_fallback)

                salvar_fn = None
                if salvar_monitoramento and salvar_monitoramento_csv_factory is not None:
                    salvar_fn = salvar_monitoramento_csv_factory(str(OUT_DIR / "base_monitoramento.csv"))

                out_df = explode_with_agent(
                    df_scope,
                    wl_map,
                    progress_cb=_cb,
                    model=agent_model,
                    use_review_pipeline=use_review_pipeline,   # NOVO
                    keep_unclassified=keep_unclassified,       # NOVO
                    salvar_fn=salvar_fn,                       # NOVO
                    limiar_conf=limiar_conf                    # NOVO
                )
                if out_df is None or out_df.empty:
                    st.warning("Agente não retornou linhas qualificadas. Mantendo dados originais do matching.")
                else:
                    # mantém colunas esperadas e metadados se existirem
                    base_keep = ["ativo","data_evento","equipamento","periodo_h","bbl","justificativa"]
                    meta_cols = ["perda_financeira_usd","origem_classificacao","confianca","motivo","proposta_bruta"]
                    for k in base_keep + meta_cols:
                        if k not in out_df.columns:
                            out_df[k] = np.nan if k not in ("justificativa","origem_classificacao","motivo","proposta_bruta") else ""
                    # salva amostra p/ auditoria
                    try:
                        out_df.head(200).to_csv(OUT_DIR / "audit_agente_sample.csv", index=False, encoding="utf-8-sig")
                    except Exception:
                        pass

                    st.success(f"Agente concluiu: {len(out_df)} linhas qualificadas (explodidas por multi‑equipamento).")
                    st.session_state["evt_from_agent"] = out_df[base_keep + [c for c in meta_cols if c in out_df.columns]]
            except Exception as e:
                st.error(f"Falha geral do agente: {e}")

# Se o agente gerou um novo evt, use-o
if "evt_from_agent" in st.session_state:
    evt = st.session_state["evt_from_agent"]

# =============================
# Brent - aplicar
# =============================
def apply_brent(df: pd.DataFrame, brent_value: float | None, brent_series: dict[str, float] | None):
    df = df.copy()
    # garante colunas
    for c in ["bbl", "periodo_h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "data_evento" in df.columns:
        df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")

    if df.empty:
        df["brent_usd"] = np.nan
        df["perda_financeira_usd"] = 0.0
        return df

    if brent_series:
        # CORREÇÃO: yyyymm -> usar %Y%m para mapear corretamente
        yyyymm = df["data_evento"].dt.strftime("%Y%m")
        df["brent_usd"] = yyyymm.map(brent_series).astype(float)
        if brent_value is not None:
            df.loc[df["brent_usd"].isna(), "brent_usd"] = float(brent_value)
    else:
        df["brent_usd"] = float(brent_value if brent_value is not None else 80.0)
    df["perda_financeira_usd"] = df["bbl"].fillna(0) * df["brent_usd"].fillna(0)
    return df

# =============================
# Filtros
# =============================
if evt.empty:
    st.info("Sem dados em outputs/eventos_qualificados.parquet. Rode o pipeline.")
    st.stop()

evt = apply_brent(evt, brent_value, brent_series)

ativos = sorted(evt["ativo"].dropna().unique().tolist())
ativo_sel = st.sidebar.multiselect("Ativo", options=ativos, default=ativos)

dt_min, dt_max = evt["data_evento"].min(), evt["data_evento"].max()
periodo = st.sidebar.date_input("Período", value=(dt_min, dt_max), min_value=dt_min, max_value=dt_max)

mask = pd.Series(True, index=evt.index)
if ativo_sel:
    mask &= evt["ativo"].isin(ativo_sel)
if isinstance(periodo, (list, tuple)) and len(periodo) == 2:
    ini = pd.to_datetime(periodo[0]); fim = pd.to_datetime(periodo[1])
    mask &= (evt["data_evento"] >= ini) & (evt["data_evento"] <= fim)

evt_f = evt.loc[mask].copy()

equip_opts = sorted(evt_f["equipamento"].dropna().unique().tolist()) if not evt_f.empty else []
equip_sel = st.sidebar.multiselect("Equipamento", options=equip_opts, default=equip_opts)
if equip_sel:
    evt_f = evt_f[evt_f["equipamento"].isin(equip_sel)]

top_n = st.sidebar.slider("Top-N por métrica", min_value=5, max_value=30, value=20, step=1)

# =============================
# KPIs
# =============================
for col in ["periodo_h", "bbl", "perda_financeira_usd", "brent_usd"]:
    if col in evt_f.columns:
        if evt_f[col].dtype == "object":
            evt_f[col] = evt_f[col].apply(_to_float_ptbr)
        evt_f[col] = pd.to_numeric(evt_f[col], errors="coerce")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Eventos qualificados", f"{len(evt_f):,}".replace(",", "."))
col2.metric("Horas paradas", f"{evt_f['periodo_h'].sum():,.2f}".replace(",", "."))
col3.metric("bbl perdidos", f"{evt_f['bbl'].sum():,.2f}".replace(",", "."))
col4.metric("Perda financeira (USD)", f"{evt_f['perda_financeira_usd'].sum():,.2f}".replace(",", "."))

st.caption("*Se a série mensal não cobrir algum mês, usamos o valor único de Brent informado no sidebar. Valores resultantes nesses meses são estimativas.*")

# =============================
# Agregações e rankings
# =============================
if evt_f.empty:
    st.info("Nenhum evento após filtros.")
else:
    agg = (
        evt_f.groupby(["ativo", "equipamento"], as_index=False)
             .agg(eventos=("equipamento", "count"),
                  horas_paradas_total=("periodo_h", "sum"),
                  bbl_perdidos_total=("bbl", "sum"),
                  perda_financeira_total_USD=("perda_financeira_usd", "sum"))
    )

    rank_horas = agg.sort_values(["ativo", "horas_paradas_total"], ascending=[True, False])
    rank_bbl   = agg.sort_values(["ativo", "bbl_perdidos_total"], ascending=[True, False])
    rank_usd   = agg.sort_values(["ativo", "perda_financeira_total_USD"], ascending=[True, False])

    def _minmax(s):
        s = s.fillna(0)
        mn, mx = float(s.min()), float(s.max())
        return (s - mn) / (mx - mn) if mx != mn else s * 0
    n_h = _minmax(agg["horas_paradas_total"])
    n_b = _minmax(agg["bbl_perdidos_total"])
    n_u = _minmax(agg["perda_financeira_total_USD"])
    agg_comp = agg.copy()
    w_h, w_b, w_u = 0.4, 0.3, 0.3
    agg_comp["score_composto"] = w_h * n_h + w_b * n_b + w_u * n_u
    rank_comp = agg_comp.sort_values(["ativo", "score_composto"], ascending=[True, False])

    # Tabelas
    st.subheader("Rankings por métrica")
    tabs = st.tabs(["Horas", "bbl", "USD", "Score Composto"])
    ranks = {"Horas": rank_horas, "bbl": rank_bbl, "USD": rank_usd, "Score Composto": rank_comp}

    for i, (name, df_) in enumerate(ranks.items()):
        with tabs[i]:
            topN = df_.groupby("ativo").head(top_n)
            st.dataframe(topN, use_container_width=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.download_button(
                    "Baixar CSV (TopN)", data=_bytes_csv(topN), file_name=f"rank_{name}_Top{top_n}.csv",
                    mime="text/csv"
                )
            with c2:
                xls_bytes = _bytes_excel({"rank": topN})
                st.download_button(
                    "Baixar Excel (TopN)", data=xls_bytes, file_name=f"rank_{name}_Top{top_n}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # Gráficos
    st.subheader("Gráficos")
    colg1, colg2 = st.columns(2)

    with colg1:
        top_plot = rank_usd.groupby("ativo").head(top_n)
        if not top_plot.empty:
            fig = px.bar(top_plot, x="equipamento", y="perda_financeira_total_USD", color="ativo",
                         title=f"Top-{top_n} por Perda Financeira (USD) — por Ativo")
            fig.update_layout(xaxis_title="Equipamento", yaxis_title="Perda (USD)", legend_title="Ativo")
            st.plotly_chart(fig, use_container_width=True)

    with colg2:
        if not evt_f.empty:
            serie = evt_f.groupby([pd.Grouper(key="data_evento", freq="D"), "ativo"], as_index=False)[
                ["periodo_h", "bbl", "perda_financeira_usd"]
            ].sum()
            fig2 = px.line(serie, x="data_evento", y="perda_financeira_usd", color="ativo",
                           title="Série temporal — Perda Financeira (USD) por Ativo")
            fig2.update_layout(xaxis_title="Data", yaxis_title="Perda (USD)", legend_title="Ativo")
            st.plotly_chart(fig2, use_container_width=True)

    # Eventos & LOG
    st.subheader("Eventos qualificados (após matching/IA)")
    st.dataframe(evt_f.sort_values(["ativo", "data_evento"]).reset_index(drop=True), use_container_width=True)
    st.download_button(
        "Baixar eventos qualificados (CSV)", data=_bytes_csv(evt_f), file_name="eventos_qualificados_filtrados.csv",
        mime="text/csv"
    )

    st.subheader("Trilha de auditoria do matching (LOG)")
    if not log.empty:
        col_mencao = "menção_bruta" if "menção_bruta" in log.columns else ("mencao_bruta" if "mencao_bruta" in log.columns else None)
        base_cols = ["id_evento","ativo","data_evento",col_mencao,"equipamento_candidato",
                     "equipamento_canonizado","fonte","score","status","regra_aplicada"]
        cols = [c for c in base_cols if c and c in log.columns]
        log_view = log[cols].copy() if cols else log.copy()
        if ativo_sel:
            log_view = log_view[log_view["ativo"].isin(ativo_sel)]
        if isinstance(periodo, (list, tuple)) and len(periodo) == 2 and "data_evento" in log_view.columns:
            ini = pd.to_datetime(periodo[0]); fim = pd.to_datetime(periodo[1])
            log_view = log_view[(pd.to_datetime(log_view["data_evento"], errors="coerce") >= ini) &
                                (pd.to_datetime(log_view["data_evento"], errors="coerce") <= fim)]
        st.dataframe(log_view, use_container_width=True)
        st.download_button(
            "Baixar LOG (CSV)", data=_bytes_csv(log_view), file_name="log_matching_filtrado.csv", mime="text/csv"
        )
    else:
        st.info("LOG não encontrado (outputs/log_matching.parquet). Rode o matching.")

# Rodapé
st.caption("© PRIO — Dashboard de Criticidade de Equipamentos. IA opcional com whitelist e revisão.")
