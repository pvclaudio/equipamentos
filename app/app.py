# =============================
# Bootstrap de paths (IMPORTANTE)
# =============================
import os, sys
from pathlib import Path

# raiz do projeto: .../Lista de Equipamentos Críticos
ROOT = Path(__file__).resolve().parents[1]
# garante que src/ seja importável como pacote
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# garante que "data/" e "outputs/" sejam relativos à raiz
os.chdir(ROOT)

# =============================
# Imports padrão do app
# =============================
from io import BytesIO
from collections import Counter
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import urllib3

# =============================
# Configurações gerais
# =============================
st.set_page_config(page_title="Criticidade de Equipamentos PRIO", layout="wide")

OUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

NAO_CLASSIFICADO = "NAO_CLASSIFICADO"

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


def ensure_pipeline() -> bool:
    """Gera outputs/eventos_qualificados.parquet e log_matching.parquet se ainda não existirem."""
    parquet_evt = OUT_DIR / "eventos_qualificados.parquet"
    parquet_log = OUT_DIR / "log_matching.parquet"
    if parquet_evt.exists() and parquet_log.exists():
        return True
    with st.spinner("Rodando ingestão e matching para preparar os dados..."):
        run_ingestao()       # (a) ingestão/limpeza
        # Mantém NAO_CLASSIFICADO p/ garantir presença de todos os ativos
        aplicar_matching(use_review_pipeline=False, keep_unclassified=True)
    ok = parquet_evt.exists() and parquet_log.exists()
    if not ok:
        st.error("Não consegui gerar os artefatos do pipeline. Verifique os arquivos em data/ e tente novamente.")
        st.stop()  # não segue sem base
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
    for col in ("periodo_h","bbl","perda_financeira_usd","brent_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # garante string
    if "equipamento" in df.columns:
        df["equipamento"] = df["equipamento"].astype(str)
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
    with st.spinner("Reprocessando pipeline..."):
        run_ingestao()
        aplicar_matching(use_review_pipeline=False, keep_unclassified=True)
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
            col_ym = [c for c in tmp.columns if c.strip().lower() in ("yyyymm","mes","ano_mes","year_month")]
            col_px = [c for c in tmp.columns if c.strip().lower() in ("brent_usd","preco","preco_usd","price")]
            if col_ym and col_px:
                tmp["yyyymm"] = tmp[col_ym[0]].astype(str).str.strip().str.replace("/", "-")
                tmp["brent_usd"] = pd.to_numeric(tmp[col_px[0]], errors="coerce")
                brent_series = {k: v for k, v in tmp.dropna(subset=["yyyymm","brent_usd"]).values}
                st.success(f"Série mensal carregada: {len(brent_series)} meses")
            else:
                st.error("CSV inválido. Esperado: colunas 'yyyymm' e 'brent_usd'.")
        except Exception as e:
            st.error(f"Falha ao ler série mensal: {e}")

# =============================
# Seção IA (opcional) — roda SÓ nas lacunas
# =============================
st.header("Complemento por IA (atua só nas lacunas)")

use_agent = st.checkbox("Habilitar agente", value=False)
agent_model = st.selectbox("Modelo", options=["gpt-4o", "gpt-4o-mini"], index=0)
limiar_conf = st.slider("Limiar de confiança do revisor (0.0–1.0)", 0.0, 1.0, 0.60, 0.01)
use_lexical_fallback = st.checkbox("Usar fallback léxico/fuzzy quando IA não retornar nada", value=True)

# Info de rede / chave
if use_agent:
    # Desarma proxies/CA do ambiente (evita herdar proxy corporativo) + suprime warning
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","no_proxy","NO_PROXY"]:
        os.environ.pop(k, None)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    st.caption(f"OPENAI_API_KEY set? {'✅' if key_present else '❌'}")
    if not key_present:
        st.warning("Defina OPENAI_API_KEY no ambiente (.env). O agente não conseguirá classificar.")

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
            for c in ["periodo_h", "bbl"]:
                if c in dfb.columns:
                    dfb[c] = pd.to_numeric(dfb[c], errors="coerce")
            # justificativa sempre str
            if "justificativa" in dfb.columns:
                dfb["justificativa"] = dfb["justificativa"].astype(str).fillna("")

            return dfb

        df_base = _load_eventos_base().copy()
        st.caption(f"Base de eventos (total): {len(df_base)}")

        # Seleciona lacunas: (a) NAO_CLASSIFICADO já salvo; (b) ids com status ambíguo/descartado no LOG
        ids_amb_desc = set()
        if not log.empty:
            tmp = log[log["status"].isin(["ambíguo", "descartado"])]["id_evento"].dropna()
            try:
                ids_amb_desc = set(tmp.astype(int).tolist())
            except Exception:
                pass

        ids_nao_class = set()
        if not evt.empty and "equipamento" in evt.columns:
            tmp2 = evt.loc[evt["equipamento"].astype(str).str.upper().eq(NAO_CLASSIFICADO), "id_evento"].dropna()
            try:
                ids_nao_class = set(tmp2.astype(int).tolist())
            except Exception:
                pass

        target_ids = ids_amb_desc.union(ids_nao_class)
        df_scope = df_base[df_base["id_evento"].isin(target_ids)].copy()

        st.caption(f"Lacunas estimadas (ambíguo/descartado ou {NAO_CLASSIFICADO}): {len(df_scope)}")

        if st.button("🤖 Rodar agente (IA) nas lacunas"):
            if df_scope.empty:
                st.warning("Não há lacunas para processar com o agente.")
            else:
                # monta whitelist localmente
                wl_map = build_whitelist_map(
                    union_top5=True, include_overrides=True, include_static_backfill=True
                )

                prog = st.progress(0.0, text="Iniciando...")
                def _cb(done, total):
                    frac = done / max(total, 1)
                    prog.progress(frac, text=f"Evento {done}/{total}")

                # opcional: logging em base de monitoramento
                salvar_fn = None
                if salvar_monitoramento_csv_factory is not None:
                    salvar_fn = salvar_monitoramento_csv_factory(str(OUT_DIR / "base_monitoramento.csv"))

                try:
                    df_scope["_lex_fallback"] = bool(use_lexical_fallback)

                    out_df = explode_with_agent(
                        df_scope,
                        wl_map,
                        progress_cb=_cb,
                        model=agent_model,
                        use_review_pipeline=True,
                        keep_unclassified=True,     # mantém NAO_CLASSIFICADO p/ rastreabilidade
                        salvar_fn=salvar_fn,
                        limiar_conf=float(limiar_conf),
                    )
                    if out_df is None or out_df.empty:
                        st.warning("Agente não retornou novas linhas. Verifique a chave de API e o limiar de confiança.")
                    else:
                        # Merge inteligente: substitui NAO_CLASSIFICADO pela classificação do agente (se houver)
                        base_evt = load_eventos_qualificados().copy()
                        out_df_use = out_df.copy()
                        out_df_use["__src"] = "AGENTE"
                        base_evt["__src"] = "MATCHING"

                        merged = pd.concat([base_evt, out_df_use], ignore_index=True)
                        merged.sort_values(["id_evento","__src"], ascending=[True, True], inplace=True)

                        def _pick(group: pd.DataFrame) -> pd.DataFrame:
                            # 1) se houver alguma linha do agente com equipamento != NAO_CLASSIFICADO, pegue essa(s)
                            g_agent = group[group["__src"]=="AGENTE"]
                            g_agent_can = g_agent[g_agent["equipamento"].astype(str).str.upper()!=NAO_CLASSIFICADO]
                            if not g_agent_can.empty:
                                return g_agent_can
                            # 2) senão, mantenha linhas do matching
                            g_match = group[group["__src"]=="MATCHING"]
                            if not g_match.empty:
                                return g_match
                            # 3) fallback
                            return group

                        merged2 = merged.groupby("id_evento", group_keys=False).apply(_pick)
                        merged2.drop(columns=[c for c in ["__src"] if c in merged2.columns], inplace=True)

                        # salva e injeta na sessão
                        merged2.to_parquet(OUT_DIR / "eventos_qualificados.parquet", index=False)
                        merged2.to_csv(OUT_DIR / "eventos_qualificados.csv", index=False, encoding="utf-8-sig")

                        load_eventos_qualificados.clear()
                        evt_new = load_eventos_qualificados()
                        st.success(f"Agente concluiu. Base qualificada agora com {len(evt_new)} linhas.")
                        st.session_state["evt_from_agent"] = evt_new
                except Exception as e:
                    st.error(f"Falha geral do agente: {e}")

# Se o agente gerou um novo evt, use-o
if "evt_from_agent" in st.session_state:
    evt = st.session_state["evt_from_agent"]

# =============================
# Brent - aplicar (aceita YYYYMM e YYYY-MM)
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
        # normaliza dicionário para suportar 'YYYYMM' e 'YYYY-MM'
        brent_series_norm = {}
        for k, v in brent_series.items():
            ks = str(k).strip()
            if not ks:
                continue
            brent_series_norm[ks] = float(v) if v is not None and not pd.isna(v) else np.nan
            if "-" in ks:
                brent_series_norm[ks.replace("-", "")] = brent_series_norm[ks]
            else:
                if len(ks) == 6 and ks.isdigit():
                    brent_series_norm[f"{ks[:4]}-{ks[4:]}"] = brent_series_norm[ks]

        y_m = df["data_evento"].dt.strftime("%Y-%m")
        ymm = df["data_evento"].dt.strftime("%Y%m")
        br1 = y_m.map(brent_series_norm)
        br2 = ymm.map(brent_series_norm)
        df["brent_usd"] = br1.combine_first(br2).astype(float)

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

# datas válidas
valid_dates = evt["data_evento"].dropna()
if valid_dates.empty:
    # Sem datas válidas → não aplicamos filtro por data
    today = pd.Timestamp.today().normalize().date()
    dt_min = today
    dt_max = today
    st.warning("Base sem datas válidas. Exibindo tudo (filtro de datas desativado).")
    periodo = (dt_min, dt_max)
else:
    dt_min = valid_dates.min().date()
    dt_max = valid_dates.max().date()
    periodo = st.sidebar.date_input("Período", value=(dt_min, dt_max), min_value=dt_min, max_value=dt_max)

mask = pd.Series(True, index=evt.index)
if ativo_sel:
    mask &= evt["ativo"].isin(ativo_sel)
if not valid_dates.empty and isinstance(periodo, (list, tuple)) and len(periodo) == 2:
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

st.caption("*Se a série mensal não cobrir algum mês, usamos o valor único de Brent informado no sidebar.*")

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
    # export com data formatada (evita células vazias)
    tmp_export = evt_f.copy()
    if "data_evento" in tmp_export.columns and pd.api.types.is_datetime64_any_dtype(tmp_export["data_evento"]):
        tmp_export["data_evento"] = tmp_export["data_evento"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Baixar eventos qualificados (CSV)", data=_bytes_csv(tmp_export), file_name="eventos_qualificados_filtrados.csv",
        mime="text/csv"
    )

# =============================
# Trilha de auditoria (LOG) + Contagens & Termos
# =============================
st.subheader("Trilha de auditoria do matching (LOG)")
if log.empty:
    st.info("LOG não encontrado (outputs/log_matching.parquet). Rode o matching.")
else:
    # recorte por filtros globais
    log_view = log.copy()
    if ativo_sel:
        log_view = log_view[log_view["ativo"].isin(ativo_sel)]
    if not valid_dates.empty and isinstance(periodo, (list, tuple)) and len(periodo) == 2 and "data_evento" in log_view.columns:
        ini = pd.to_datetime(periodo[0]); fim = pd.to_datetime(periodo[1])
        log_view = log_view[(pd.to_datetime(log_view["data_evento"], errors="coerce") >= ini) &
                            (pd.to_datetime(log_view["data_evento"], errors="coerce") <= fim)]

    # mostra o log filtrado
    col_mencao = "menção_bruta" if "menção_bruta" in log_view.columns else ("mencao_bruta" if "mencao_bruta" in log_view.columns else None)
    base_cols = ["id_evento","ativo","data_evento",col_mencao,"equipamento_candidato",
                 "equipamento_canonizado","fonte","score","status","regra_aplicada"]
    cols = [c for c in base_cols if c and c in log_view.columns]
    log_table = log_view[cols].copy() if cols else log_view.copy()
    st.dataframe(log_table, use_container_width=True)
    st.download_button(
        "Baixar LOG (CSV)", data=_bytes_csv(log_table), file_name="log_matching_filtrado.csv", mime="text/csv"
    )

    # Contagens por status/ativo
    st.subheader("Qualidade do matching — contagens por status/ativo")
    cont = (log_view.groupby(["status", "ativo"]).size()
                    .reset_index(name="qtd")
                    .pivot(index="status", columns="ativo", values="qtd")
                    .fillna(0).astype(int))
    st.dataframe(cont, use_container_width=True)

    # Termos mais frequentes em ambíguo/descartado
    st.subheader("Termos mais frequentes em justificativas (Ambíguo/Descartado)")
    colf1, colf2, colf3 = st.columns([1, 1, 1])
    with colf1:
        status_pick = st.selectbox("Status", options=["ambíguo", "descartado"], index=0)
    with colf2:
        ativo_pick = st.selectbox("Ativo (opcional)", options=["(Todos)"] + sorted(log_view["ativo"].dropna().unique().tolist()), index=0)
    with colf3:
        top_terms_n = st.number_input("Top-N termos", min_value=10, max_value=200, value=40, step=5)

    df_terms = log_view[log_view["status"] == status_pick].copy()
    if ativo_pick != "(Todos)":
        df_terms = df_terms[df_terms["ativo"] == ativo_pick]

    # coluna de menção
    col_menc = None
    for c_try in ("menção_bruta", "mencao_bruta"):
        if c_try in df_terms.columns:
            col_menc = c_try
            break

    def _tokenize(text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        t = (text or "").lower()
        try:
            from unidecode import unidecode
            t = unidecode(t)
        except Exception:
            pass
        toks = re.findall(r"[a-z0-9][a-z0-9._\-]*", t)
        return toks

    basic_sw = {
        "a","o","os","as","de","do","da","dos","das","para","por","no","na","nos","nas","em",
        "um","uma","e","ou","com","sem","que","se","ao","aos","à","às","entre","sobre","como",
        "foi","ser","estar","estava","estao","esta","está","estão","sendo","teve","devido",
        "após","apos","antes","durante","pelo","pela","pelos","pelas",
        "-", "_", ".", ":", ";", ",", "—", "–"
    }

    if col_menc is None or df_terms.empty:
        st.info("Sem justificativas para extrair termos neste recorte.")
    else:
        tokens = []
        for s in df_terms[col_menc].dropna().astype(str):
            tokens.extend([t for t in _tokenize(s) if len(t) >= 2 and t not in basic_sw])

        freq = Counter(tokens)
        top_df = (pd.DataFrame(freq.most_common(int(top_terms_n)), columns=["termo", "frequencia"])
                    .assign(share=lambda d: d["frequencia"] / max(d["frequencia"].sum(), 1.0)))

        if top_df.empty:
            st.info("Nenhum termo encontrado após normalização.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(top_df, use_container_width=True)
                st.download_button(
                    "Baixar CSV — termos",
                    data=top_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"top_termos_{status_pick}_{ativo_pick.replace(' ','_')}.csv",
                    mime="text/csv"
                )
            with c2:
                fig_top = px.bar(top_df.head(30), x="termo", y="frequencia",
                                 title=f"Top termos — {status_pick}{'' if ativo_pick=='(Todos)' else f' · {ativo_pick}'}")
                fig_top.update_layout(xaxis_title="Termo", yaxis_title="Frequência", xaxis_tickangle=-35)
                st.plotly_chart(fig_top, use_container_width=True)

# =============================
# Correção manual & Aprendizado contínuo
# =============================
st.header("Correção manual & aprendizado contínuo")

RULES_CSV = DATA_DIR / "rules_map.csv"
OVR_CSV   = DATA_DIR / "whitelist_overrides.csv"

def _load_or_init_rules() -> pd.DataFrame:
    cols = ["ativo", "regex", "equipamento"]
    if RULES_CSV.exists():
        try:
            df = pd.read_csv(RULES_CSV, dtype=str).fillna("")
            for c in cols:
                if c not in df.columns: df[c] = ""
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

def _save_rules(df: pd.DataFrame):
    df = df.copy()
    for c in ["ativo", "regex", "equipamento"]:
        if c not in df.columns: df[c] = ""
    df.to_csv(RULES_CSV, index=False, encoding="utf-8-sig")

def _load_or_init_overrides() -> pd.DataFrame:
    cols = ["ativo", "equipamento_canonico"]
    if OVR_CSV.exists():
        try:
            df = pd.read_csv(OVR_CSV, dtype=str).fillna("")
            for c in cols:
                if c not in df.columns: df[c] = ""
            return df[cols].drop_duplicates()
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

def _save_overrides(df: pd.DataFrame):
    df = df.copy()
    for c in ["ativo", "equipamento_canonico"]:
        if c not in df.columns: df[c] = ""
    df = df[df["equipamento_canonico"].astype(str).str.strip()!=""]
    df.to_csv(OVR_CSV, index=False, encoding="utf-8-sig")

rules_df = _load_or_init_rules()
ovr_df   = _load_or_init_overrides()

# --- Universo de edição: lacunas (ambíguo/descartado/NAO_CLASSIFICADO) ---
if log.empty:
    st.info("Sem LOG carregado para correção manual.")
else:
    # identifica NAO_CLASSIFICADO no eventos_qualificados
    evt_nao = pd.DataFrame()
    if not evt.empty and "equipamento" in evt.columns:
        evt_nao = evt[evt["equipamento"].astype(str).str.upper().eq(NAO_CLASSIFICADO)]

    lacunas_ids = set()
    if not log.empty:
        lacunas_ids |= set(log.loc[log["status"].isin(["ambíguo","descartado"]), "id_evento"].dropna().astype(int).tolist())
    if not evt_nao.empty:
        lacunas_ids |= set(evt_nao["id_evento"].dropna().astype(int).tolist())

    # junta infos para correção
    base_corr = pd.merge(
        log, evt[["id_evento","equipamento"]].rename(columns={"equipamento":"equipamento_atual"}),
        on="id_evento", how="left"
    )
    base_corr = base_corr[base_corr["id_evento"].isin(lacunas_ids)].copy()

    st.caption(f"Lacunas disponíveis para correção: {len(base_corr)} eventos")
    with st.expander("Ver amostra das lacunas", expanded=False):
        show_cols = ["id_evento","ativo","data_evento",
                     "menção_bruta" if "menção_bruta" in base_corr.columns else "mencao_bruta",
                     "equipamento_atual","status","fonte","score","regra_aplicada",
                     "origem_classificacao","confianca","motivo"]
        show_cols = [c for c in show_cols if c in base_corr.columns]
        st.dataframe(base_corr[show_cols].sort_values(["ativo","data_evento"]).head(100), use_container_width=True)

    # ========= 1) Correção manual de um evento =========
    st.subheader("1) Correção manual pontual")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        evento_id = st.number_input("id_evento", min_value=0, value=0, step=1)
    with c2:
        ativos_all = sorted(base_corr["ativo"].dropna().unique().tolist())
        ativo_sel_cm = st.selectbox("Ativo", options=["(auto)"] + ativos_all, index=0)
    with c3:
        equipamento_novo = st.text_input("Equipamento (canônico desejado)", value="", placeholder="Ex.: VFD, MC A, BOMBA DE INJEÇÃO C …")
    with c4:
        st.caption("Dica: você pode digitar livre; para consolidar no dicionário, use 'Salvar correção' e reprocessar.")

    row_sel = base_corr[base_corr["id_evento"]==evento_id].head(1)
    if evento_id and row_sel.empty:
        st.warning("id_evento não encontrado nas lacunas atuais.")
    else:
        if not row_sel.empty:
            st.write("Justificativa:", st.code(str(row_sel["menção_bruta" if "menção_bruta" in base_corr.columns else "mencao_bruta"].iloc[0]), language=None))
            st.write("Equipamento atual:", str(row_sel["equipamento_atual"].iloc[0]))

        colb1, colb2 = st.columns([1,1])
        with colb1:
            if st.button("💾 Salvar correção (override canônico)"):
                if not row_sel.empty:
                    ativo_eff = row_sel["ativo"].iloc[0] if ativo_sel_cm=="(auto)" else ativo_sel_cm
                    new = pd.DataFrame([{"ativo": ativo_eff, "equipamento_canonico": equipamento_novo.strip()}])
                    ovr_df = pd.concat([ovr_df, new], ignore_index=True).drop_duplicates()
                    _save_overrides(ovr_df)
                    st.success(f"Override salvo em {OVR_CSV.name}: ({ativo_eff}, {equipamento_novo}).")
                else:
                    st.warning("Selecione um id_evento válido.")

        with colb2:
            if st.button("🔁 Reprocessar (ingestão + matching) com correções"):
                with st.spinner("Reprocessando…"):
                    run_ingestao()
                    # reprocessa já usando overrides e NAO_CLASSIFICADO para manter cobertura
                    aplicar_matching(use_review_pipeline=False, keep_unclassified=True)
                load_eventos_qualificados.clear(); load_log_matching.clear()
                st.success("Pipeline reprocessado. Recarregue os dados (ou clique em Reprocessar no topo).")

    st.markdown("---")

    # ========= 2) Criar regra (regex → equipamento) =========
    st.subheader("2) Criar regra (regex → equipamento)")
    colr1, colr2, colr3 = st.columns([1.2, 1.2, 0.6])
    with colr1:
        ativo_rule = st.selectbox("Ativo alvo da regra", options=["*"] + ativos_all, index=0)
    with colr2:
        regex_rule = st.text_input("Regex (Python, case‑insensitive)", value=r"\bvfd\b", placeholder=r"\btrip\b.*\bhp\b")
    with colr3:
        equip_rule = st.text_input("Equipamento canônico", value="VFD")

    # Ajuda: gerar regex a partir de um id_evento
    with st.expander("Ajudar a criar a regex a partir de um evento"):
        ev_for_rule = st.number_input("id_evento (exemplo)", min_value=0, value=0, step=1, key="ev_for_rule")
        row_r = base_corr[base_corr["id_evento"]==ev_for_rule].head(1)
        if not row_r.empty:
            raw_txt = str(row_r["menção_bruta" if "menção_bruta" in base_corr.columns else "mencao_bruta"].iloc[0])
            st.write("Texto bruto:", st.code(raw_txt, language=None))
            toks = re.findall(r"[A-Za-z0-9\-]{4,}", raw_txt)
            sug = r"\b" + re.escape(toks[0]) + r"\b" if toks else r"\bpalavra\b"
            st.caption(f"Sugestão de ponto de partida: `{sug}`")

    colr4, colr5 = st.columns([1,1])
    with colr4:
        if st.button("🧪 Validar regex nas lacunas"):
            try:
                rx = re.compile(regex_rule, re.IGNORECASE)
            except re.error as e:
                st.error(f"Regex inválida: {e}")
                rx = None
            if rx is not None:
                sub = base_corr.copy()
                if ativo_rule != "*":
                    sub = sub[sub["ativo"]==ativo_rule]
                col_m = "menção_bruta" if "menção_bruta" in sub.columns else "mencao_bruta"
                hits = sub[sub[col_m].astype(str).str.contains(rx)]
                st.success(f"Hits: {len(hits)} (mostrando até 50)")
                st.dataframe(hits[["id_evento","ativo","data_evento",col_m,"equipamento_atual"]].head(50), use_container_width=True)

    with colr5:
        if st.button("💾 Gravar regra no rules_map.csv"):
            try:
                re.compile(regex_rule, re.IGNORECASE)
            except re.error as e:
                st.error(f"Regex inválida: {e}")
            else:
                new = pd.DataFrame([{"ativo": ativo_rule, "regex": regex_rule, "equipamento": equip_rule.strip()}])
                rules_df = pd.concat([rules_df, new], ignore_index=True).drop_duplicates()
                _save_rules(rules_df)
                st.success(f"Regra gravada em {RULES_CSV.name}: ({ativo_rule}, {regex_rule}) → {equip_rule}")

    st.markdown("---")

    # ========= 3) Aceitar em lote sugestões do agente =========
    st.subheader("3) Aceitar em lote sugestões do agente")
    lo = log.copy()
    if "origem_classificacao" in lo.columns and "confianca" in lo.columns:
        lo_ok = lo[(lo["origem_classificacao"].astype(str).str.contains("interpretador", case=False, na=False)) &
                   (pd.to_numeric(lo["confianca"], errors="coerce") >= 0.60)]
        st.caption(f"Sugestões elegíveis (confiança ≥ 0.60): {len(lo_ok)}")
        lim = st.slider("Limiar de confiança para aceite em lote", 0.0, 1.0, 0.70, 0.01, key="aceite_lote")
        if st.button("✅ Gravar canônicos sugeridos (override por ativo)"):
            if "equipamento_canonizado" not in lo_ok.columns:
                st.warning("Coluna equipamento_canonizado não disponível no LOG.")
            else:
                acc = lo_ok[pd.to_numeric(lo_ok["confianca"], errors="coerce") >= lim]
                if acc.empty:
                    st.info("Nenhuma sugestão acima do limiar.")
                else:
                    ins = acc[["ativo","equipamento_canonizado"]].rename(columns={"equipamento_canonizado":"equipamento_canonico"}).dropna()
                    ins["equipamento_canonico"] = ins["equipamento_canonico"].astype(str).str.strip()
                    ins = ins[ins["equipamento_canonico"]!=""]
                    ovr_df2 = pd.concat([ovr_df, ins], ignore_index=True).drop_duplicates()
                    _save_overrides(ovr_df2)
                    st.success(f"Salvo {len(ins)} overrides em {OVR_CSV.name}. Reprocesse o pipeline para refletir.")

    # ========= 4) Botão rápido de reprocessar após edições =========
    if st.button("⚙️ Reprocessar agora com regras/overrides"):
        with st.spinner("Reprocessando…"):
            run_ingestao()
            aplicar_matching(use_review_pipeline=False, keep_unclassified=True)
        load_eventos_qualificados.clear(); load_log_matching.clear()
        st.success("Concluído. Recarregue os dados (ou clique em Reprocessar no topo).")

# Rodapé
st.caption("© PRIO — Dashboard de Criticidade de Equipamentos. Matching clássico + IA (opcional) nas lacunas, com correção manual e aprendizado contínuo.")
