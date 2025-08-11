# app/app.py
import os
from pathlib import Path
from io import BytesIO
import sys
import re

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

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


# ---------- Coerção numérica segura ----------
# Não mexe em números já-numéricos, só converte "pt-BR" se tiver vírgula,
# e NUNCA altera strings em notação científica (e.g., "1.23e+05").
SCI_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$')

def coerce_number_ptbr(series: pd.Series) -> pd.Series:
    # Numérico -> só coerce
    if hasattr(series, "dtype") and series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # notação científica -> to_numeric direto (não tocar)
    mask_sci = s.str.match(SCI_RE)

    # strings com vírgula decimal -> remover separador de milhar "." e trocar "," por "."
    mask_pt = (~mask_sci) & s.str.contains(",", na=False)
    s = s.where(~mask_pt, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

    # demais casos -> to_numeric direto
    return pd.to_numeric(s, errors="coerce")
# ---------------------------------------------


@st.cache_data(show_spinner=False)
def load_eventos_qualificados() -> pd.DataFrame:
    """Lê o parquet final e corrige tipos:
       - data_evento -> datetime
       - periodo_h   -> horas (float)
       - bbl         -> inteiro (Int64)
    """
    fp = OUT_DIR / "eventos_qualificados.parquet"
    if not fp.exists():
        return pd.DataFrame(columns=[
            "ativo", "data_evento", "equipamento", "periodo_h", "bbl", "justificativa"
        ])

    df = pd.read_parquet(fp)

    # Datas
    if "data_evento" in df.columns:
        df["data_evento"] = pd.to_datetime(df["data_evento"], errors="coerce")

    # Horas
    if "periodo_h" in df.columns:
        df["periodo_h"] = coerce_number_ptbr(df["periodo_h"])

    # bbl inteiro
    if "bbl" in df.columns:
        df["bbl"] = coerce_number_ptbr(df["bbl"]).round().astype("Int64")

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


@st.cache_data(show_spinner=False)
def load_rankings_precomp():
    files = {
        "horas": OUT_DIR / "rank_por_horas.parquet",
        "bbl": OUT_DIR / "rank_por_bbl.parquet",
        "usd": OUT_DIR / "rank_por_usd.parquet",
        "comp": OUT_DIR / "rank_composto.parquet",
    }
    dfs = {}
    for k, p in files.items():
        if p.exists():
            dfs[k] = pd.read_parquet(p)
    return dfs


# =============================
# Garantir dados
# =============================
ensure_pipeline()

# =============================
# Sidebar (filtros e parâmetros)
# =============================
st.title("Equipamentos — Eventos Qualificados")

evt = load_eventos_qualificados()
for col in ("bbl", "periodo_h"):
    if col not in evt.columns:
        evt[col] = np.nan

log = load_log_matching()
precomp = load_rankings_precomp()

st.sidebar.header("Parâmetros")
# Reprocessamento manual
if st.sidebar.button("🔄 Reprocessar dados"):
    if ensure_pipeline():
        load_eventos_qualificados.clear()
        load_log_matching.clear()
        load_rankings_precomp.clear()
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
# Brent - aplicar
# =============================
def apply_brent(df: pd.DataFrame, brent_value: float | None, brent_series: dict[str, float] | None):
    df = df.copy()
    if df.empty:
        df["brent_usd"] = np.nan
        df["perda_financeira_usd"] = 0.0
        return df

    # garante numérico correto para bbl
    if "bbl" in df.columns:
        df["bbl"] = coerce_number_ptbr(df["bbl"])

    if brent_series:
        yyyymm = df["data_evento"].dt.strftime("%Y-%m")
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

# Garantir tipos numéricos ANTES dos KPIs/aggregations
for col in ["periodo_h", "bbl", "perda_financeira_usd", "brent_usd"]:
    if col in evt_f.columns:
        evt_f[col] = coerce_number_ptbr(evt_f[col])

# =============================
# KPIs
# =============================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Eventos qualificados", f"{len(evt_f):,}".replace(",", "."))
col2.metric("Horas paradas", f"{evt_f['periodo_h'].sum():,.2f}".replace(",", "."))
col3.metric("bbl perdidos", f"{evt_f['bbl'].sum():,.2f}".replace(",", "."))
col4.metric("Perda financeira (USD)", f"{evt_f['perda_financeira_usd'].sum():,.2f}".replace(",", "."))

st.caption("*Se a série mensal não cobrir algum mês, usamos o valor único de Brent informado no sidebar. Valores resultantes nesses meses são marcados como estimativas.*")

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
    st.subheader("Eventos qualificados (após matching)")
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
st.caption("© PRIO — Dashboard de Criticidade de Equipamentos. Valores financeiros dependem do Brent informado/série mensal. Bravo≡TBMT e ABL≡Forte já normalizados no pipeline.")
