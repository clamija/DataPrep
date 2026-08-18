from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from config import DATASET_CONFIGS, get_config
from train import run_pipeline
from reporting import (
    cleaning_report_table,
    data_quality_score,
    decisions_table,
    improvement_text,
    metrics_comparison_table,
    per_column_table,
)
from utils import dataframe_profile, safe_drop_columns


st.set_page_config(page_title="DataPrep", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1200px; }
    h1 { color: #2B6CB0 !important; }
    div.stButton > button,
    div.stButton > button:hover,
    div.stButton > button:focus,
    div.stButton > button:active,
    div.stButton > button:focus:not(:active) {
        background-color: #2B6CB0 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.45rem 1.1rem;
        box-shadow: none !important;
    }
    [data-testid="stMetric"] {
        background: #f7fafc;
        border: 1px solid #edf2f7;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #4a5568 !important;
    }
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #2B6CB0 !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {
        color: #2B6CB0 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #2B6CB0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("DataPrep")
st.caption("Automatska priprema tabelarnih podataka")
st.divider()


def _load_dataset_from_config(config: dict) -> pd.DataFrame:
    dataset_path = config.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path nije definisan u config profilu.")
    return pd.read_csv(dataset_path)


def _plot_confusion_matrices(baseline_cm: np.ndarray, cleaned_cm: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, cm, title in (
        (axes[0], baseline_cm, "BASELINE"),
        (axes[1], cleaned_cm, "CLEANED"),
    ):
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Šta je model predvidio", fontsize=10)
        ax.set_ylabel("Šta je zaista u podacima", fontsize=10)
        ax.set_xticklabels(["Ne (0)", "Da (1)"], fontsize=9)
        ax.set_yticklabels(["Ne (0)", "Da (1)"], fontsize=9)
    fig.tight_layout()
    return fig


st.subheader("Odaberi dataset")
select_col, _ = st.columns([1, 1])
with select_col:
    dataset_name = st.selectbox(
        "Odaberi dataset",
        list(DATASET_CONFIGS.keys()),
        label_visibility="collapsed",
    )
config = get_config(dataset_name)

if dataset_name == "custom":
    uploaded = st.file_uploader("Upload CSV dataset (custom profil)", type=["csv"])
    if uploaded is None:
        st.info("Za custom profil uploaduj CSV i unesi target kolonu.")
        st.stop()
    df = pd.read_csv(uploaded)
    with select_col:
        config["target_column"] = st.text_input("Target kolona", value="")
        drop_cols_text = st.text_input("Kolone za drop (zarezom odvojeno)", value="")
    config["drop_columns"] = [c.strip() for c in drop_cols_text.split(",") if c.strip()]
else:
    df = _load_dataset_from_config(config)

st.subheader("Osnovne statistike")
profile = dataframe_profile(df)
c1, c2 = st.columns(2)
c1.metric("Redovi", profile["rows"])
c2.metric("Kolone", profile["columns"])

with st.expander("Tipovi kolona (opcionalno)"):
    st.caption("Pregled tipova podataka po kolonama. Nije potreban za pokretanje eksperimenta.")
    st.json(profile["dtypes"])

st.subheader("Postavke ovog skupa")
target = config.get("target_column") or "—"
drop_cols = config.get("drop_columns") or []
drop_text = ", ".join(drop_cols) if drop_cols else "nema"
st.markdown(
    f"- **Šta predviđamo:** `{target}`\n"
    f"- **Šta izbacujemo:** {drop_text}\n"
)

run_clicked = st.button("Pokreni BASELINE vs CLEANED eksperiment")
if run_clicked:
    if not config.get("target_column"):
        st.error("Unesi target kolonu za custom dataset.")
        st.stop()
    try:
        payload = run_pipeline(df, config)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    tab_quality, tab_model, tab_pipeline, tab_cm = st.tabs(
        ["Kvalitet podataka", "Metrike modela", "Šta je pipeline uradio?", "Koliko često model pogodi?"]
    )

    with tab_quality:
        st.caption(
            "Veći broj označava čišće podatke."
        )
        raw_for_score = safe_drop_columns(df, config.get("drop_columns", []))
        raw_score = data_quality_score(raw_for_score)
        cleaned_score = data_quality_score(
            payload["cleaned_df"],
            outlier_bounds=payload["decisions"].get("outlier_bounds"),
        )
        q1, q2 = st.columns(2)
        q1.metric("RAW (sirov dataset)", f"{raw_score:.2f}/100")
        q2.metric("CLEANED", f"{cleaned_score:.2f}/100", delta=f"{cleaned_score - raw_score:+.2f}")

    with tab_model:
        st.caption(
            "BASELINE ima ujednačen tekst, bez duplikata, te je obavljeno i skaliranje i one-hot encoding. Nepotpuni train redovi "
            "su se odbacili. CLEANED dodatno vrši imputaciju i tretira outliere. "
            "**Test set je isti za oba**, dok train može biti veći kod CLEANED-a jer taj scenario koristi popunjene redove."
        )
        table = metrics_comparison_table(payload["results"])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.success(improvement_text(payload["results"]))
        n_test = int(payload["results"]["cleaned"]["n_test"])
        if n_test < 80:
            st.warning(
                "Test skup je mali jer se ocjenjuju samo redovi koji su već bili kompletni (bez praznih polja). "
                "Razlog velike razlike u F1 ponekad može biti kolona sa puno nedostajućih vrijednosti."
            )

    with tab_pipeline:
        st.markdown("**Obrađeni problemi**")
        st.dataframe(
            cleaning_report_table(payload["cleaning_report"]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Odluke o pripremi modela**")
        st.caption("Numeričke, tekstualne, skalirane i izbačene kolone.")
        st.dataframe(decisions_table(payload["decisions"]), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        missing_tbl = per_column_table(payload["decisions"]["missing_fixed_by_column"])
        outlier_tbl = per_column_table(payload["decisions"]["outliers_treated_by_column"])
        with d1:
            st.markdown("**Gdje su popunjene praznine:**")
            if missing_tbl.empty:
                st.info("Nije bilo praznih polja za popuniti.")
            else:
                st.dataframe(missing_tbl, use_container_width=True, hide_index=True)
        with d2:
            st.markdown("**Gdje su ograničene ekstremne vrijednosti:**")
            if outlier_tbl.empty:
                st.info("Nije bilo ekstremnih vrijednosti za ograničiti.")
            else:
                st.dataframe(outlier_tbl, use_container_width=True, hide_index=True)

    with tab_cm:
        st.markdown(
            """
Lijevo i desno su **isti test primjeri**.

**Da** = ono što predviđamo (npr. putnik je preživio, korisnik je napustio uslugu).  
**Ne** = to se nije dogodilo.

Objašnjenje:

- **Gore lijevo** - model je rekao *ne* i bio je u pravu.
- **Gore desno** - model je rekao *da*, a nije bilo tako.
- **Dolje lijevo** - model je rekao *ne*, a trebalo je *da*.
- **Dolje desno** - model je rekao *da* i bio je u pravu.

            """
        )
        baseline_cm = np.array(payload["results"]["baseline"]["confusion_matrix"])
        clean_cm = np.array(payload["results"]["cleaned"]["confusion_matrix"])
        cm_col, _ = st.columns([7, 3])
        with cm_col:
            st.pyplot(_plot_confusion_matrices(baseline_cm, clean_cm), use_container_width=True)
