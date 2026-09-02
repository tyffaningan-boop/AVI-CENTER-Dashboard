# -*- coding: utf-8 -*-
"""
AVI CENTER CAMEROUN — DASHBOARD D'ANALYSE ET DE PRÉVISION
Architecture :
fichier Excel/CSV -> lecteur_universel -> préparation -> XGBoost -> prévisions -> dashboard

XGBoost est le seul modèle opérationnel.
La comparaison des modèles n'est pas relancée dans le dashboard.
Horizon ferme retenu dans l'étude : 4 mois.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lecteur_universel import normaliser_donnees
from prevision_xgboost import executer_prevision


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AVI CENTER — Analyse prévisionnelle",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
HORIZON = 12
HORIZON_FIABLE = 4
TARIF_DEFAUT = 310000

LOGO_CANDIDATES = [
    BASE_DIR / "LOGO.webp",
    BASE_DIR / "LOGO.png",
    BASE_DIR / "LOGO.jpg",
    BASE_DIR / "LOGO.jpeg",
]
CITY_CANDIDATES = [
    BASE_DIR / "Yaounde.webp",
    BASE_DIR / "Yaounde.png",
    BASE_DIR / "Yaounde.jpg",
    BASE_DIR / "Yaounde.jpeg",
]

PAGES = [
    "Accueil",
    "Analyse descriptive",
    "Prévision",
    "Recommandations",
]

MOIS = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}


# ============================================================
# VISUELS
# ============================================================

def image_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/webp"
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(
        path.read_bytes()
    ).decode("ascii")


def svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(
        svg.encode("utf-8")
    ).decode("ascii")


def mascot_svg() -> str:
    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="180" height="195"
         viewBox="0 0 180 195">
      <ellipse cx="92" cy="185" rx="47" ry="7" fill="#061728" opacity=".4"/>
      <circle cx="82" cy="58" r="36" fill="#efbc91"/>
      <path d="M48 55c2-26 20-42 40-42 24 0 39 19 38 43
      -11-8-22-13-36-14-11 9-26 14-42 13z" fill="#382419"/>
      <circle cx="70" cy="59" r="4.5" fill="#1a2430"/>
      <circle cx="94" cy="59" r="4.5" fill="#1a2430"/>
      <path d="M71 77c7 8 18 9 26 0" fill="none"
            stroke="#873f2c" stroke-width="4" stroke-linecap="round"/>
      <path d="M66 98h51c14 0 24 10 24 25v30H52v-30c0-15 4-25 14-25z"
            fill="#153e69"/>
      <rect x="70" y="117" width="42" height="18" rx="5" fill="#fff"/>
      <text x="91" y="130" text-anchor="middle" font-family="Arial"
            font-size="10" font-weight="700" fill="#163b67">AVI</text>
      <path d="M61 153h25l-7 28H57z" fill="#273c56"/>
      <path d="M96 153h25l8 28h-22z" fill="#273c56"/>
      <path d="M53 183h30M105 183h28" stroke="#fff"
            stroke-width="7" stroke-linecap="round"/>
      <path d="M129 58v51" stroke="#e8e8e8" stroke-width="3"/>
      <path d="M131 60h35v23h-35z" fill="#fff"/>
      <path d="M131 60h12v23h-12z" fill="#1d438c"/>
      <path d="M154 60h12v23h-12z" fill="#d43140"/>
    </svg>
    """


def flag_cm_svg() -> str:
    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="165" height="84">
      <rect width="165" height="84" rx="9" fill="#0b1f36" opacity=".1"/>
      <rect x="11" y="9" width="48" height="66" fill="#128a47"/>
      <rect x="59" y="9" width="47" height="66" fill="#d42d3c"/>
      <rect x="106" y="9" width="48" height="66" fill="#e9c92e"/>
      <polygon points="82,26 85,36 95,36 87,42 90,52
                       82,46 74,52 77,42 69,36 79,36"
               fill="#f7db4b"/>
    </svg>
    """


def style() -> None:
    logo = next((p for p in LOGO_CANDIDATES if p.exists()), None)
    city = next((p for p in CITY_CANDIDATES if p.exists()), None)

    bg = (
        f'.stApp{{background:linear-gradient(rgba(255,255,255,.86),'
        f'rgba(255,255,255,.92)),url("{image_uri(city)}") center/cover fixed no-repeat;}}'
        if city else
        ".stApp{background:#f4f6f8;}"
    )

    logo_html = (
        f'<img src="{image_uri(logo)}" class="brand-logo" alt="AVI CENTER">'
        if logo else
        '<div class="brand-placeholder">AVI CENTER</div>'
    )

    st.markdown(
        f"""
        <style>
        {bg}
        html,body,[class*="css"]{{font-family:"Segoe UI",Arial,sans-serif;}}
        [data-testid="stSidebar"]{{background:#0b2747;border-right:1px solid rgba(255,255,255,.09);}}
        [data-testid="stSidebar"]>div:first-child{{padding-top:1rem;}}
        .sidebar-mascot{{display:flex;justify-content:center;}}
        .sidebar-slogan{{color:#f4f7fb;font-family:Georgia,serif;font-size:1.02rem;
        font-style:italic;line-height:1.4;text-align:center;}}
        .side-rule{{height:1px;background:rgba(255,255,255,.16);margin:1rem 0;}}
        .nav-label{{color:rgba(255,255,255,.52);font-size:.7rem;
        letter-spacing:.16em;text-transform:uppercase;margin-bottom:.5rem;}}
        .cameroon-flag{{display:flex;justify-content:center;margin-top:1.5rem;}}
        .page-title{{color:#0a2c53;font-size:clamp(1.8rem,3.3vw,3rem);
        line-height:1.05;font-weight:780;letter-spacing:.015em;margin:0 0 .25rem;}}
        .page-subtitle{{color:#607285;font-size:.9rem;margin-bottom:.7rem;}}
        .brand-zone{{display:flex;justify-content:center;margin:.2rem 0 .8rem;}}
        .brand-logo{{max-width:230px;max-height:72px;object-fit:contain;}}
        .brand-placeholder{{padding:.55rem .9rem;border:1px dashed #99a9b9;
        color:#687b8d;border-radius:8px;background:rgba(255,255,255,.78);}}
        .panel{{background:rgba(255,255,255,.94);border:1px solid rgba(12,43,81,.09);
        border-radius:10px;padding:1rem 1.05rem;box-shadow:0 10px 26px rgba(18,45,75,.07);margin-bottom:1rem;}}
        .panel-title{{color:#607286;font-size:.74rem;letter-spacing:.1em;
        text-transform:uppercase;font-weight:750;margin-bottom:.45rem;}}
        .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.8rem 0 1rem;}}
        .kpi{{background:rgba(255,255,255,.97);border:1px solid rgba(14,44,77,.08);
        border-radius:9px;padding:1rem;box-shadow:0 8px 19px rgba(18,45,75,.07);
        position:relative;overflow:hidden;}}
        .kpi:before{{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:#153f6b;}}
        .kpi-label{{color:#6d7e8c;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;}}
        .kpi-value{{color:#0a2c53;font-size:1.55rem;font-weight:780;margin-top:.25rem;}}
        .kpi-note{{color:#83919e;font-size:.74rem;margin-top:.15rem;}}
        .metric-line{{border-left:4px solid #1f4f7e;background:rgba(255,255,255,.92);
        border-radius:7px;padding:.7rem .85rem;margin-bottom:.55rem;}}
        .metric-small{{color:#728393;font-size:.71rem;text-transform:uppercase;letter-spacing:.08em;}}
        .metric-large{{color:#0b3159;font-size:1.15rem;font-weight:760;}}
        .notice{{background:rgba(255,255,255,.94);border-left:5px solid #315d87;
        border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem;color:#53697c;}}
        [data-testid="stSidebar"] .stButton>button{{width:100%;background:rgba(255,255,255,.055)!important;
        border:1px solid rgba(255,255,255,.13)!important;color:#f2f7fc!important;font-weight:650!important;
        transition:background .15s ease,border-color .15s ease,box-shadow .15s ease;}}
        [data-testid="stSidebar"] .stButton>button:hover{{background:rgba(255,255,255,.15)!important;
        border-color:rgba(255,255,255,.38)!important;transform:translateY(-1px);}}
        [data-testid="stSidebar"] .stButton>button[kind="primary"],
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{{
        background:rgba(255,255,255,.26)!important;
        border:1px solid rgba(255,255,255,.62)!important;
        color:#ffffff!important;font-weight:770!important;
        box-shadow:0 0 0 1px rgba(255,255,255,.18) inset,0 0 16px rgba(255,255,255,.28)!important;
        }}
        [data-testid="stSidebar"] .stButton>button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover{{
        background:rgba(255,255,255,.32)!important;
        }}
        @media(max-width:1000px){{.kpi-grid{{grid-template-columns:repeat(2,1fr);}}}}
        @media(max-width:650px){{.kpi-grid{{grid-template-columns:1fr;}}}}
        </style>
        <div class="page-title">ANALYSE PRÉVISIONNELLE D’AVI CENTER CAMEROUN</div>
        <div class="page-subtitle">Pilotage commercial • historique • prévision • aide à la décision</div>
        <div class="brand-zone">{logo_html}</div>
        """,
        unsafe_allow_html=True,
    )


style()


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Accueil"
if "resultats" not in st.session_state:
    st.session_state.resultats = None
if "nom_fichier" not in st.session_state:
    st.session_state.nom_fichier = "Aucun fichier importé"
if "tarif" not in st.session_state:
    st.session_state.tarif = TARIF_DEFAUT


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-mascot"><img src="{svg_uri(mascot_svg())}" width="145"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-slogan">Avec <b>AVI CENTER</b>,<br>simplifiez-vous l’AVI</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    for page in PAGES:
        est_actif = st.session_state.page == page
        if st.button(
            page,
            key="nav_" + page,
            use_container_width=True,
            type="primary" if est_actif else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

    st.markdown(
        f'<div class="cameroon-flag"><img src="{svg_uri(flag_cm_svg())}" width="150"></div>',
        unsafe_allow_html=True,
    )
    st.caption("Outil de pilotage prévisionnel")


# ============================================================
# IMPORT — visible et actif uniquement sur la page Accueil
# ============================================================

uploaded = None
tarif = st.session_state.tarif
lancer = False

if st.session_state.page == "Accueil":

    st.markdown(
        '<div class="panel"><div class="panel-title">Actualisation de l’analyse</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.35, 1.0, .95])

    with c1:
        uploaded = st.file_uploader(
            "Importer une base Excel ou CSV",
            type=["xlsx", "xls", "csv"],
            help="Le nom du fichier n'a aucune importance.",
        )

    with c2:
        tarif = st.number_input(
            "Tarif unitaire (FCFA)",
            min_value=0,
            value=int(st.session_state.tarif),
            step=5000,
        )

    with c3:
        st.write("")
        st.write("")
        lancer = st.button(
            "ACTUALISER L’ANALYSE",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        "Le lecteur universel standardise la base. "
        "L'actualisation entraîne uniquement XGBoost et génère les prévisions."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ACTUALISATION
# ============================================================

if uploaded is not None:
    st.session_state.nom_fichier = uploaded.name

if lancer:
    if uploaded is None:
        default = BASE_DIR / "Base.xlsx"
        if not default.exists():
            st.error("Importe un fichier Excel ou CSV pour commencer.")
            st.stop()
        raw_bytes = default.read_bytes()
        source_name = default.name
    else:
        raw_bytes = uploaded.getvalue()
        source_name = uploaded.name

    try:
        with st.spinner("Lecture, préparation et prévision XGBoost en cours..."):

            with tempfile.TemporaryDirectory() as td:
                source = Path(td) / source_name
                source.write_bytes(raw_bytes)

                # 1. Lecteur universel
                standard = normaliser_donnees(source)

                # 2. Les modules de préparation/entraînement validés
                # utilisent Années | Périodes | volume.
                standard_file = Path(td) / "base_standardisee.xlsx"

                standard[
                    ["Années", "Périodes", "volume"]
                ].to_excel(
                    standard_file,
                    index=False,
                    engine="openpyxl",
                )

                # Le fichier Excel est complètement fermé avant la lecture suivante.
                modele, df_ml, previsions_brutes = executer_prevision(
                    chemin_fichier=standard_file,
                    horizon=HORIZON,
                )

            historique = standard.copy()
            historique["ca"] = historique["volume"] * float(tarif)

            previsions = previsions_brutes.copy()
            previsions["horizon"] = np.arange(1, len(previsions) + 1)
            previsions["statut"] = np.where(
                previsions["horizon"] <= HORIZON_FIABLE,
                "Ferme",
                "Indicatif",
            )
            previsions["ca_prevu"] = (
                previsions["volume_pred"] * float(tarif)
            )
            previsions["ca_prevu_millions"] = (
                previsions["ca_prevu"] / 1_000_000
            )

            fermes = previsions[previsions["statut"] == "Ferme"]

            volume_ferme = float(fermes["volume_pred"].sum())
            volume_total = float(previsions["volume_pred"].sum())
            ca_ferme = float(fermes["ca_prevu"].sum())
            ca_total = float(previsions["ca_prevu"].sum())

            mean_hist = float(historique["volume"].mean())
            mean_forecast = float(previsions["volume_pred"].mean())

            evolution = (
                (mean_forecast - mean_hist) / mean_hist * 100
                if mean_hist else np.nan
            )

            st.session_state.resultats = {
                "historique": historique,
                "historique_ml": df_ml.copy(),
                "previsions": previsions,
                "modele": modele,
                "source": source_name,
                "horizon_fiable": HORIZON_FIABLE,
                "tarif": float(tarif),
                "resume": {
                    "volume_ferme": volume_ferme,
                    "volume_total": volume_total,
                    "ca_ferme": ca_ferme,
                    "ca_total": ca_total,
                    "evolution": evolution,
                },
            }

            st.session_state.tarif = int(tarif)
            st.session_state.nom_fichier = source_name
            st.session_state.page = "Accueil"

        st.success("Analyse actualisée avec succès avec XGBoost.")

    except Exception as exc:
        st.error("Impossible d'actualiser l'analyse.")
        st.exception(exc)


# ============================================================
# AUCUNE DONNÉE
# ============================================================

results = st.session_state.resultats

if results is None:
    st.markdown(
        """
        <div class="panel">
        <div class="panel-title">Bienvenue dans l'outil de pilotage AVI CENTER</div>
        <div class="notice">
        Rendez-vous sur l'onglet <b>Accueil</b>, importez une base Excel ou CSV
        puis cliquez sur <b>ACTUALISER L’ANALYSE</b>.
        Le système reconnaît la structure temporelle, prépare les données
        et produit les prévisions avec <b>XGBoost</b>.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


df = results["historique"].copy()
df_ml = results["historique_ml"].copy()
previsions = results["previsions"].copy()
horizon_fiable = int(results["horizon_fiable"])
resume = results["resume"]


# ============================================================
# OUTILS
# ============================================================

def money(v):
    if pd.isna(v):
        return "N/D"
    v = float(v)
    if abs(v) >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f} Md FCFA"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f} M FCFA"
    return f"{v:,.0f} FCFA".replace(",", " ")


def number(v):
    return f"{float(v):,.0f}".replace(",", " ")


def history_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["volume"],
        mode="lines+markers", name="Volume réel",
        line=dict(color="#174b7a", width=3),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Volume : %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=410, margin=dict(l=8,r=8,t=20,b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.76)",
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Dossiers", gridcolor="rgba(15,45,75,.08)"),
        legend=dict(orientation="h", y=1.07, x=0),
    )
    return fig


def forecast_chart(hist, fut):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["volume"],
        mode="lines+markers", name="Historique réel",
        line=dict(color="#173f67", width=3),
        marker=dict(size=4.5),
    ))

    ferme = fut[fut["statut"] == "Ferme"]
    indic = fut[fut["statut"] == "Indicatif"]

    if not ferme.empty:
        fig.add_trace(go.Scatter(
            x=ferme["date"], y=ferme["volume_pred"],
            mode="lines+markers", name="Prévision ferme",
            line=dict(color="#c17a2c", width=3, dash="dash"),
            marker=dict(size=6),
        ))

    if not indic.empty:
        fig.add_trace(go.Scatter(
            x=indic["date"], y=indic["volume_pred"],
            mode="lines+markers", name="Prévision indicative",
            line=dict(color="#7d8792", width=2, dash="dot"),
            marker=dict(size=5),
        ))

    fig.add_vline(
        x=hist["date"].max(),
        line_width=1,
        line_dash="dot",
        line_color="#7d8792",
    )

    fig.update_layout(
        height=470, margin=dict(l=8,r=8,t=20,b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.76)",
        hovermode="x unified",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Volume de dossiers",
                   gridcolor="rgba(15,45,75,.08)"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def stats(data):
    s = data["volume"].describe(percentiles=[.25,.5,.75])
    return pd.DataFrame({
        "Indicateur": [
            "Observations","Moyenne","Médiane","Minimum",
            "Maximum","Écart-type","1er quartile","3e quartile"
        ],
        "Valeur": [
            int(s["count"]),s["mean"],s["50%"],s["min"],
            s["max"],s["std"],s["25%"],s["75%"]
        ],
    })


def annual_growth(data, year):
    current = data.loc[data["date"].dt.year == year, "volume"].sum()
    previous = data.loc[data["date"].dt.year == year-1, "volume"].sum()
    if previous == 0:
        return np.nan
    return (current-previous)/previous*100


def monthly_growth(data, year, month):
    current = data.loc[
        (data["date"].dt.year == year) & (data["date"].dt.month == month),
        "volume",
    ].sum()
    previous = data.loc[
        (data["date"].dt.year == year - 1) & (data["date"].dt.month == month),
        "volume",
    ].sum()
    if previous == 0:
        return np.nan
    return (current-previous)/previous*100


# ============================================================
# ACCUEIL
# ============================================================

if st.session_state.page == "Accueil":

    MOIS_OPTIONS = ["Toute l’année"] + [MOIS[m] for m in range(1, 13)]
    years = sorted(df["date"].dt.year.unique().tolist())

    st.markdown(
        '<div class="panel"><div class="panel-title">Filtre temporel</div>',
        unsafe_allow_html=True,
    )
    fc1, fc2 = st.columns(2)
    with fc1:
        selected_year = st.selectbox(
            "Année d’analyse",
            years,
            index=len(years) - 1,
        )
    with fc2:
        selected_month_label = st.selectbox(
            "Période",
            MOIS_OPTIONS,
            index=0,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_month_label == "Toute l’année":
        periode_df = df[df["date"].dt.year == selected_year]
        volume_periode = float(periode_df["volume"].sum())
        ca_periode = float(periode_df["ca"].sum())
        growth = annual_growth(df, int(selected_year))
        periode_label = str(selected_year)
        label_volume = "Volume annuel"
        label_croissance = "Croissance annuelle"
        note_croissance = "Par rapport à l’année précédente"
    else:
        selected_month_num = next(
            m for m, nom in MOIS.items() if nom == selected_month_label
        )
        periode_df = df[
            (df["date"].dt.year == selected_year)
            & (df["date"].dt.month == selected_month_num)
        ]
        volume_periode = float(periode_df["volume"].sum())
        ca_periode = float(periode_df["ca"].sum())
        growth = monthly_growth(df, int(selected_year), selected_month_num)
        periode_label = f"{selected_month_label} {selected_year}"
        label_volume = "Volume mensuel"
        label_croissance = "Croissance mensuelle"
        note_croissance = "Par rapport au même mois de l’année précédente"

    st.markdown(
        f"""
        <div class="kpi-grid">
        <div class="kpi">
        <div class="kpi-label">{label_volume}</div>
        <div class="kpi-value">{number(volume_periode)}</div>
        <div class="kpi-note">Données réelles • {periode_label}</div>
        </div>
        <div class="kpi">
        <div class="kpi-label">Chiffre d’affaires</div>
        <div class="kpi-value">{money(ca_periode)}</div>
        <div class="kpi-note">Tarif : {money(st.session_state.tarif)}</div>
        </div>
        <div class="kpi">
        <div class="kpi-label">{label_croissance}</div>
        <div class="kpi-value">
        {"N/D" if pd.isna(growth) else f"{growth:+.1f} %"}
        </div>
        <div class="kpi-note">{note_croissance}</div>
        </div>
        <div class="kpi">
        <div class="kpi-label">Horizon ferme</div>
        <div class="kpi-value">{horizon_fiable} mois</div>
        <div class="kpi-note">Sur {len(previsions)} mois prévus</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15,1])

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Chronique de l’activité</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(history_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">État du dispositif</div>',
            unsafe_allow_html=True,
        )
        for label, value in [
            ("Base active", results["source"]),
            ("Période historique",
             f"{df['date'].min():%b %Y} → {df['date'].max():%b %Y}"),
            ("Observations historiques", len(df)),
            ("Observations ML", len(df_ml)),
            ("Modèle opérationnel", "XGBoost"),
        ]:
            st.markdown(
                f'<div class="metric-line"><div class="metric-small">{label}</div>'
                f'<div class="metric-large">{value}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="panel"><div class="panel-title">Données historiques</div>',
        unsafe_allow_html=True,
    )
    shown = df.copy()
    shown["date"] = shown["date"].dt.strftime("%B %Y")
    st.dataframe(shown, use_container_width=True, hide_index=True, height=310)
    st.download_button(
        "Télécharger l’historique analysé",
        df.to_csv(index=False).encode("utf-8"),
        "AVI_CENTER_historique_analyse.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYSE DESCRIPTIVE
# ============================================================

elif st.session_state.page == "Analyse descriptive":

    st.markdown(
        '<div class="panel"><div class="panel-title">Évolution du volume de ventes</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(history_chart(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Statistiques descriptives</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            stats(df).style.format({"Valeur":"{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">Saisonnalité observée</div>',
            unsafe_allow_html=True,
        )
        temp = df.copy()
        temp["mois_num"] = temp["date"].dt.month
        temp["mois"] = temp["mois_num"].map(MOIS)
        monthly = (
            temp.groupby(["mois_num","mois"])["volume"]
            .mean().reset_index().sort_values("mois_num")
        )
        fig = go.Figure(go.Bar(
            x=monthly["mois"], y=monthly["volume"],
            name="Volume moyen",
        ))
        fig.update_layout(
            height=360, margin=dict(l=8,r=8,t=20,b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.76)",
            xaxis=dict(showgrid=False,tickangle=-35),
            yaxis=dict(title="Volume moyen",
                       gridcolor="rgba(15,45,75,.08)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="panel">
        <div class="panel-title">Dispositif de prévision</div>
        <div class="metric-line"><div class="metric-small">Modèle opérationnel</div>
        <div class="metric-large">XGBoost</div></div>
        <div class="metric-line"><div class="metric-small">Variables explicatives</div>
        <div class="metric-large">mois • trimestre • saison • lag_1 • lag_2 • moyenne_mobile_3</div></div>
        <div class="metric-line"><div class="metric-small">Horizon ferme</div>
        <div class="metric-large">{horizon_fiable} mois</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PREVISION
# ============================================================

elif st.session_state.page == "Prévision":

    ferme = previsions[previsions["statut"] == "Ferme"]

    volume_ferme = float(ferme["volume_pred"].sum())
    ca_ferme = float(ferme["ca_prevu"].sum())

    st.markdown(
        f"""
        <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Horizon ferme</div>
        <div class="kpi-value">{horizon_fiable} mois</div>
        <div class="kpi-note">Horizon retenu dans l’étude</div></div>
        <div class="kpi"><div class="kpi-label">Volume ferme</div>
        <div class="kpi-value">{number(volume_ferme)}</div>
        <div class="kpi-note">Dossiers prévus</div></div>
        <div class="kpi"><div class="kpi-label">CA ferme</div>
        <div class="kpi-value">{money(ca_ferme)}</div>
        <div class="kpi-note">Sur les {horizon_fiable} premiers mois</div></div>
        <div class="kpi"><div class="kpi-label">CA total indicatif</div>
        <div class="kpi-value">{money(resume["ca_total"])}</div>
        <div class="kpi-note">12 mois</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel"><div class="panel-title">Prévision dynamique — XGBoost</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        forecast_chart(df[["date","volume"]], previsions),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="notice">
        <b>Lecture managériale :</b> les {horizon_fiable} premiers mois
        constituent l’horizon ferme retenu dans l’étude. Les mois 5 à 12
        sont des prévisions indicatives destinées à l’anticipation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    table = previsions.copy()
    table["Période"] = table["date"].apply(
        lambda x: f"{MOIS[x.month]} {x.year}"
    )
    table["Volume prévu"] = table["volume_pred"].round(2)
    table["CA prévu"] = table["ca_prevu"].round(0)
    table["CA prévu (M FCFA)"] = table["ca_prevu_millions"].round(2)
    table = table[
        ["Période","horizon","Volume prévu","statut",
         "CA prévu","CA prévu (M FCFA)"]
    ].rename(columns={"horizon":"Horizon","statut":"Statut"})

    st.markdown(
        '<div class="panel"><div class="panel-title">Tableau de prévision</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Télécharger les prévisions",
        previsions.to_csv(index=False).encode("utf-8"),
        "AVI_CENTER_previsions_xgboost.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RECOMMANDATIONS
# ============================================================

elif st.session_state.page == "Recommandations":

    st.markdown(
        f"""
        <div class="panel">
        <div class="panel-title">Repères managériaux</div>
        <div class="metric-line"><div class="metric-small">Modèle</div>
        <div class="metric-large">XGBoost</div></div>
        <div class="metric-line"><div class="metric-small">Horizon ferme</div>
        <div class="metric-large">{horizon_fiable} mois</div></div>
        <div class="metric-line"><div class="metric-small">Tarif unitaire</div>
        <div class="metric-large">{money(st.session_state.tarif)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mean_hist = float(df["volume"].mean())
    max_hist = float(df["volume"].max())

    phases = {
        1:"Préparation du cycle", 2:"Préparation des dossiers",
        3:"Dépôts et accompagnement", 4:"Dépôts et accompagnement",
        5:"Traitement administratif", 6:"Période de tension potentielle",
        7:"Forte demande potentielle", 8:"Forte demande potentielle",
        9:"Rentrée universitaire", 10:"Rentrée / régularisation",
        11:"Transition", 12:"Préparation du cycle suivant",
    }

    rows = []

    for _, r in previsions.iterrows():
        value = float(r["volume_pred"])
        month = int(r["date"].month)
        ratio = value / mean_hist if mean_hist else 1

        if value >= max_hist*.85 or ratio >= 1.20:
            level = "Forte activité"
            action = "Renforcer la capacité de traitement et anticiper les demandes."
        elif value <= mean_hist*.80:
            level = "Activité modérée"
            action = "Renforcer la prospection, la communication et le suivi commercial."
        else:
            level = "Activité intermédiaire"
            action = "Maintenir le dispositif et suivre l’évolution de la demande."

        rows.append({
            "Période": f"{MOIS[month]} {r['date'].year}",
            "Prévision": round(value),
            "Statut": r["statut"],
            "Repère": phases[month],
            "Niveau": level,
            "Action": action,
        })

    recommendations = pd.DataFrame(rows)
    recommendations["ordre"] = np.where(
        recommendations["Statut"] == "Ferme", 0, 1
    )
    recommendations = recommendations.sort_values(
        ["ordre","Prévision"], ascending=[True,False]
    ).drop(columns="ordre")

    for _, r in recommendations.iterrows():
        border = "#b06b30" if r["Statut"] == "Ferme" else "#687888"
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,.94);border-left:5px solid {border};
            border-radius:8px;padding:12px 15px;margin-bottom:9px;
            box-shadow:0 7px 16px rgba(18,45,75,.06);">
            <div style="display:flex;justify-content:space-between;gap:14px;">
            <div>
            <div style="font-size:.71rem;letter-spacing:.08em;text-transform:uppercase;
            color:#738392;">{r["Période"]} • {r["Repère"]}</div>
            <div style="font-size:1.02rem;font-weight:730;color:#123b62;">{r["Niveau"]}</div>
            </div>
            <div style="font-size:1.12rem;font-weight:760;color:#123b62;">
            {r["Prévision"]:,.0f}
            </div></div>
            <div style="margin-top:6px;color:#566a7b;font-size:.87rem;">{r["Action"]}</div>
            <div style="margin-top:6px;color:#84919d;font-size:.7rem;text-transform:uppercase;">
            {r["Statut"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.download_button(
        "Télécharger le plan de recommandations",
        recommendations.drop(columns="ordre", errors="ignore")
        .to_csv(index=False).encode("utf-8"),
        "AVI_CENTER_recommandations.csv",
        "text/csv",
    )
