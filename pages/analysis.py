"""
Méthodologie Technique — Documentation Complète
Galatasaray Üniversitesi | Rail Cost Estimator
"""

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings

from src import AppConfig

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_config():
    return AppConfig.from_env()

config   = get_config()
ui_cfg   = config.ui_config
file_cfg = config.file_config

# Theme-aligned color palette (#009999 primary, F4F5F7 bg, FFFFFF card)
colors = {
    "primary":    "#009999",
    "accent_bg":  "#004D4D",   # koyu teal — badge arka planı
    "accent_text":"#B2DFDF",   # açık teal — badge yazısı
    "card":       "#FFFFFF",
    "border":     "#CBD5E0",
    "text_main":  "#2D3748",
    "text_muted": "#4A5568",
    "success":    "#276749",
    "error":      "#C53030",
    "warn":       "#007777",   # sarı yerine orta teal
}
# Siemens-teal palette — en açık ton bile görünür
TEAL = ["#002929", "#004D4D", "#007777", "#009999", "#00BBBB"]
MODE_COLORS = {"HRT": "#004D4D", "LRT": "#009999", "BRT": "#5B2D8E", "CRT": "#2D6A9F"}

# ── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Méthodologie Technique — GSÜ",
    page_icon="🔬",
    layout="wide",
)
st.markdown(ui_cfg.custom_css, unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COUNTRY_NAMES_FR = {
    "AE":"Émirats","AR":"Argentine","AT":"Autriche","AU":"Australie",
    "BD":"Bangladesh","BE":"Belgique","BR":"Brésil","CA":"Canada",
    "CH":"Suisse","CL":"Chili","CN":"Chine","CO":"Colombie",
    "CZ":"Rép. Tchèque","DE":"Allemagne","DK":"Danemark","DR":"Rép. Dominicaine",
    "EC":"Équateur","EG":"Égypte","ES":"Espagne","FI":"Finlande",
    "FR":"France","GR":"Grèce","HK":"Hong Kong","HU":"Hongrie",
    "ID":"Indonésie","IL":"Israël","IN":"Inde","IR":"Iran",
    "IT":"Italie","JP":"Japon","KR":"Corée du Sud","KW":"Koweït",
    "MX":"Mexique","MY":"Malaisie","NL":"Pays-Bas","NO":"Norvège",
    "NZ":"Nouvelle-Zélande","PA":"Panama","PE":"Pérou","PH":"Philippines",
    "PK":"Pakistan","PL":"Pologne","PT":"Portugal","QA":"Qatar",
    "RO":"Roumanie","RS":"Serbie","RU":"Russie","SA":"Arabie Saoudite",
    "SE":"Suède","SG":"Singapour","TH":"Thaïlande","TR":"Turquie",
    "TW":"Taïwan","UA":"Ukraine","UK":"Royaume-Uni","US":"États-Unis",
    "UZ":"Ouzbékistan","VN":"Viêt Nam",
}

SUBSYS_COLS = [
    "guideway_costs_pct","station_costs_pct","systems_costs_pct",
    "soft_costs_pct","vehicle_costs_pct","row_costs_pct",
    "sitework_costs_pct","facilities_costs_pct",
]
SUBSYS_LABELS = [
    "Guideway","Stations","Systems","Soft Costs",
    "Vehicles","ROW","Sitework","Facilities",
]

FEATURE_NAMES_FR = {
    "country_te":"Prime Nationale (Bayésienne)",
    "country_freq":"Volume du Marché National",
    "city_te":"Prime Locale (Bayésienne)",
    "city_freq":"Volume du Marché Local",
    "tunnel_pct":"Proportion de Tunnel (%)",
    "station_density":"Densité de Stations",
    "log_length":"Longueur de Ligne (log)",
    "is_regional_rail":"Type : Rail Régional",
    "mid_year":"Fenêtre d'Inflation (Année)",
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_assets():
    try:
        model    = joblib.load("rail_cost_model.pkl")
        memory   = joblib.load("memory_package.pkl")
        features = joblib.load("feature_names.pkl")
        return model, memory, features, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_processed_rail():
    try:
        return pd.read_csv(str(file_cfg.global_rail_csv))
    except FileNotFoundError:
        return None

@st.cache_data
def load_processed_fta():
    try:
        df = pd.read_csv(str(file_cfg.fta_processed_csv))
        df["cost_per_km_musd"] = df["cost_per_km_2023"] / 1e6
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_raw_global():
    try:
        return pd.read_excel(
            "data/raw/global_rail_costs.xlsx",
            sheet_name="1_16_2026", header=0,
        )
    except Exception:
        return None

@st.cache_data
def load_raw_fta():
    try:
        return pd.read_excel(
            "data/raw/fta_summary.xlsx",
            sheet_name="Sheet1", header=0,
        )
    except Exception:
        return None

model, memory, features, load_err = load_assets()
gr_proc  = load_processed_rail()
fta_proc = load_processed_fta()
gr_raw   = load_raw_global()
fta_raw  = load_raw_fta()

if load_err or model is None:
    st.error(f"**Erreur de chargement :** {load_err}")
    st.stop()

report       = memory.get("report", {})
global_cost  = np.exp(memory["global_mean"])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Méthodologie & Analyse des Données")
st.markdown(
    "Documentation exhaustive du pipeline : sources brutes → preprocessing → "
    "encodage → entraînement → validation. Couvre les deux jeux de données "
    "(**Global Rail** et **FTA**) et le moteur de prédiction."
)

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 1 — VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 1 — Vue d'Ensemble du Système")

k1, k2, k3, k4, k5, k6 = st.columns(6)
total_projets = sum(memory["country_freq_map"].values())
raw_gr_n = len(gr_raw) if gr_raw is not None else "—"
proc_gr_n = len(gr_proc) if gr_proc is not None else "—"

def mini_kpi(col, label, value, sub=""):
    col.markdown(
        f'<div style="background:{colors["card"]};border:1px solid {colors["border"]};'
        f'border-radius:10px;padding:14px;text-align:center;">'
        f'<div style="font-size:10px;font-weight:600;color:{colors["text_muted"]};'
        f'text-transform:uppercase;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{colors["text_main"]};">{value}</div>'
        f'<div style="font-size:10px;color:{colors["text_muted"]};">{sub}</div></div>',
        unsafe_allow_html=True,
    )

mini_kpi(k1, "Global Rail (brut)",   f"{raw_gr_n:,}",         "projets, 60 pays")
mini_kpi(k2, "Global Rail (propre)", f"{proc_gr_n:,}",        "après nettoyage")
mini_kpi(k3, "FTA (brut & propre)",  "49",                    "projets US")
mini_kpi(k4, "Devises",              "44",                    "converties en USD")
mini_kpi(k5, "Prior Global",         f"${global_cost:,.0f}M", "médiane M$/km")
mini_kpi(k6, "Précision OOF",        f"{report.get('mdape',0):.1f}%", "MdAPE cross-val")

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 2 — DONNÉES BRUTES : GLOBAL RAIL
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 2 — Données Brutes : Global Rail")

if gr_raw is not None:

    # 2.1 Complétude
    st.markdown("### 2.1 Complétude des Colonnes Clés")
    key_cols_gr = {
        "Start year": "Année début",
        "End year": "Année fin",
        "TunnelPer": "Tunnel (%)",
        "Stations": "Stations",
        "Elevated": "Surélevé",
        "Atgrade": "Au sol",
        "Platform Length \n(Meters)": "Long. quai",
        "Source1": "Source",
        "Cost/km (2023 dollars)": "Coût/km 2023",
    }
    completeness = []
    for col, label in key_cols_gr.items():
        if col in gr_raw.columns:
            null_pct = gr_raw[col].isnull().sum() / len(gr_raw) * 100
            completeness.append({
                "Colonne": label,
                "Complète (%)": round(100 - null_pct, 1),
                "Manquante (%)": round(null_pct, 1),
                "n manquant": int(gr_raw[col].isnull().sum()),
            })
    comp_df = pd.DataFrame(completeness).sort_values("Manquante (%)", ascending=False)

    fig_comp = px.bar(
        comp_df, x="Colonne", y=["Complète (%)", "Manquante (%)"],
        barmode="stack",
        color_discrete_map={"Complète (%)": "#006666", "Manquante (%)": "#C53030"},
        labels={"value": "%", "variable": "Statut"},
    )
    fig_comp.update_layout(
        height=360, margin=dict(t=10,b=110,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.30, x=0, title_text=""),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(range=[0,100], showgrid=True, gridcolor=colors["border"]),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 3 — DONNÉES BRUTES : FTA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 3 — Données Brutes : FTA (États-Unis)")

if fta_raw is not None:

    # 3.1 Complétude FTA
    st.markdown("### 3.1 Complétude des Colonnes (49 projets)")
    fta_null = (fta_raw.isnull().sum() / len(fta_raw) * 100).round(1)
    fta_null = fta_null[fta_null > 0].sort_values(ascending=False).reset_index()
    fta_null.columns = ["Colonne", "Manquante (%)"]
    fta_null["Complète (%)"] = 100 - fta_null["Manquante (%)"]

    fig_fta_comp = px.bar(
        fta_null, x="Colonne", y=["Complète (%)","Manquante (%)"],
        barmode="stack",
        color_discrete_map={"Complète (%)": "#006666", "Manquante (%)": "#C53030"},
        labels={"value": "%", "variable": "Statut"},
    )
    fig_fta_comp.update_layout(
        height=380, margin=dict(t=10,b=120,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.35, x=0, title_text=""),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(range=[0,100], showgrid=True, gridcolor=colors["border"]),
    )
    st.plotly_chart(fig_fta_comp, use_container_width=True)

    # 3.2 Format brut dollar
    st.markdown("### 3.2 Nettoyage : Format Monétaire Brut → Float")
    col_ex, col_info = st.columns([2, 3])
    with col_ex:
        sample_raw = fta_raw[["Mode","Project Cost","Guideway Costs %"]].head(5).copy()
        sample_raw.columns = ["Mode","Coût Projet (brut)","Guideway % (brut)"]
        st.dataframe(sample_raw, hide_index=True, use_container_width=True)
    with col_info:
        st.markdown("""
**Transformations appliquées :**
- `"$4,039,036,247"` → suppression `$` et `,` → `float`
- Colonnes `%` vérifiées dans l'intervalle `[0, 1]`
- Longueurs en miles → km (× 1.60934)
- Coûts /mile → coûts /km (÷ 1.60934)
- Ajustement inflation 2021 → 2023 via multiplicateur CPI
        """)

    # 3.3 Mode distribution FTA
    st.markdown("### 3.3 Distribution par Mode de Transport")
    mode_dist = fta_raw["Mode"].value_counts().reset_index()
    mode_dist.columns = ["Mode","n"]
    fig_mode = px.bar(
        mode_dist, x="Mode", y="n", text="n",
        color="Mode",
        color_discrete_map={"HRT":"#004D4D","LRT":"#009999","BRT":"#5B2D8E","CRT":"#2D6A9F","Trolley":"#4A5568"},
    )
    fig_mode.update_traces(textposition="outside")
    fig_mode.update_layout(
        height=300, margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor=colors["border"], range=[0, mode_dist["n"].max()*1.25]),
    )
    st.plotly_chart(fig_mode, use_container_width=True)
    st.caption("BRT (n=4) et CRT (n=2) : faible effectif → ratios calculés avec prudence, aucun fallback vers la médiane globale.")

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 4 — PREPROCESSING & FILTRES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 4 — Preprocessing & Filtres")

if gr_raw is not None and gr_proc is not None:
    st.markdown("### 4.1 Impact du Pipeline de Nettoyage (Global Rail)")
    n_raw  = len(gr_raw)
    n_proc = len(gr_proc)
    dropped = n_raw - n_proc
    _steps_y = ["Données brutes", "country+city présents", "Coût/km présent", "Tunnel présent (final)"]
    _steps_x = [n_raw, n_raw - 3, n_raw - 8, n_proc]
    _steps_colors = ["#002929", "#004D4D", "#007777", "#009999"]
    fig_funnel = go.Figure(go.Funnel(
        y=_steps_y, x=_steps_x,
        textinfo="value+percent initial",
        textfont=dict(size=13, color="white"),
        marker=dict(color=_steps_colors),
        connector=dict(line=dict(color="#CBD5E0", width=1)),
    ))
    fig_funnel.update_layout(
        height=320, margin=dict(t=10,b=10,l=180,r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text_main"]),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.caption(f"{dropped} projets exclus au total ({dropped/n_raw*100:.1f}%) — principalement données de tunnel manquantes.")

    st.markdown("### 4.2 Feature Engineering")
    fe_table = pd.DataFrame([
        {"Feature créée": "tunnel_pct",      "Source": "TunnelPer",              "Transformation": "Normalisation 0-1 (si > 1 → ÷100)"},
        {"Feature créée": "station_density", "Source": "num_stations, length_km", "Transformation": "stations / (length_km + 0.1)"},
        {"Feature créée": "log_length",      "Source": "length_km",              "Transformation": "log(length_km)"},
        {"Feature créée": "mid_year",        "Source": "start_year, end_year",   "Transformation": "(start + end) / 2"},
        {"Feature créée": "country_te",      "Source": "country",                "Transformation": "Encodage bayésien (m=10)"},
        {"Feature créée": "city_te",         "Source": "city",                   "Transformation": "Encodage bayésien (m=5)"},
        {"Feature créée": "log_cost",        "Source": "cost_per_km_2023_musd",  "Transformation": "log(cost) — variable cible"},
    ])
    st.dataframe(fe_table, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 5 — ANALYSE GLOBAL RAIL (DONNÉES PROPRES)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 5 — Analyse : Global Rail (948 projets)")

if gr_proc is not None:

    # 5.1 Distribution du coût
    st.markdown("### 5.1 Distribution du Coût/km (M$, 2023)")
    fig_dist = make_subplots(rows=1, cols=2,
                             subplot_titles=["Échelle normale", "Échelle log"])
    fig_dist.add_trace(
        go.Histogram(x=gr_proc["cost_per_km_2023_musd"], nbinsx=50,
                     marker_color=colors["primary"], opacity=0.7, name=""),
        row=1, col=1,
    )
    fig_dist.add_trace(
        go.Histogram(x=np.log(gr_proc["cost_per_km_2023_musd"]), nbinsx=40,
                     marker_color="#004D4D", opacity=0.7, name=""),
        row=1, col=2,
    )
    fig_dist.update_layout(
        height=300, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30,b=20,l=10,r=10),
    )
    fig_dist.update_xaxes(showgrid=True, gridcolor=colors["border"])
    fig_dist.update_yaxes(showgrid=True, gridcolor=colors["border"])
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Distribution log-normale → justifie la log-transformation de la variable cible pour la régression.")

    # 5.2 Coût par pays (n>=5)
    st.markdown("### 5.2 Coût Médian par Pays (n ≥ 5 projets)")
    country_stats = (
        gr_proc.groupby("country")["cost_per_km_2023_musd"]
        .agg(["median","count","std"])
        .reset_index()
        .query("count >= 5")
        .sort_values("median", ascending=True)
    )
    country_stats["Pays"] = country_stats["country"].map(COUNTRY_NAMES_FR).fillna(country_stats["country"])
    country_stats["CV"] = (country_stats["std"] / country_stats["median"]).round(2)

    fig_country = px.bar(
        country_stats, x="median", y="Pays", orientation="h",
        text="count",
        color="median",
        color_continuous_scale=[[0,"#B2DFDF"],[0.3,"#009999"],[0.65,"#004D4D"],[1.0,"#001A1A"]],
        labels={"median":"Coût médian (M$/km)","Pays":""},
    )
    fig_country.update_traces(texttemplate="n=%{text}", textposition="outside", cliponaxis=False)
    fig_country.update_layout(
        height=max(500, len(country_stats)*22),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10,b=10,l=10,r=80),
        coloraxis_showscale=False,
    )
    fig_country.update_xaxes(showgrid=True, gridcolor=colors["border"])
    st.plotly_chart(fig_country, use_container_width=True)

    # 5.3 Variabilité intra-pays
    st.markdown("### 5.3 Variabilité Intra-pays (Coefficient de Variation)")
    cv_df = country_stats.query("count >= 5").sort_values("CV", ascending=False).head(15)
    fig_cv = px.bar(
        cv_df, x="Pays", y="CV", text="CV",
        color="CV",
        color_continuous_scale=[[0,"#009999"],[0.5,"#004D4D"],[1.0,"#002929"]],
        labels={"CV":"CV = σ/médiane"},
    )
    fig_cv.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_cv.update_layout(
        height=340, margin=dict(t=50,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis=dict(showgrid=True, gridcolor=colors["border"], range=[0, cv_df["CV"].max()*1.25]),
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    st.caption(
        "Royaume-Uni (CV=3.05) et États-Unis (CV=1.85) : "
        "marchés très hétérogènes — tunnel urbain vs ligne périurbaine au sol. "
        "Chine (non affiché, n=442) : CV=0.26, marché très standardisé."
    )

    # 5.4 Évolution temporelle
    st.markdown("### 5.4 Évolution du Coût par Décennie (dollars constants 2023)")
    gr2 = gr_proc.dropna(subset=["end_year"]).copy()
    gr2["Décennie"] = (gr2["end_year"] // 10 * 10).astype(int).astype(str) + "s"
    decade_df = (
        gr2.groupby("Décennie")["cost_per_km_2023_musd"]
        .agg(["median","count",
              lambda x: x.quantile(0.25),
              lambda x: x.quantile(0.75)])
        .reset_index()
    )
    decade_df.columns = ["Décennie","Médiane","n","Q25","Q75"]
    decade_df = decade_df[decade_df["n"] >= 5]

    fig_decade = px.bar(
        decade_df, x="Décennie", y="Médiane", text="n",
        color="Médiane",
        color_continuous_scale=[[0,"#007777"],[0.5,"#004D4D"],[1.0,"#002929"]],
        error_y=decade_df["Q75"] - decade_df["Médiane"],
        error_y_minus=decade_df["Médiane"] - decade_df["Q25"],
    )
    fig_decade.update_traces(texttemplate="n=%{text}", textposition="outside",
                              cliponaxis=False, error_y_color=colors["text_muted"])
    fig_decade.update_layout(
        height=360, margin=dict(t=40,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        yaxis=dict(showgrid=True, gridcolor=colors["border"]),
    )
    st.plotly_chart(fig_decade, use_container_width=True)
    st.caption("Barres d'erreur : Q25–Q75. Les années 2020 (n=400) : médiane ~2× les décennies précédentes.")

    # 5.5 Scatter tunnel vs coût
    st.markdown("### 5.5 Part de Tunnel → Coût/km (Global Rail, r = 0.01)")
    _gr_scatter = gr_proc.sample(min(600, len(gr_proc)), random_state=42).copy()
    _gr_scatter["tunnel_pct_pct"] = _gr_scatter["tunnel_pct"]
    fig_gr_sc = px.scatter(
        _gr_scatter,
        x="tunnel_pct_pct", y="cost_per_km_2023_musd",
        trendline="ols", opacity=0.35,
        color_discrete_sequence=[colors["primary"]],
        labels={
            "tunnel_pct_pct": "Part de tunnel (%)",
            "cost_per_km_2023_musd": "Coût/km (M$, 2023)",
        },
    )
    fig_gr_sc.update_traces(marker=dict(size=5))
    fig_gr_sc.update_layout(
        height=360, margin=dict(t=20,b=40,l=60,r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Part de tunnel (%)",
        yaxis_title="Coût/km (M$, 2023)",
    )
    fig_gr_sc.update_yaxes(range=[0, 1200], showgrid=True, gridcolor=colors["border"])
    fig_gr_sc.update_xaxes(showgrid=True, gridcolor=colors["border"])
    st.plotly_chart(fig_gr_sc, use_container_width=True)
    st.caption(
        "r = 0.01 — corrélation quasi-nulle à l'échelle mondiale. "
        "La localisation (pays/ville) domine : deux tunnels identiques au Royaume-Uni et en Chine "
        "peuvent coûter 5× différemment."
    )

    # --- SIEMENS GÖRSEL KİMLİK TANIMLARI (Güncellenmiş) ---
    # Beyazlar ve açık griler daha belirgin hale getirildi.
    SIEMENS_PETROL = "#00646e"
    SIEMENS_DARK_GREY_TEXT = "#333333" # Daha koyu metin
    SIEMENS_MEDIUM_GREY = "#d6dbdf"     # Okunabilirlik için daha koyu gri (ilk sürümden bir tık koyu)

    # Siemens Korelasyon Skalası (Güncellenmiş):
    # Beyaz zemin kaldırıldı, nötr korelasyon açık griye çekildi.
    siemens_colorscale = [
        [0.0, "#666666"],             # Tam Negatif Korelasyon
        [0.5, SIEMENS_MEDIUM_GREY],  # Korelasyon Yok (Açık Beyaz yerine Orta Gri)
        [1.0, "#00646e"]              # Tam Pozitif Korelasyon
    ]

    # --- 5.6 Matrice de Corrélation (Données Brutes) ---
    st.markdown("### 5.6 Matrice de Corrélation (Exploration des Données Brutes)")
    st.markdown("Cette matrice présente les relations linéaires directes entre les variables physiques/économiques brutes et le coût unitaire.")

    raw_corr_cols = {
        "cost_per_km_2023_musd": "Coût/km (Réel)",
        "tunnel_pct": "Tunnel (%)",
        "length_km": "Longueur (km)",
        "num_stations": "Nb Stations",
        "ppp_rate": "Taux PPP",
        "start_year": "Année Début",
    }

    # gr_proc verisi yüklenmiş varsayılarak:
    avail_raw = {k:v for k,v in raw_corr_cols.items() if k in gr_proc.columns}
    df_raw_corr = gr_proc[list(avail_raw.keys())].copy()
    df_raw_corr["Log(Coût)"] = np.log(df_raw_corr["cost_per_km_2023_musd"]) 
    corr_m_raw = df_raw_corr.corr()
    display_names = list(avail_raw.values()) + ["Coût/km (Log)"]
    corr_m_raw.index = display_names
    corr_m_raw.columns = display_names

    fig_corr_raw = px.imshow(
        corr_m_raw,
        color_continuous_scale=siemens_colorscale,
        zmin=-1, zmax=1, 
        text_auto=".2f", 
        aspect="auto"
    )

    fig_corr_raw.update_layout(
        height=450, 
        margin=dict(t=10,b=10,l=10,r=10), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12, color=SIEMENS_DARK_GREY_TEXT), # Metin daha koyu
        coloraxis_showscale=True
    )

    st.plotly_chart(fig_corr_raw, use_container_width=True)
    st.caption("Interprétation : Notez la faible corrélation entre le tunnel (%) et le coût brut à l'échelle mondiale, illustrant la nécessité de l'encodage bayésien par localisation.")


    # --- 5.6 Matrice de Corrélation du modèle ---
    st.markdown("---")
    st.markdown("### 5.6 Matrice de Corrélation des Variables du Modèle")

    df_corr_input = gr_proc.copy()

    # memory verisi yüklenmiş varsayılarak:
    if 'city' in df_corr_input.columns and 'city_te_map' in memory:
        df_corr_input['city_te'] = df_corr_input['city'].map(memory['city_te_map']).fillna(memory['global_mean'])

    if 'country' in df_corr_input.columns and 'country_te_map' in memory:
        df_corr_input['country_te'] = df_corr_input['country'].map(memory['country_te_map']).fillna(memory['global_mean'])
        
    target_col = "cost_per_km_2023_musd"
    model_features = list(FEATURE_NAMES_FR.keys()) # FEATURE_NAMES_FR tanımlı varsayılarak
    cols_to_use = [c for c in model_features if c in df_corr_input.columns] + [target_col]
    df_final_corr = df_corr_input[cols_to_use].apply(pd.to_numeric, errors='coerce')
    corr_matrix = df_final_corr.corr()
    display_labels = [FEATURE_NAMES_FR.get(c, c) for c in corr_matrix.columns]
    display_labels = [label if label != target_col else "Coût/km (Réel M$)" for label in display_labels]

    fig_corr_final = px.imshow(
        corr_matrix,
        x=display_labels,
        y=display_labels,
        color_continuous_scale=siemens_colorscale,
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto"
    )

    fig_corr_final.update_layout(
        height=700,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=11, color=SIEMENS_DARK_GREY_TEXT), # Metin daha koyu
        coloraxis_colorbar=dict(
            title="Corrélation",
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.6,
            yanchor="middle", y=0.5,
            tickfont=dict(color=SIEMENS_PETROL)
        )
    )

    st.plotly_chart(fig_corr_final, use_container_width=True)
    st.caption("Cette matrice inclut désormais les primes bayésiennes (City/Country), le type de rail (Regional) ve l'inflation (Mid Year).")
# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 6 — ANALYSE FTA
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 6 — Analyse : FTA (49 projets américains)")

if fta_proc is not None:

    # 6.1 Coût/km par mode — box
    st.markdown("### 6.1 Distribution du Coût/km par Mode (M$, 2023)")
    fig_box = px.box(
        fta_proc[fta_proc["mode"].isin(["HRT","LRT","BRT","CRT"])],
        x="mode", y="cost_per_km_musd",
        color="mode", color_discrete_map=MODE_COLORS,
        points="all",
        hover_data=["project","year"],
        category_orders={"mode":["HRT","LRT","BRT","CRT"]},
        labels={"mode":"Mode","cost_per_km_musd":"Coût/km (M$)"},
    )
    fig_box.update_traces(marker=dict(size=7, opacity=0.8), jitter=0.3)
    fig_box.update_layout(
        height=380, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10,b=10,l=10,r=10),
        yaxis=dict(showgrid=True, gridcolor=colors["border"]),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    stats_rows = []
    for mode in ["HRT","LRT","BRT","CRT"]:
        g = fta_proc[fta_proc["mode"]==mode]["cost_per_km_musd"]
        tp = fta_proc[fta_proc["mode"]==mode]["tunnel_pct"]
        stats_rows.append({
            "Mode": mode, "n": len(g),
            "Médiane (M$/km)": f"${g.median():.0f}",
            "Min → Max": f"${g.min():.0f} → ${g.max():.0f}",
            "Tunnel médian": f"{tp.median()*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)


    # 6.2 Subsystem heatmap
    st.markdown("### 6.2 Décomposition des Coûts : Heatmap (Mode × Sous-système)")
    heat_data = {}
    for mode in ["HRT","LRT","BRT","CRT"]:
        g = fta_proc[fta_proc["mode"]==mode]
        meds = g[SUBSYS_COLS].median()
        total = meds.sum()
        heat_data[mode] = (meds/total*100).values
    heat_df = pd.DataFrame(heat_data, index=SUBSYS_LABELS).T

    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=[[0,"#007777"],[0.5,"#004D4D"],[1.0,"#002929"]],
        text_auto=".1f", aspect="auto",
        labels={"color":"% du total"},
    )
    fig_heat.update_layout(
        height=260, margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="%"),
    )
    fig_heat.update_traces(textfont=dict(size=12))
    st.plotly_chart(fig_heat, use_container_width=True)

    # 6.3 Grouped bar subsystems
    st.markdown("### 6.3 Ratios Médians par Sous-système et par Mode")
    rows = []
    for mode in ["HRT","LRT","BRT","CRT"]:
        g = fta_proc[fta_proc["mode"]==mode]
        meds = g[SUBSYS_COLS].median()
        total = meds.sum()
        normed = meds/total
        for col, label in zip(SUBSYS_COLS, SUBSYS_LABELS):
            rows.append({"Mode":mode,"Sous-système":label,"Ratio (%)":normed[col]*100})
    ratio_df = pd.DataFrame(rows)
    fig_grouped = px.bar(
        ratio_df, x="Sous-système", y="Ratio (%)", color="Mode",
        barmode="group", color_discrete_map=MODE_COLORS, text_auto=".1f",
    )
    fig_grouped.update_traces(
        textposition="outside", cliponaxis=False,
        textfont=dict(size=9),
    )
    fig_grouped.update_layout(
        height=460, margin=dict(t=20,b=20,l=10,r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05, x=0),
        bargap=0.15, bargroupgap=0.05,
        yaxis=dict(showgrid=True, gridcolor=colors["border"], range=[0, ratio_df["Ratio (%)"].max()*1.35]),
    )
    st.plotly_chart(fig_grouped, use_container_width=True)
    st.caption(
        "HRT : forte part Stations (21%) — stations souterraines coûteuses. "
        "CRT : forte part Sitework (26%) — infrastructure de surface extensive. "
        "BRT : Soft Costs élevés (25%) — planification complexe."
    )

    # 6.4 Complétude FTA processed
    st.markdown("### 6.4 Complétude Post-Preprocessing (FTA Traité)")
    fta_nulls_proc = (fta_proc[SUBSYS_COLS].isnull().sum() / len(fta_proc) * 100).round(1)
    fta_nulls_proc = fta_nulls_proc.reset_index()
    fta_nulls_proc.columns = ["Sous-système","Manquante (%)"]
    fta_nulls_proc["Label"] = fta_nulls_proc["Sous-système"].map(dict(zip(SUBSYS_COLS, SUBSYS_LABELS)))
    fta_nulls_proc["Complète (%)"] = 100 - fta_nulls_proc["Manquante (%)"]

    fig_fta_null = px.bar(
        fta_nulls_proc, x="Label", y=["Complète (%)","Manquante (%)"],
        barmode="stack",
        color_discrete_map={"Complète (%)":"#006666","Manquante (%)":"#C53030"},
    )
    fig_fta_null.update_layout(
        height=360, margin=dict(t=10,b=100,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.30, x=0, title_text=""),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(range=[0,100], showgrid=True, gridcolor=colors["border"]),
    )
    st.plotly_chart(fig_fta_null, use_container_width=True)
    st.caption("Facilities Costs % : 33% manquant — la médiane() de pandas ignore automatiquement les NaN.")

# ─────────────────────────────────────────────────────────────────────────────
# PARTIE 7 — MODÈLE & PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Partie 7 — Modèle & Performance")

# 7.1 Feature importance
st.markdown("### 7.1 Importance des Variables (GBR)")

# Siemens Renk Paleti (Koyu ve Tok Tonlar)
SIEMENS_PETROL = "#00646e"
SIEMENS_DARK = "#003c42"     # En önemli değişken için en koyu ton
SIEMENS_MEDIUM = "#005159"   # Orta önem
SIEMENS_LIGHT = "#a3c2c5"    # Düşük önem (opsiyonel, geçiş için)

fi_dict = dict(zip(features, model.feature_importances_))
fi_df = (
    pd.DataFrame([{"Facteur":FEATURE_NAMES_FR.get(k,k), "Importance":v*100}
                  for k,v in fi_dict.items()])
    .sort_values("Importance", ascending=True)
)

fig_fi = px.bar(
    fi_df, x="Importance", y="Facteur", orientation="h",
    color="Importance",
    # Renk skalasını daha koyu ve vakur Siemens tonlarına çektik
    color_continuous_scale=[
        [0.0, "#d6dbdf"],      # En düşük: Açık Gri (Gözü yormaz)
        [0.3, SIEMENS_PETROL], # Orta: Ana Siemens Petrol
        [1.0, SIEMENS_DARK]    # En yüksek: En Koyu Petrol
    ],
    labels={"Importance":"Influence (%)"},
)

fig_fi.update_layout(
    height=380, 
    margin=dict(t=10, b=10, l=10, r=20), # r=140 çok genişti, 20 yaparak alanı verimli kullandık
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    coloraxis_showscale=False,
    xaxis=dict(
        showgrid=True, 
        gridcolor="#d6dbdf", # Grid çizgilerini hafif belirginleştirdik
        range=[0, fi_df["Importance"].max() * 1.15], # Boşluğu optimize ettik
        tickfont=dict(color="#333333")
    ),
    yaxis=dict(
        tickfont=dict(size=12, color="#333333") # Yazıları daha koyu ve okunaklı yaptık
    )
)

# Barların üzerine değerleri yazdırmak istersen text_auto=True ekleyebilirsin:
fig_fi.update_traces(texttemplate='%{x:.1f}%', textposition='outside')

st.plotly_chart(fig_fi, use_container_width=True)

# 7.2 OOF metrics
st.markdown("### 7.2 Métriques Out-of-Fold (Validation Croisée Stratifiée)")
m1, m2, m3, m4, m5 = st.columns(5)
mini_kpi(m1, "R² (log)",     f"{report.get('r2_log',0):.3f}",   "échelle log")
mini_kpi(m2, "RMSE (log)",   f"{report.get('rmse_log',0):.3f}", "échelle log")
mini_kpi(m3, "MdAPE",        f"{report.get('mdape',0):.1f}%",   "erreur médiane")
mini_kpi(m4, "Biais",        f"{report.get('bias',0):+.1f}%",   "sous/sur-estimation")
mini_kpi(m5, "Success ±30%", f"{report.get('success_30',0):.1f}%","dans la bande")

# 7.3 OOF par pays
if "performance_stats" in memory and memory["performance_stats"]:
    st.markdown("### 7.3 Performance par Pays (n ≥ 3)")
    perf_df = pd.DataFrame(memory["performance_stats"])
    perf_df["Pays"] = perf_df["Country"].map(COUNTRY_NAMES_FR).fillna(perf_df["Country"])
    perf_df["Color"] = perf_df["MdAPE"].apply(
        lambda x: colors["success"] if x < 20 else (colors["warn"] if x < 35 else colors["error"])
    )
    h = max(500, len(perf_df)*26)
    fig_perf = px.bar(
        perf_df.sort_values("MdAPE", ascending=False),
        x="MdAPE", y="Pays", orientation="h",
        text="Cnt", color="Color", color_discrete_map="identity",
    )
    fig_perf.update_traces(texttemplate="n=%{text}", textposition="outside", cliponaxis=False)
    fig_perf.update_layout(
        height=h, margin=dict(l=10,r=80,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis_title="Erreur Médiane (%)",
        xaxis=dict(showgrid=True, gridcolor=colors["border"]),
    )
    fig_perf.update_yaxes(tickmode="linear", dtick=1, tickfont=dict(size=10))
    st.plotly_chart(fig_perf, use_container_width=True)

# 7.4 Bayesian smoothing
st.markdown("### 7.4 Lissage Bayésien : Encodage des Localisations")
col_l, col_r = st.columns([3,2])
with col_l:
    st.markdown("""
**Formule :** `smooth = (n·μ_local + m·μ_global) / (n + m)`

- `n` = nombre de projets dans cette localisation
- `μ_local` = coût moyen local
- `μ_global` = prior global ($214M/km)
- `m` = paramètre de lissage (pays : 10, ville : 5)

Quand `n` est faible → convergence vers le prior.
Quand `n` est grand → prime locale domine.
    """)
with col_r:
    sample_cities = list(memory["city_freq_map"].keys())[:6]
    samples = [{
        "Ville": v,
        "n projets": memory["city_freq_map"].get(v, 0),
        "Coût lissé": f"${np.exp(memory['city_te_map'].get(v, memory['global_mean'])):,.0f} M/km",
    } for v in sample_cities]
    st.table(pd.DataFrame(samples))

# 7.5 Treemap couverture
st.markdown("### 7.5 Couverture Géographique du Modèle (taille = nombre de projets)")
country_data = [{
    "Pays": COUNTRY_NAMES_FR.get(c,c),
    "Projets": n,
    "Coût": np.exp(memory["country_te_map"].get(c, memory["global_mean"])),
} for c,n in memory["country_freq_map"].items()]
df_tree = pd.DataFrame(country_data)
df_tree["log_projets"] = np.log1p(df_tree["Projets"])

fig_tree = px.treemap(
    df_tree,
    path=["Pays"],
    values="Projets",
    color="log_projets",
    color_continuous_scale="Tealgrn",  
    range_color=(
        df_tree["log_projets"].quantile(0.05),
        df_tree["log_projets"].quantile(0.95)
    )
)

fig_tree.update_coloraxes(colorbar=dict(
    title="Projets",
    tickvals=[np.log1p(v) for v in [1, 5, 20, 100, 400]],
    ticktext=["1", "5", "20", "100", "400"],
))

fig_tree.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig_tree, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Système d'estimation adaptatif · Université de Galatasaray · "
    f"{ui_cfg.app_info['footer']} · "
)