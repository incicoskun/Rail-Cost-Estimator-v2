"""Rail Cost Estimator - Streamlit Application"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
import time

from src import AppConfig
from src.constants import TransitMode
from src.inference.service import InferenceService

warnings.filterwarnings("ignore")

# Initialize configuration
@st.cache_resource
def get_config():
    """Load configuration once."""
    return AppConfig.from_env()

config = get_config()
ui_cfg = config.ui_config
file_cfg = config.file_config

# ─────────────────────────────────────────────────────────────────────────────
# Setup Streamlit page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=ui_cfg.app_info["title"],
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(ui_cfg.custom_css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_training_data():
    """Load and prepare training data for similarity matching."""
    try:
        df = pd.read_csv(file_cfg.global_rail_csv)
        df = df.dropna(subset=["cost_per_km_2023_musd"])
        df["tunnel_pct"] = df["tunnel_pct"].apply(lambda x: x / 100 if x > 1 else x)
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_model_artifacts():
    """Load trained model and memory package (cached for session)."""
    t0 = time.time()
    try:
        out_dir = file_cfg.data_processed.parent.parent
        model = joblib.load(out_dir / "rail_cost_model.pkl")
        memory = joblib.load(out_dir / "memory_package.pkl")
        elapsed = time.time() - t0
        st.session_state["model_load_time"] = elapsed
        return model, memory, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_fta_lookup():
    """Load FTA lookup for subsystem breakdown (cached for session)."""
    t0 = time.time()
    try:
        from src import FTALookup
        fta = FTALookup.from_csv(str(file_cfg.fta_processed_csv))
        elapsed = time.time() - t0
        st.session_state["fta_load_time"] = elapsed
        return fta, None
    except FileNotFoundError as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity and prediction functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_inference_service(_model, _memory, _fta_lookup) -> InferenceService:
    """Instantiate and cache InferenceService for the session."""
    return InferenceService(
        model=_model,
        memory=_memory,
        fta_lookup=_fta_lookup,
        config=config.model_config,
        ui_config=config.ui_config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────

model, memory, load_error = load_model_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar - Input collection
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Project Configuration")

    clist = sorted(ui_cfg.city_map.keys()) + ["Other"]
    country_choice = st.selectbox(
        "Country", clist,
        index=clist.index("TR") if "TR" in clist else 0,
        format_func=lambda x: ui_cfg.country_names.get(x, x)
    )

    if country_choice == "Other":
        country = st.text_input("New Country Name", "Unknown")
        city = st.text_input("New City Name", "New City")
        st.info("Using global baseline for unknown locations.")
    else:
        country = country_choice
        city_choice = st.selectbox("City", ui_cfg.city_map.get(country, []) + ["Other"])
        if city_choice == "Other":
            city = st.text_input("New City Name", "New City")
            st.info(f"Using {country} national baseline for this city.")
        else:
            city = city_choice

    st.divider()

    length_km = st.number_input(
        "Line length (km)",
        ui_cfg.defaults["length"]["min"],
        ui_cfg.defaults["length"]["max"],
        ui_cfg.defaults["length"]["default"],
        ui_cfg.defaults["length"]["step"]
    )

    tunnel_pct = st.slider(
        "Tunnel share (%)",
        ui_cfg.defaults["tunnel"]["min"],
        ui_cfg.defaults["tunnel"]["max"],
        ui_cfg.defaults["tunnel"]["default"],
        ui_cfg.defaults["tunnel"]["step"]
    ) / 100.0

    num_stations = st.number_input(
        "Stations",
        ui_cfg.defaults["stations"]["min"],
        ui_cfg.defaults["stations"]["max"],
        ui_cfg.defaults["stations"]["default"],
        ui_cfg.defaults["stations"]["step"]
    )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input(
            "Start",
            ui_cfg.defaults["year"]["min"],
            ui_cfg.defaults["year"]["max"],
            ui_cfg.defaults["year"]["default_start"]
        )
    with c2:
        end_year = st.number_input(
            "End",
            ui_cfg.defaults["year"]["min"],
            ui_cfg.defaults["year"]["max"],
            ui_cfg.defaults["year"]["default_end"]
        )

    is_regional = st.checkbox("Regional Rail project")

    transit_mode = st.selectbox(
        "Transit Mode", [m.value for m in TransitMode], index=0,
        help="HRT: Heavy Rail · LRT: Light Rail · BRT: Bus Rapid Transit · CRT: Commuter Rail"
    )

    st.divider()
    run = st.button("Generate Prediction →")


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(ui_cfg.header_html, unsafe_allow_html=True)

if load_error:
    st.error(f"**Asset Load Error:** {load_error}")
    st.stop()

if run and end_year >= start_year:
    fta_lookup, fta_err = load_fta_lookup()
    svc = get_inference_service(model, memory, fta_lookup)

    result = svc.predict({
        "country": country,
        "city": city,
        "length_km": length_km,
        "tunnel_pct": tunnel_pct,
        "num_stations": num_stations,
        "start_year": start_year,
        "end_year": end_year,
        "is_regional": is_regional,
        "transit_mode": transit_mode,
    })
    st.session_state["result"] = result

r = st.session_state.get("result")
if not r:
    st.markdown(ui_cfg.empty_state_html, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

full_country_name = ui_cfg.country_names.get(r["country"], r["country"])
pred, lo, hi = r["pred"], r["lo"], r["hi"]
total_b = r["pred"] * r["length_km"] / 1000.0
colors = ui_cfg.colors
fta_lookup, _ = load_fta_lookup()
svc = get_inference_service(model, memory, fta_lookup)

# 1. KPIs
def kpi_card(label: str, value: str) -> str:
    """Generate KPI card HTML."""
    return f"""
    <div style="background:{colors['card']}; border:1px solid {colors['border']};
                border-radius:12px; padding:16px 20px; margin-bottom:10px;">
        <div style="font-size:13px; color:{colors['text_muted']}; margin-bottom:6px;">{label}</div>
        <div style="font-size:28px; font-weight:600; color:{colors['text_main']};">{value}</div>
    </div>
    """

def card_bar_ratio(pct: float) -> str:
    """Generate progress bar HTML."""
    pct_clamped = max(0.0, min(pct, 100.0))
    return (
        f"<div style='background:#EDF2F7; height:8px; border-radius:4px;'>"
        f"<div style='background:{colors['primary']}; width:{pct_clamped}%; "
        f"height:100%; border-radius:4px;'></div></div>"
    )

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(kpi_card("Predicted Cost / KM", f"${pred:,.0f} M"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Probability Band", f"${lo:,.0f} – ${hi:,.0f} M"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Total Project Budget", f"${total_b:.2f} B"), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# 2. Subsystem Breakdown
st.markdown(
    f"<h4 style='font-size:13px; color:{colors['text_muted']}; text-transform:uppercase;'>"
    f"Estimated Subsystem Cost Breakdown</h4>",
    unsafe_allow_html=True
)

_breakdown = svc.get_breakdown(r)
bdi = _breakdown["breakdown"]

breakdown_df = pd.DataFrame([
    {
        "Category": label,
        "Ratio": bdi[label]["point"] / pred,
        "Cost per km (M$)": bdi[label]["point"],
        "Lo (M$/km)": bdi[label]["lo"],
        "Hi (M$/km)": bdi[label]["hi"],
        "Total Point (M$)": bdi[label]["point"] * r["length_km"],
        "Total Lo (M$)": bdi[label]["lo"] * r["length_km"],
        "Total Hi (M$)": bdi[label]["hi"] * r["length_km"],
    }
    for label in bdi
])

breakdown_df["range_str"] = breakdown_df.apply(
    lambda row: f"${row['Lo (M$/km)']:,.1f} – ${row['Hi (M$/km)']:,.1f} M/km", axis=1
)

col_chart, col_table = st.columns([2, 3])
with col_chart:
    fig_pie = go.Figure(go.Pie(
        labels=breakdown_df["Category"],
        values=breakdown_df["Cost per km (M$)"],
        hole=0.6,
        textposition="inside",
        textinfo="percent+label",
        marker_colors=px.colors.qualitative.Prism[:len(breakdown_df)],
        customdata=breakdown_df["range_str"].values,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Point: $%{value:,.1f} M/km<br>"
            "Range: %{customdata}"
            "<extra></extra>"
        ),
    ))
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"Total<br>${r['pred'] * r['length_km']:,.0f}M",
            x=0.5, y=0.5, font_size=20, showarrow=False
        )]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_table:
    display_bd = breakdown_df[[
        "Category", "Ratio", "Cost per km (M$)", "Lo (M$/km)", "Hi (M$/km)",
        "Total Point (M$)", "Total Lo (M$)", "Total Hi (M$)"
    ]].copy()
    display_bd["Share"] = (display_bd["Ratio"] * 100).map("{:.1f}%".format)
    display_bd["Point (M$/km)"] = display_bd["Cost per km (M$)"].map("${:,.1f}".format)
    display_bd["Range (M$/km)"] = display_bd.apply(
        lambda row: f"${row['Lo (M$/km)']:,.1f} – ${row['Hi (M$/km)']:,.1f}", axis=1
    )
    display_bd["Total Point"] = display_bd["Total Point (M$)"].map("${:,.0f}M".format)
    display_bd["Total Range"] = display_bd.apply(
        lambda row: f"${row['Total Lo (M$)']:,.0f}M – ${row['Total Hi (M$)']:,.0f}M", axis=1
    )
    st.dataframe(
        display_bd[["Category", "Share", "Point (M$/km)", "Range (M$/km)", "Total Point", "Total Range"]].style.set_properties(**{
            "background-color": colors["card"],
            "color": colors["text_main"],
            "border-color": colors["border"],
        }),
        use_container_width=True,
        hide_index=True
    )


# 4. Context & Summary
left, right = st.columns([3, 2])
with left:
    st.markdown(
        f"<h4 style='font-size:13px; color:{colors['text_muted']}; text-transform:uppercase;'>"
        f"Market Context</h4>",
        unsafe_allow_html=True
    )
    _ctx = svc.get_market_context(r)
    st.table(pd.DataFrame({
        "City Baseline": f"${np.exp(_ctx['city_baseline']):,.0f} M/km",
        f"{full_country_name} Baseline": f"${np.exp(_ctx['country_baseline']):,.0f} M/km",
        "Sample Density": f"{_ctx['sample_density']} Projects",
    }.items(), columns=["Signal", "Status"]))

with right:
    st.markdown(
        f"<h4 style='font-size:13px; color:{colors['text_muted']}; text-transform:uppercase;'>"
        f"Project Summary</h4>",
        unsafe_allow_html=True
    )
    st.table(pd.DataFrame({
        "Location": f"{r['city']}, {full_country_name}",
        "Topology": f"{r['length_km']:.1f} km / {r['tunnel_pct']*100:.0f}% Tunnel",
        "Stations": f"{r['num_stations']} Units",
        "Mode": r.get("transit_mode", "—"),
    }.items(), columns=["Field", "Value"]))

# 5. Global Benchmarks
st.divider()
st.markdown(
    f"<h4 style='font-size:13px; color:{colors['text_muted']}; text-transform:uppercase;'>"
    f"Global Comparative Benchmarks</h4>",
    unsafe_allow_html=True
)

train_df = load_training_data()
if train_df is not None:
    disp = svc.find_similar_projects(train_df, r)
    disp["CountryName"] = disp["country"].map(ui_cfg.country_names).fillna(disp["country"])

    max_score = ui_cfg.similarity_max_score
    disp["Match"] = (
        (disp["score"] / max_score) * 100
    ).clip(0, 100).round(0).astype(int).astype(str) + "%"
    disp["Length (km)"] = disp["length_km"].apply(lambda x: f"{x:.1f}")
    disp["Tunnel"] = (disp["tunnel_pct"] * 100).round(0).astype(int).astype(str) + "%"
    disp["Cost (M$/km)"] = disp["cost_per_km_2023_musd"].apply(lambda x: f"${x:,.0f} M")

    disp_final = disp[[
        "Match", "CountryName", "city", "line", "Length (km)", "Tunnel", "Cost (M$/km)"
    ]].rename(columns={
        "CountryName": "Country", "city": "City", "line": "Line Name"
    })

    st.dataframe(
        disp_final.style.apply(
            lambda row: [f"background-color: {colors['accent_bg']}; color: {colors['primary']}; font-weight: 600"] * len(row)
            if row["Country"] == full_country_name and row["City"] == r["city"]
            else [""] * len(row),
            axis=1
        ),
        use_container_width=True,
        hide_index=True
    )

# 6. Interactive Cost Drivers (SHAP)
st.divider()
st.markdown(
    f"<h4 style='font-size:13px; color:{colors['text_muted']}; text-transform:uppercase;'>"
    f"Interactive Cost Drivers</h4>",
    unsafe_allow_html=True
)

try:
    bar_df = svc.explain(r)
    bar_df["Text Value"] = bar_df["Signed_Percent"].apply(lambda x: f"{x:+.1f}%")

    fig_bar = px.bar(
        bar_df, x="Signed_Percent", y="Factor", orientation="h",
        color="Direction",
        color_discrete_map={
            "Increases Cost (+)": colors["primary"],
            "Decreases Cost (-)": colors["error"],
        },
        text="Text Value"
    )
    fig_bar.update_traces(
        textposition="outside",
        marker=dict(line=dict(color=colors["card"], width=1.5)),
        insidetextfont=dict(color=colors["text_main"], family="Sora, sans-serif"),
        outsidetextfont=dict(color=colors["text_main"], family="Sora, sans-serif"),
    )
    fig_bar.update_layout(
        font=dict(family="Sora, sans-serif", color=colors["text_main"]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        xaxis_title="Relative Impact on Final Cost (%)",
        yaxis_title="",
        margin=dict(t=60, b=40, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    fig_bar.update_xaxes(
        showgrid=True,
        gridcolor=colors["border"],
        zeroline=True,
        zerolinecolor=colors["text_muted"],
        zerolinewidth=2,
        tickfont=dict(size=12, color=colors["text_main"]),
    )
    fig_bar.update_yaxes(tickmode="linear", tickfont=dict(size=12, color=colors["text_main"]))
    st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.warning(f"Could not generate interactive explanation: {e}")

st.caption(ui_cfg.app_info["footer"])