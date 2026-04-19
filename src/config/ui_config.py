"""UI and theming configuration.."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UIConfig:
    """UI styling, colors, defaults, and templates."""

    # Color scheme
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#009999",
        "primary_hover": "#007A7A",
        "background": "#F4F5F7",
        "card": "#FFFFFF",
        "text_main": "#2D3748",
        "text_muted": "#718096",
        "accent_bg": "#E6F5F5",
        "border": "#E2E8F0",
        "success": "#059669",
        "error": "#E53E3E",
        "benchmark": "#94A3B8",
        "decrease": "#3182CE",
    })

    # Default input ranges
    defaults: Dict[str, Dict] = field(default_factory=lambda: {
        "length": {"min": 0.5, "max": 800.0, "default": 15.0, "step": 0.5},
        "tunnel": {"min": 0, "max": 100, "default": 80, "step": 5},
        "stations": {"min": 0, "max": 150, "default": 12, "step": 1},
        "year": {"min": 1950, "max": 2040, "default_start": 2022, "default_end": 2028},
        "actual_cost": {"min": 0.0, "max": 10000.0, "default": 0.0, "step": 10.0},
    })

    # Country-to-ISO-code mapping
    city_map: Dict[str, list] = field(default_factory=lambda: {
        "AE": ["Dubai"], "AR": ["Buenos Aires"], "AT": ["Vienna"],
        "AU": ["Melbourne", "Perth", "Sydney"], "BD": ["Dhaka"], "BE": ["Brussels"],
        "BG": ["Sofia"], "BH": ["Bahrain"], "BR": ["Rio de Janeiro", "Salvador", "Sao Paulo"],
        "CA": ["Calgary", "Edmonton", "Hamilton", "Mississauga", "Montreal", "Ottawa", "Toronto", "Vancouver"],
        "CH": ["Geneva", "Lucerne", "Zurich"], "CL": ["Santiago"],
        "CN": ["Beijing", "Changchun", "Changsha", "Changzhou", "Chengdu", "Chongqing",
               "Dalian", "Dongguan", "Foshan", "Fuzhou", "Guangzhou", "Guilin", "Guiyang",
               "Hangzhou", "Harbin", "Hefei", "Hohhot", "Jiaxing", "Jinan", "Kunming",
               "Lanzhou", "Luoyang", "Nanchang", "Nanjing", "Nanning", "Nantong", "Ningbo",
               "Putian", "Qingdao", "Shanghai", "Shaoxing", "Shenyang", "Shenzhen",
               "Shijiazhuang", "Suzhou", "Taiyuan", "Taizhou", "Tianjin", "Urumqi",
               "Wenzhou", "Wuhan", "Wuhu", "Wuxi", "Xi'an", "Xiamen", "Xuzhou", "Zhengzhou", "Zibo"],
        "CO": ["Bogotá"], "CZ": ["Prague"],
        "DE": ["Berlin", "Cologne", "Dusseldorf", "Frankfurt", "Hamburg", "Karlsruhe", "Leipzig", "Munich", "Nuremberg"],
        "DK": ["Copenhagen"], "DR": ["Santo Domingo"], "EC": ["Quito"], "EG": ["Cairo"],
        "ES": ["Barcelona", "Bilbao", "Madrid", "Malaga", "Seville"], "FI": ["Helsinki"],
        "FR": ["Lyon", "Paris", "Rennes", "Toulouse"],
        "GR": ["Athens", "Thessaloniki"], "HK": ["Hong Kong"], "HU": ["Budapest"],
        "ID": ["Jakarta"], "IL": ["Tel Aviv"],
        "IN": ["Agra", "Ahmadabad", "Bangalore", "Chennai", "Delhi", "Gurgaon", "Hyderabad",
               "Jaipur", "Kanpur", "Kochi", "Lucknow", "Mumbai", "Nagpur", "Patna", "Pune", "Surat"],
        "IR": ["Tehran"],
        "IT": ["Brescia", "Catania", "Genova", "Milan", "Naples", "Rome", "Turin"],
        "JP": ["Fukuoka", "Hiroshima", "Kobe", "Nagoya", "Osaka", "Sapporo", "Sendai", "Tokyo"],
        "KR": ["Busan", "Incheon", "Seoul"], "KW": ["Kuwait City"],
        "MX": ["Guadalajara", "Mexico City"], "MY": ["Kuala Lumpur"],
        "NL": ["Amsterdam"], "NO": ["Oslo"], "NZ": ["Auckland"], "PA": ["Panama City"],
        "PE": ["Lima"], "PH": ["Manila"], "PK": ["Lahore"], "PL": ["Warsaw", "Łódź"],
        "PT": ["Lisbon", "Porto"], "QA": ["Doha"],
        "RO": ["Bucharest", "Cluj-Napoca"], "RS": ["Belgrade"],
        "RU": ["Kazan", "Moscow", "N. Novgorod", "Samara", "St. Petersburg", "Yekaterinburg"],
        "SA": ["Ad Dammam", "Jeddah", "Riyadh"], "SE": ["Gothenburg", "Malmo", "Stockholm"],
        "SG": ["Singapore"], "TH": ["Bangkok", "Chiang Mai"],
        "TR": ["Adana", "Ankara", "Bursa", "Istanbul", "Izmir"],
        "TW": ["Kaohsiung", "Taichung", "Tainan", "Taipei"],
        "UA": ["Kharkiv", "Kyiv"], "UK": ["London"],
        "US": ["Boston", "Chicago", "Honolulu", "Los Angeles", "Miami", "New York",
               "Philadelphia", "San Francisco", "Seattle", "Washington"],
        "UZ": ["Tashkent"], "VN": ["Hanoi", "Ho Chi Minh City"],
    })

    # Country code to full name
    country_names: Dict[str, str] = field(default_factory=lambda: {
        "AE": "United Arab Emirates", "AR": "Argentina", "AT": "Austria",
        "AU": "Australia", "BD": "Bangladesh", "BE": "Belgium",
        "BG": "Bulgaria", "BH": "Bahrain", "BR": "Brazil",
        "CA": "Canada", "CH": "Switzerland", "CL": "Chile",
        "CN": "China", "CO": "Colombia", "CZ": "Czech Republic",
        "DE": "Germany", "DK": "Denmark", "DR": "Dominican Republic",
        "EC": "Ecuador", "EG": "Egypt", "ES": "Spain",
        "FI": "Finland", "FR": "France", "GR": "Greece",
        "HK": "Hong Kong", "HU": "Hungary", "ID": "Indonesia",
        "IL": "Israel", "IN": "India", "IR": "Iran",
        "IT": "Italy", "JP": "Japan", "KR": "South Korea",
        "KW": "Kuwait", "MX": "Mexico", "MY": "Malaysia",
        "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand",
        "PA": "Panama", "PE": "Peru", "PH": "Philippines",
        "PK": "Pakistan", "PL": "Poland", "PT": "Portugal",
        "QA": "Qatar", "RO": "Romania", "RS": "Serbia",
        "RU": "Russia", "SA": "Saudi Arabia", "SE": "Sweden",
        "SG": "Singapore", "TH": "Thailand", "TR": "Turkey",
        "TW": "Taiwan", "UA": "Ukraine", "UK": "United Kingdom",
        "US": "United States", "UZ": "Uzbekistan", "VN": "Vietnam",
    })

    # App metadata
    app_info: Dict[str, str] = field(default_factory=lambda: {
        "title": "Data Adaptive Rail Cost Estimator",
        "footer": "Galatasaray University - Computer Engineering Graduation Project - Inci Coskun",
    })

    # Similarity scoring weights
    similarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "loc_match": 3.0,
        "country_match": 1.5,
        "len_diff_scale": 1.0,
        "tun_diff_scale": 3.0,
    })

    # Max similarity score for normalizing
    similarity_max_score: float = 6.5

    # Constants for similarity calculation
    min_length_clamp: float = 0.5

    # Default subsystem cost ratios (fallback if FTA not available)
    default_subsystem_ratios: Dict[str, float] = field(default_factory=lambda: {
        "Guideway": 0.32,
        "Stations": 0.20,
        "Systems": 0.16,
        "Soft Costs": 0.14,
        "Vehicles": 0.08,
        "Right-of-Way (ROW)": 0.05,
        "Sitework": 0.03,
        "Facilities": 0.02,
    })

    # CSS styling
    @property
    def custom_css(self) -> str:
        """Return custom CSS for Streamlit app."""
        return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Sora', sans-serif !important; }}

[data-testid="stSidebar"] {{
    background-color: {self.colors['card']} !important;
    border-right: 1px solid {self.colors['border']} !important;
}}

div.stButton > button {{
    background-color: {self.colors['primary']} !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    width: 100% !important;
}}

[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button:hover,
[data-testid="stNumberInput"] button:focus,
[data-testid="stNumberInput"] button:active,
[data-testid="stNumberInput"] button:focus-visible {{
    background-color: transparent !important;
    color: {self.colors['text_muted']} !important;
    border: 1px solid {self.colors['border']} !important;
    box-shadow: none !important;
    outline: none !important;
    transition: background-color 0s !important;
}}
[data-testid="stNumberInput"] button:hover {{
    background-color: {self.colors['accent_bg']} !important;
    color: {self.colors['primary']} !important;
    border-color: {self.colors['primary']} !important;
}}
[data-testid="stNumberInput"] button svg {{
    fill: currentColor !important;
}}
</style>
"""

    @property
    def header_html(self) -> str:
        """Return header HTML template."""
        return f"""
<div style="padding-bottom:1.8rem; margin-bottom:2.2rem; border-bottom:1px solid {self.colors['border']};">
  <h1 style="font-size:38px; margin:0; font-weight:700; letter-spacing:-0.4px; color:{self.colors['text_main']};">
    {self.app_info['title']}
  </h1>
</div>
"""

    @property
    def empty_state_html(self) -> str:
        """Return empty state HTML template."""
        return f"""
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:45vh; text-align:center;">
  <div style="font-size:72px; margin-bottom:20px; opacity:0.8;">🚉</div>
  <div style="font-size:16px; color:{self.colors['text_muted']};">Select project parameters and click Generate Prediction to begin analysis.</div>
</div>
"""
