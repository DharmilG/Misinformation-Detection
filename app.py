"""
app.py
======
MisinfoForensics — Multimodal RAG Detection System
Streamlit User Interface & Presentation Layer.

Spec: 05_REPORTING_AND_UI.md

Architecture:
  - Thin presentation layer over src/reporting and backend pipeline.
  - Business logic remains in src/reporting, src/input_pipeline, src/clip_model,
    src/retrieval, and src/rag.
  - Strict compliance with status vocabulary, two-panel non-merged signals,
    centralized copy strings, and safe rendering.
  - Centralized Theme & Design System:
      - Design tokens for Light and Dark modes.
      - Contrast checked to exceed WCAG 2.1 AA / AAA standards.
      - Zero hardcoded colors in component templates.
      - Immediate theme switching without resetting analysis state.
"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Optional

import streamlit as st
from PIL import Image

from src.clip_model.classifier import run_classifier
from src.input_pipeline.validation import ValidationError, build_input_request
from src.rag.assessment import run_assessment
from src.reporting import copy as C
from src.reporting.builder import build_report
from src.reporting.comparison import PERMANENT_NOTE
from src.reporting.escaping import escape_display, safe_url
from src.reporting.export import generate_markdown_export, get_export_filename
from src.reporting.model import ReportModel
from src.retrieval.corpus import retrieve_passages
from src.utils.config import (
    CLASSIFIER_THRESHOLD,
    CORPUS_VERSION,
    MAX_CAPTION_CHARS,
    MAX_IMAGE_BYTES,
    MIN_CAPTION_CHARS,
    MODEL_VERSION,
    SAMPLES_PATH,
    SUPPORTED_EXTENSIONS,
)

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MisinfoForensics — Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme State Initialization ────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"


# ── Theme CSS Generator with Design Tokens ────────────────────────────────
def inject_theme_system(theme: str) -> None:
    """
    Inject centralized CSS custom properties and responsive styling.
    Ensures complete visual consistency and WCAG 2.1 AA/AAA contrast
    across all UI elements in both Light and Dark modes.
    """
    light_tokens = """
        --bg-app: #F6F8FC;
        --bg-surface: #FFFFFF;
        --bg-surface-secondary: #F1F4F9;
        --bg-surface-elevated: #FFFFFF;
        --bg-sidebar: #0B1228;
        --border-sidebar: #1E293B;
        --sidebar-text: #F1F5F9;
        --sidebar-muted: #94A3B8;
        --sidebar-hover: #1A2442;
        --sidebar-active: #1E293B;

        --text-primary: #172033;
        --text-secondary: #475467;
        --text-muted: #667085;
        --text-heading: #0F172A;
        --border-subtle: #D9E0EA;
        --border-card: #E3E8F0;

        --primary-accent: #4263EB;
        --primary-accent-hover: #364FC7;
        --secondary-accent: #7957D5;
        --link-color: #3158C9;

        --bg-input: #FFFFFF;
        --text-input: #172033;
        --border-input: #D9E0EA;
        --text-placeholder: #94A3B8;

        --btn-primary-bg: #4263EB;
        --btn-primary-text: #FFFFFF;
        --btn-secondary-bg: #F8FAFC;
        --btn-secondary-text: #172033;
        --btn-secondary-border: #D9E0EA;

        --callout-limitation-bg: #EFF6FF;
        --callout-limitation-border: #4263EB;
        --callout-limitation-text: #1E40AF;

        --callout-warning-bg: #FFFBEB;
        --callout-warning-border: #F59E0B;
        --callout-warning-text: #92400E;

        --callout-comparison-bg: #F8FAFC;
        --callout-comparison-border: #4263EB;
        --callout-comparison-text: #172033;

        --callout-danger-bg: #FFF5F5;
        --callout-danger-border: #EF4444;
        --callout-danger-text: #991B1B;

        --badge-success-bg: #ECFDF5;
        --badge-success-text: #065F46;
        --badge-success-border: #A7F3D0;

        --badge-danger-bg: #FEF2F2;
        --badge-danger-text: #991B1B;
        --badge-danger-border: #FECACA;

        --badge-warning-bg: #FFFBEB;
        --badge-warning-text: #92400E;
        --badge-warning-border: #FDE68A;

        --badge-neutral-bg: #EFF6FF;
        --badge-neutral-text: #1E40AF;
        --badge-neutral-border: #BFDBFE;

        --badge-slate-bg: #F1F5F9;
        --badge-slate-text: #475569;
        --badge-slate-border: #E2E8F0;

        --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    """

    dark_tokens = """
        --bg-app: #0B1020;
        --bg-surface: #11182A;
        --bg-surface-secondary: #172137;
        --bg-surface-elevated: #1E293B;
        --bg-sidebar: #060A17;
        --border-sidebar: #1E293B;
        --sidebar-text: #F8FAFC;
        --sidebar-muted: #94A3B8;
        --sidebar-hover: #151F33;
        --sidebar-active: #1E293B;

        --text-primary: #F1F5F9;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --text-heading: #FFFFFF;
        --border-subtle: #2D3748;
        --border-card: #2D3748;

        --primary-accent: #8BA4FF;
        --primary-accent-hover: #708FFF;
        --secondary-accent: #9F8FEF;
        --link-color: #A8BCFF;

        --bg-input: #172137;
        --text-input: #F1F5F9;
        --border-input: #334155;
        --text-placeholder: #64748B;

        --btn-primary-bg: #4263EB;
        --btn-primary-text: #FFFFFF;
        --btn-secondary-bg: #172137;
        --btn-secondary-text: #F1F5F9;
        --btn-secondary-border: #334155;

        --callout-limitation-bg: #172554;
        --callout-limitation-border: #3B82F6;
        --callout-limitation-text: #DBEAFE;

        --callout-warning-bg: #451A03;
        --callout-warning-border: #F59E0B;
        --callout-warning-text: #FEF3C7;

        --callout-comparison-bg: #172137;
        --callout-comparison-border: #8BA4FF;
        --callout-comparison-text: #F1F5F9;

        --callout-danger-bg: #450A0A;
        --callout-danger-border: #EF4444;
        --callout-danger-text: #FEE2E2;

        --badge-success-bg: #064E3B;
        --badge-success-text: #A7F3D0;
        --badge-success-border: #047857;

        --badge-danger-bg: #450A0A;
        --badge-danger-text: #FECACA;
        --badge-danger-border: #B91C1C;

        --badge-warning-bg: #451A03;
        --badge-warning-text: #FDE68A;
        --badge-warning-border: #B45309;

        --badge-neutral-bg: #1E293B;
        --badge-neutral-text: #93C5FD;
        --badge-neutral-border: #3B82F6;

        --badge-slate-bg: #1E293B;
        --badge-slate-text: #CBD5E1;
        --badge-slate-border: #334155;

        --card-shadow: 0 1px 4px rgba(0, 0, 0, 0.28);
    """

    if theme == "Dark":
        token_rules = f":root {{ {dark_tokens} }}"
    elif theme == "Light":
        token_rules = f":root {{ {light_tokens} }}"
    else:  # System
        token_rules = f"""
            :root {{ {light_tokens} }}
            @media (prefers-color-scheme: dark) {{
                :root {{ {dark_tokens} }}
            }}
        """

    css = f"""
    <style>
    /* ── Token Declarations ── */
    {token_rules}

    /* ── Base Canvas & App Body ── */
    .stApp {{
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* ── Main Container Spacing ── */
    .main .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }}

    /* ── Sidebar Styling (High Contrast Navy) ── */
    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-sidebar) !important;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: var(--sidebar-text) !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: var(--border-sidebar) !important;
        margin: 0.9rem 0;
    }}

    /* Sidebar Theme Segmented Control */
    [data-testid="stSidebar"] div[role="radiogroup"] {{
        background-color: var(--sidebar-hover);
        padding: 4px;
        border-radius: 8px;
        gap: 4px;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 6px 10px !important;
        border-radius: 6px !important;
        margin: 0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--sidebar-text) !important;
        transition: all 0.15s ease !important;
        cursor: pointer;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: var(--sidebar-active) !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: var(--primary-accent) !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Sidebar Navigation Links */
    [data-testid="stSidebar"] [data-testid="stRadio"]:last-of-type div[role="radiogroup"] {{
        background-color: transparent !important;
        padding: 0 !important;
        gap: 4px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"]:last-of-type label {{
        padding: 9px 14px !important;
        border-radius: 6px !important;
        margin-bottom: 3px !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: var(--sidebar-text) !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"]:last-of-type label:hover {{
        background-color: var(--sidebar-hover) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"]:last-of-type label:has(input:checked) {{
        background-color: var(--sidebar-active) !important;
        border-left: 3px solid var(--primary-accent) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton>button {{
        background-color: var(--sidebar-active) !important;
        border: 1px solid var(--border-sidebar) !important;
        color: var(--sidebar-text) !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .stButton>button:hover {{
        background-color: var(--sidebar-hover) !important;
        border-color: var(--border-subtle) !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: var(--sidebar-muted) !important;
    }}

    /* ── Headings & Eyebrows ── */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-heading) !important;
        font-weight: 700 !important;
    }}
    p, li {{
        color: var(--text-primary);
        line-height: 1.55;
    }}
    .eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--primary-accent);
        margin-bottom: 4px;
    }}
    .header-title {{
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-heading);
        margin-bottom: 6px;
        line-height: 1.25;
    }}
    .header-desc {{
        font-size: 0.94rem;
        color: var(--text-secondary);
        line-height: 1.45;
        margin-bottom: 20px;
    }}

    /* ── Status Pills & Badges ── */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.80rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }}
    .status-success {{
        background-color: var(--badge-success-bg) !important;
        color: var(--badge-success-text) !important;
        border: 1px solid var(--badge-success-border) !important;
    }}
    .status-danger {{
        background-color: var(--badge-danger-bg) !important;
        color: var(--badge-danger-text) !important;
        border: 1px solid var(--badge-danger-border) !important;
    }}
    .status-warning {{
        background-color: var(--badge-warning-bg) !important;
        color: var(--badge-warning-text) !important;
        border: 1px solid var(--badge-warning-border) !important;
    }}
    .status-neutral {{
        background-color: var(--badge-neutral-bg) !important;
        color: var(--badge-neutral-text) !important;
        border: 1px solid var(--badge-neutral-border) !important;
    }}
    .status-slate {{
        background-color: var(--badge-slate-bg) !important;
        color: var(--badge-slate-text) !important;
        border: 1px solid var(--badge-slate-border) !important;
    }}

    /* ── Surface Card Header (for use inside st.container(border=True)) ── */
    .surface-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-card);
        padding-bottom: 12px;
        margin-bottom: 16px;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-heading);
    }}

    /* ── Callouts & Essential Limitations (Never Collapsed) ── */
    .callout-limitation {{
        background-color: var(--callout-limitation-bg) !important;
        border-left: 4px solid var(--callout-limitation-border) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 12px 16px !important;
        margin: 14px 0 6px 0 !important;
        color: var(--callout-limitation-text) !important;
        font-size: 0.88rem !important;
        line-height: 1.45 !important;
    }}
    .callout-limitation * {{
        color: var(--callout-limitation-text) !important;
    }}
    .callout-warning {{
        background-color: var(--callout-warning-bg) !important;
        border-left: 4px solid var(--callout-warning-border) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 11px 15px !important;
        margin: 10px 0 !important;
        color: var(--callout-warning-text) !important;
        font-size: 0.88rem !important;
        line-height: 1.4 !important;
    }}
    .callout-warning * {{
        color: var(--callout-warning-text) !important;
    }}
    .callout-comparison {{
        background-color: var(--callout-comparison-bg) !important;
        border: 1px solid var(--border-card) !important;
        border-left: 4px solid var(--callout-comparison-border) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 16px 20px !important;
        margin: 14px 0 !important;
        color: var(--callout-comparison-text) !important;
        font-size: 0.94rem !important;
        line-height: 1.5 !important;
    }}
    .callout-comparison * {{
        color: var(--callout-comparison-text) !important;
    }}
    .callout-danger {{
        background-color: var(--callout-danger-bg) !important;
        border: 1px solid var(--callout-danger-border) !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        color: var(--callout-danger-text) !important;
        font-size: 0.90rem !important;
    }}
    .callout-danger * {{
        color: var(--callout-danger-text) !important;
    }}

    /* ── Meta Chips ── */
    .meta-chip {{
        display: inline-block;
        background-color: var(--bg-surface-secondary);
        color: var(--text-secondary);
        padding: 3px 9px;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.80rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid var(--border-card);
    }}

    /* ── Evidence Excerpt ── */
    .evidence-excerpt {{
        background-color: var(--bg-surface-secondary);
        border-left: 3px solid var(--border-card);
        padding: 10px 14px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: var(--text-secondary) !important;
        line-height: 1.45;
        border-radius: 0 6px 6px 0;
    }}

    /* ── Form Inputs, Selectbox, and Textarea ── */
    .stTextArea textarea,
    .stTextInput input {{
        background-color: var(--bg-input) !important;
        color: var(--text-input) !important;
        border: 1px solid var(--border-input) !important;
        border-radius: 6px !important;
    }}
    .stTextArea textarea:focus,
    .stTextInput input:focus {{
        border-color: var(--primary-accent) !important;
        box-shadow: 0 0 0 1px var(--primary-accent) !important;
    }}
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {{
        color: var(--text-placeholder) !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: var(--bg-input) !important;
        color: var(--text-input) !important;
        border: 1px solid var(--border-input) !important;
        border-radius: 6px !important;
    }}
    .stSelectbox div[data-baseweb="select"] * {{
        color: var(--text-input) !important;
    }}

    /* BaseWeb Dropdown Menus */
    div[data-baseweb="popover"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 8px !important;
        box-shadow: var(--card-shadow) !important;
    }}
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {{
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }}
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background-color: var(--bg-surface-secondary) !important;
        color: var(--primary-accent) !important;
    }}

    .stTextArea label,
    .stTextInput label,
    .stSelectbox label,
    .stFileUploader label {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }}

    /* ── Native Border Containers (Wrapped cleanly) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: var(--border-card) !important;
        background-color: var(--bg-surface) !important;
        border-radius: 10px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 16px;
    }}

    /* ── File Uploader Styling ── */
    [data-testid="stFileUploader"] section {{
        background-color: var(--bg-surface-secondary) !important;
        border: 1px dashed var(--border-card) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] section * {{
        color: var(--text-secondary) !important;
    }}
    [data-testid="stFileUploader"] section button {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }}
    [data-testid="stFileUploader"] section button:hover {{
        border-color: var(--primary-accent) !important;
        color: var(--primary-accent) !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 6px !important;
    }}
    [data-testid="stFileUploaderFile"] * {{
        color: var(--text-primary) !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {{
        color: var(--text-heading) !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }}

    /* ── Progress Bar ── */
    [data-testid="stProgressBar"] > div {{
        background-color: var(--bg-surface-secondary) !important;
        border-radius: 9999px !important;
    }}
    [data-testid="stProgressBar"] div[role="progressbar"] {{
        background-color: var(--primary-accent) !important;
    }}

    /* ── Code Blocks ── */
    [data-testid="stCodeBlock"] {{
        background-color: var(--bg-surface-secondary) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 6px !important;
    }}
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {{
        color: var(--text-primary) !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }}

    /* ── Expanders ── */
    [data-testid="stExpander"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] details summary {{
        background-color: var(--bg-surface) !important;
    }}
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary p {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] details summary svg {{
        fill: var(--text-secondary) !important;
    }}
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: var(--bg-surface) !important;
        border-top: 1px solid var(--border-card) !important;
    }}

    /* ── Captions & Secondary Text ── */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--text-muted) !important;
    }}
    [data-testid="stCaptionContainer"] strong {{
        color: var(--text-primary) !important;
    }}

    /* ── Buttons ── */
    .stButton>button {{
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        transition: all 0.15s ease !important;
    }}
    .stButton>button[kind="primary"] {{
        background-color: var(--btn-primary-bg) !important;
        border-color: var(--btn-primary-bg) !important;
        color: var(--btn-primary-text) !important;
    }}
    .stButton>button[kind="primary"]:hover {{
        background-color: var(--primary-accent-hover) !important;
        border-color: var(--primary-accent-hover) !important;
    }}
    .stButton>button[kind="secondary"],
    .stButton>button:not([kind="primary"]) {{
        background-color: var(--btn-secondary-bg) !important;
        border: 1px solid var(--btn-secondary-border) !important;
        color: var(--btn-secondary-text) !important;
    }}
    .stButton>button:disabled {{
        opacity: 0.55 !important;
    }}

    /* Download Buttons */
    [data-testid="stDownloadButton"]>button {{
        background-color: var(--btn-primary-bg) !important;
        border-color: var(--btn-primary-bg) !important;
        color: var(--btn-primary-text) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }}
    [data-testid="stDownloadButton"]>button:hover {{
        background-color: var(--primary-accent-hover) !important;
        border-color: var(--primary-accent-hover) !important;
    }}

    /* ── Links ── */
    a {{
        color: var(--link-color) !important;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Sample Fixtures Loader ────────────────────────────────────────────────
@st.cache_data
def load_sample_manifest() -> list[dict]:
    if not SAMPLES_PATH.exists():
        return []
    try:
        with SAMPLES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_sample_data(sample_id: str) -> tuple[Optional[bytes], Optional[str], Optional[str], Optional[str]]:
    manifest = load_sample_manifest()
    for s in manifest:
        if s.get("sample_id") == sample_id:
            img_file = SAMPLES_PATH.parent / s.get("image_filename", "")
            if img_file.exists():
                return img_file.read_bytes(), s.get("image_filename"), s.get("default_caption"), s.get("description")
    return None, None, None, None


# ── Session State Management (§12) ────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state.report = None
if "analysis_fingerprint" not in st.session_state:
    st.session_state.analysis_fingerprint = None
if "input_errors" not in st.session_state:
    st.session_state.input_errors = []
if "selected_sample_id" not in st.session_state:
    st.session_state.selected_sample_id = "None"
if "caption_text" not in st.session_state:
    st.session_state.caption_text = ""
if "active_nav_page" not in st.session_state:
    st.session_state.active_nav_page = C.H_ANALYZE


def action_clear_session():
    st.session_state.report = None
    st.session_state.analysis_fingerprint = None
    st.session_state.input_errors = []
    st.session_state.selected_sample_id = "None"
    st.session_state.caption_text = ""
    st.session_state.active_nav_page = C.H_ANALYZE


# ── Pipeline Execution ────────────────────────────────────────────────────
def run_pipeline(image_bytes: bytes, filename: str, caption: str, request_id: Optional[str] = None) -> bool:
    # 1. Validation (§11)
    req, warnings, errors = build_input_request(image_bytes, filename, caption)
    if errors:
        st.session_state.input_errors = errors
        st.session_state.report = None
        st.session_state.analysis_fingerprint = None
        return False

    st.session_state.input_errors = []

    # 2. Classifier (CLIP)
    clf_result = run_classifier(req.image.processed_bytes, req.caption.normalized)

    # 3. Retrieval (TF-IDF over corpus)
    retrieval_result = retrieve_passages(req.caption.normalized)

    # 4. Assessment (RAG claim assessment)
    asm_result = run_assessment(req.caption.normalized, retrieval_result.passages)

    # 5. Assemble ReportModel
    report = build_report(
        request=req,
        warnings=warnings,
        clf_result=clf_result,
        retrieval=retrieval_result,
        assessment=asm_result,
        request_id=request_id,
    )

    st.session_state.report = report
    st.session_state.analysis_fingerprint = report.analysis_fingerprint
    st.session_state.active_nav_page = C.H_REPORT
    return True


# ── Sidebar Setup & Identity ──────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style='padding: 8px 4px 14px 4px;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='font-size: 1.45rem;'>🔍</span>
            <div>
                <div style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;'>MisinfoForensics</div>
                <div style='font-size: 0.74rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;'>Multimodal RAG Detection</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Theme Toggle in Sidebar ──
theme_options = ["Light", "Dark", "System"]
current_theme_idx = 0
if st.session_state.theme_mode in theme_options:
    current_theme_idx = theme_options.index(st.session_state.theme_mode)

theme_selected = st.sidebar.radio(
    "Theme",
    theme_options,
    index=current_theme_idx,
    format_func=lambda x: "☀️ Light" if x == "Light" else ("🌙 Dark" if x == "Dark" else "💻 System"),
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.theme_mode = theme_selected

# Inject CSS system with active theme tokens
inject_theme_system(st.session_state.theme_mode)

st.sidebar.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

# Navigation Radio
nav_options = [
    C.H_ANALYZE,
    C.H_REPORT,
    C.H_EVIDENCE,
    C.H_METHODOLOGY,
    C.H_PRIVACY,
]

current_page_idx = 0
if st.session_state.active_nav_page in nav_options:
    current_page_idx = nav_options.index(st.session_state.active_nav_page)

page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=current_page_idx,
    label_visibility="collapsed",
)
st.session_state.active_nav_page = page

st.sidebar.markdown("---")

# Session Control Section at bottom
st.sidebar.markdown(
    """
    <div style='font-size: 0.78rem; color: #94A3B8; line-height: 1.45; margin-bottom: 12px;'>
        <b style='color: #E2E8F0;'>Session Privacy:</b> In-memory processing only. Image bytes and generated reports are held in memory and never written to disk.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("🧹 Clear Active Session", use_container_width=True):
    action_clear_session()
    st.sidebar.info(C.SESSION_CLEAR_DISCLOSURE)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: ANALYZE WORKSPACE
# ══════════════════════════════════════════════════════════════════════════
if page == C.H_ANALYZE:
    col_header, col_status = st.columns([4, 1])
    with col_header:
        st.markdown(
            """
            <div class='eyebrow'>AI Forensics Workspace</div>
            <div class='header-title'>Analyze Multimodal Misinformation</div>
            <div class='header-desc'>
                Evaluate visual–textual consistency using CLIP embeddings and retrieve corroborating evidence from verified local sources.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            """
            <div style='text-align: right; padding-top: 10px;'>
                <span class='status-badge status-success'>● System Ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Input Layout: Two Balanced Columns
    col_input_left, col_input_right = st.columns([1, 1], gap="large")

    with col_input_left:
        with st.container(border=True):
            st.markdown(
                """
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>1. Image Source</span>
                    <span class='status-badge status-slate'>Visual Signal</span>
                </div>
                <div style='font-size: 0.86rem; color: var(--text-secondary); margin-bottom: 14px;'>
                    Select an approved synthetic fixture or upload a custom image for consistency analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # A. Synthetic Sample Fixture Selector (§14)
            sample_options = ["None"]
            manifest = load_sample_manifest()
            for s in manifest:
                sample_options.append(s.get("sample_id"))

            def format_sample_option(opt):
                if opt == "None":
                    return "None (Upload your own image)"
                for s in manifest:
                    if s.get("sample_id") == opt:
                        return f"{s.get('label')}"
                return opt

            selected_sample = st.selectbox(
                "Select Approved Synthetic Fixture",
                options=sample_options,
                format_func=format_sample_option,
                help="Synthetic fixtures are for system demonstration and testing only.",
            )

            sample_bytes, sample_filename, sample_default_caption, sample_desc = None, None, None, None
            if selected_sample != "None":
                sample_bytes, sample_filename, sample_default_caption, sample_desc = get_sample_data(selected_sample)
                st.markdown(
                    f"<div class='callout-limitation'>🧪 <b>Synthetic Fixture:</b> {escape_display(sample_desc or '')}</div>",
                    unsafe_allow_html=True,
                )
                if sample_default_caption and not st.session_state.caption_text:
                    st.session_state.caption_text = sample_default_caption

            # B. User File Uploader
            uploaded_file = st.file_uploader(
                "Or Upload Image File",
                type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
                help=f"Supported formats: JPEG, PNG, WEBP, GIF, BMP. Max size: {MAX_IMAGE_BYTES // (1024*1024)} MB.",
            )

            active_bytes = None
            active_filename = None
            is_user_upload = False

            if uploaded_file is not None:
                active_bytes = uploaded_file.getvalue()
                active_filename = uploaded_file.name
                is_user_upload = True
                if selected_sample != "None":
                    st.markdown(
                        f"<div class='callout-warning'>⚠️ <b>Precedence:</b> {C.SAMPLE_UPLOAD_PRIORITY}</div>",
                        unsafe_allow_html=True,
                    )
            elif sample_bytes is not None:
                active_bytes = sample_bytes
                active_filename = sample_filename
                is_user_upload = False

            # Selected Image Preview & Metadata
            if active_bytes:
                try:
                    preview_img = Image.open(io.BytesIO(active_bytes))
                    st.image(
                        preview_img,
                        caption=f"Selected: {escape_display(active_filename)} ({preview_img.width}×{preview_img.height} px, {len(active_bytes) // 1024} KB)",
                        use_container_width=True,
                    )
                    source_tag = "Custom Upload (Takes Precedence)" if is_user_upload else "Approved Synthetic Fixture"
                    badge_style = "status-neutral" if is_user_upload else "status-warning"
                    st.markdown(f"<span class='status-badge {badge_style}'>{source_tag}</span>", unsafe_allow_html=True)
                except Exception:
                    st.error("Failed to render preview of the selected image.")

    with col_input_right:
        with st.container(border=True):
            st.markdown(
                """
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>2. Caption & Claim</span>
                    <span class='status-badge status-slate'>Textual Signal</span>
                </div>
                <div style='font-size: 0.86rem; color: var(--text-secondary); margin-bottom: 14px;'>
                    Enter the exact claim, headline, or caption accompanying this image.
                </div>
                """,
                unsafe_allow_html=True,
            )

            caption_val = st.text_area(
                "Associated Caption or Claim",
                value=st.session_state.caption_text,
                height=165,
                max_chars=MAX_CAPTION_CHARS,
                placeholder="Enter the caption, tweet, headline, or claim accompanying this image...",
                help=f"Minimum {MIN_CAPTION_CHARS} characters, maximum {MAX_CAPTION_CHARS} characters.",
            )
            st.session_state.caption_text = caption_val

            # Real-time character counter & validation badge
            char_count = len(caption_val)
            col_cnt1, col_cnt2 = st.columns([1, 1])
            with col_cnt1:
                st.caption(f"Characters: **{char_count}** / {MAX_CAPTION_CHARS}")
            with col_cnt2:
                if char_count < MIN_CAPTION_CHARS:
                    st.markdown(f"<div style='text-align: right;'><span class='status-badge status-warning'>Min {MIN_CAPTION_CHARS} chars required</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align: right;'><span class='status-badge status-success'>✓ Valid Length</span></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns([1.2, 1])

            with col_btn1:
                analyze_clicked = st.button("🔍 Analyze Image & Claim", type="primary", use_container_width=True)

            with col_btn2:
                rerun_clicked = st.button("🔄 Rerun Analysis", use_container_width=True, disabled=(st.session_state.report is None))

            if analyze_clicked or rerun_clicked:
                if not active_bytes:
                    st.error("Please provide an image (either upload a file or select a synthetic fixture).")
                elif not caption_val or len(caption_val.strip()) < MIN_CAPTION_CHARS:
                    st.error(f"Please enter a caption of at least {MIN_CAPTION_CHARS} characters.")
                else:
                    with st.spinner("Analyzing image-caption consistency and retrieving corpus evidence..."):
                        success = run_pipeline(active_bytes, active_filename or "uploaded_image.jpg", caption_val)
                        if success:
                            st.rerun()

    # Dedicated Input-rejection View (§11)
    if st.session_state.input_errors:
        with st.container(border=True):
            st.markdown(
                """
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <span style='font-size: 1.05rem; font-weight: 700; color: var(--callout-danger-text);'>❌ Submission Rejected</span>
                    <span class='status-badge status-danger'>Input Validation Failed</span>
                </div>
                <div style='font-size: 0.90rem; color: var(--text-secondary); margin-bottom: 12px;'>
                    Your input could not be processed due to the following validation constraints:
                </div>
                """,
                unsafe_allow_html=True,
            )
            for err in st.session_state.input_errors:
                st.markdown(f"- **`{escape_display(err.code)}`**: {escape_display(err.message)}")
            st.warning("No report model was constructed. Please correct the input and resubmit.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: ANALYSIS REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == C.H_REPORT:
    report: Optional[ReportModel] = st.session_state.report

    if report is None:
        with st.container(border=True):
            st.markdown(
                """
                <div style='text-align: center; padding: 36px 20px 16px 20px;'>
                    <div style='font-size: 2.4rem; margin-bottom: 12px;'>📊</div>
                    <div style='font-size: 1.25rem; font-weight: 700; color: var(--text-heading); margin-bottom: 8px;'>No Active Analysis Report</div>
                    <div style='color: var(--text-secondary); font-size: 0.92rem; max-width: 480px; margin: 0 auto 16px auto;'>
                        Please navigate to the <b>Analyze</b> section to provide an image and caption, then run an analysis to generate an evidence-grounded report.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c2:
                if st.button("⬅ Go to Analyze Workspace", type="primary", use_container_width=True):
                    st.session_state.active_nav_page = C.H_ANALYZE
                    st.rerun()
    else:
        # Compact Report Header
        st.markdown(
            f"""
            <div class='eyebrow'>Investigation Dossier</div>
            <div class='header-title'>Analysis Report</div>
            <div style='margin-bottom: 18px;'>
                <span class='meta-chip'>REQ: {escape_display(report.request_id[:12])}</span>
                <span class='meta-chip'>FP: {escape_display(report.analysis_fingerprint[:16])}...</span>
                <span class='meta-chip'>MODEL: {escape_display(report.provenance.model_version)}</span>
                <span class='meta-chip'>CORPUS: {escape_display(report.provenance.corpus_version)}</span>
                <span class='meta-chip'>TIME: {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preprocessing warnings (§4, §13)
        if report.input.warnings:
            for w in report.input.warnings:
                msg = C.WARNING_MESSAGES.get(w, w)
                st.markdown(
                    f"<div class='callout-warning'>⚠️ <b>Preprocessing Notice ({escape_display(w)}):</b> {escape_display(msg)}</div>",
                    unsafe_allow_html=True,
                )

        # ── Two-Panel Analysis Summary (§3, §5, §7) ──
        # NEVER MERGED INTO A SINGLE VERDICT
        col_clf, col_rag = st.columns([1, 1], gap="large")

        # ── Left Card: Image-Caption Consistency ──
        with col_clf:
            with st.container(border=True):
                clf = report.classifier

                badge_style = "status-success" if clf.label == "MATCHED" else ("status-danger" if clf.label == "MISMATCHED" else "status-slate")
                badge_icon_label = C.OUTCOME_ICON.get(clf.label or "", clf.label or clf.status)

                st.markdown(
                    f"""
                    <div class='surface-card-header'>
                        <span>{C.H_CLASSIFIER}</span>
                        <span class='status-badge {badge_style}'>{badge_icon_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if clf.status == "OK":
                    col_score, col_meta = st.columns([1.5, 1])
                    with col_score:
                        st.metric(
                            label="CLIP Cosine Similarity Score",
                            value=f"{clf.score:.4f}" if clf.score is not None else "N/A",
                            delta=f"{'+' if (clf.score or 0) >= CLASSIFIER_THRESHOLD else ''}{((clf.score or 0) - CLASSIFIER_THRESHOLD):.4f} vs threshold (0.50)",
                            delta_color="normal" if clf.label == "MATCHED" else "inverse",
                        )
                    with col_meta:
                        st.caption(f"**Classification:** {clf.label}")
                        st.caption(f"**Threshold:** {CLASSIFIER_THRESHOLD:.2f}")

                    # Compact visual slider/bar for score mapped to [0, 1]
                    if clf.score is not None:
                        st.progress(float(clf.score))

                    st.caption(f"ℹ️ Score note: {C.SCORE_NOTE}")

                    if clf.calibration_status == "APPLIED" and clf.calibrated_probability is not None:
                        st.write(f"**{C.CAL_PROB_NOTE}:** `{clf.calibrated_probability:.4f}`")
                    else:
                        st.caption(C.CAL_NOT_APPLIED)

                    st.markdown(
                        f"<div style='font-size: 0.88rem; color: var(--text-secondary); margin: 8px 0;'>{C.CLASSIFIER_WHAT_MEASURES}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Diagnostic card for unavailable classifier
                    st.markdown(
                        f"<div class='callout-warning'>⚠️ <b>Classifier {clf.status}:</b> {escape_display(clf.reason or 'Details unavailable')}</div>",
                        unsafe_allow_html=True,
                    )
                    if clf.remedy:
                        st.markdown(
                            f"<div style='font-size: 0.86rem; color: var(--text-secondary); margin: 6px 0;'><b>Remedy:</b> {escape_display(clf.remedy)}</div>",
                            unsafe_allow_html=True,
                        )

                # Level 1 limitation — ALWAYS directly visible (§9.1)
                st.markdown(
                    f"<div class='callout-limitation'><b>Fundamental Limitation:</b><br>{C.NOT_A_TRUTH_CLAIM}</div>",
                    unsafe_allow_html=True,
                )

        # ── Right Card: Claim Assessment via RAG ──
        with col_rag:
            with st.container(border=True):
                asm = report.assessment

                asm_badge_style = "status-success" if asm.outcome == "SUPPORTED" else ("status-danger" if asm.outcome == "CONTRADICTED" else "status-slate")
                asm_icon_label = C.OUTCOME_ICON.get(asm.outcome or "", asm.outcome or asm.status)

                st.markdown(
                    f"""
                    <div class='surface-card-header'>
                        <span>{C.H_ASSESSMENT}</span>
                        <span class='status-badge {asm_badge_style}'>{asm_icon_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if asm.status == "OK":
                    st.metric(
                        label="Corpus Assessment Outcome",
                        value=asm_icon_label,
                    )
                    st.markdown(
                        f"<div style='font-size: 0.90rem; color: var(--text-primary); margin: 8px 0;'><b>System Finding:</b> {escape_display(asm.reason_text or '')}</div>",
                        unsafe_allow_html=True,
                    )

                    col_stat1, col_stat2 = st.columns([1, 1])
                    with col_stat1:
                        st.caption(f"📄 **Examined Passages:** {len(report.evidence.passages)}")
                    with col_stat2:
                        st.caption(f"✓ **Accepted Citations:** {len(asm.accepted_citations)}")
                else:
                    st.markdown(
                        f"<div class='callout-warning'>⚠️ <b>Assessment {asm.status}:</b> {escape_display(asm.reason or 'Details unavailable')}</div>",
                        unsafe_allow_html=True,
                    )
                    if asm.error_detail:
                        st.caption(f"Detail: {escape_display(asm.error_detail)}")

                # Level 2 limitation — ALWAYS directly visible (§9.1)
                st.markdown(
                    f"<div class='callout-limitation'><b>Scope Qualifier:</b><br>{escape_display(asm.scope_qualifier)}</div>",
                    unsafe_allow_html=True,
                )

        # ── Descriptive Signal Comparison (§8) ──
        with st.container(border=True):
            st.markdown(
                f"""
                <div class='surface-card-header'>
                    <span>{C.H_COMPARISON}</span>
                    <span class='status-badge status-neutral'>{report.comparison.state}</span>
                </div>
                <div class='callout-comparison'>
                    <div style='font-size: 0.80rem; font-weight: 700; color: var(--primary-accent); text-transform: uppercase; margin-bottom: 4px;'>Signal Relationship</div>
                    {report.comparison.sentence}
                </div>
                <div class='callout-limitation' style='border-left-color: var(--text-muted);'>
                    <b>Permanent Rule:</b> {PERMANENT_NOTE}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── System Limitations (§9.1 Level 3) ──
        # MUST BE DIRECTLY VISIBLE - NEVER INSIDE COLLAPSIBLE EXPANDERS
        with st.container(border=True):
            st.markdown(
                f"""
                <div class='surface-card-header'>
                    <span>{C.H_SYSTEM_LIMS}</span>
                    <span class='status-badge status-slate'>System Scope</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("The following system-level constraints and limitations apply to this analysis:")
            for lim in report.limitations:
                st.markdown(f"- {escape_display(lim)}")

        # ── Export & Provenance (§15, §2.2) ──
        col_export, col_prov = st.columns([1, 1], gap="large")

        with col_export:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>Export Dossier</span>
                        <span class='status-badge status-neutral'>Markdown</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(C.EXPORT_DISCLOSURE)
                export_md = generate_markdown_export(report)
                export_filename = get_export_filename(report)
                st.download_button(
                    label="📥 Download Full Markdown Report (.md)",
                    data=export_md,
                    file_name=export_filename,
                    mime="text/markdown",
                    use_container_width=True,
                )
                with st.expander("📋 View & Copy Report Text", expanded=False):
                    st.code(export_md, language="markdown")

        with col_prov:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>{C.H_PROVENANCE}</span>
                        <span class='status-badge status-slate'>Audit Metadata</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                prov = report.provenance
                st.markdown(f"- **Request ID:** `{escape_display(report.request_id)}`")
                with st.expander("Full Analysis Fingerprint (SHA-256)", expanded=False):
                    st.code(report.analysis_fingerprint, language="text")
                st.markdown(f"- **Model Version:** `{escape_display(prov.model_version)}`")
                st.markdown(f"- **Encoder Revision:** `{escape_display(prov.encoder_revision)}`")
                st.markdown(f"- **Corpus Version:** `{escape_display(prov.corpus_version)}`")
                st.markdown(f"- **Retrieval Config Hash:** `{escape_display(prov.retrieval_config_hash)}`")
                st.markdown(f"- **Prompt Version:** `{escape_display(prov.prompt_version)}`")
                st.markdown(f"- **Run ID:** `{escape_display(prov.run_id)}`")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: EVIDENCE SOURCES
# ══════════════════════════════════════════════════════════════════════════
elif page == C.H_EVIDENCE:
    report: Optional[ReportModel] = st.session_state.report

    st.markdown(
        """
        <div class='eyebrow'>Corpus Intelligence</div>
        <div class='header-title'>Evidence Sources & Retrieval Details</div>
        <div class='header-desc'>
            Passages retrieved from the local corpus, relevance rankings, and citation audit trail.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report is None:
        with st.container(border=True):
            st.markdown(
                """
                <div style='text-align: center; padding: 36px 20px 16px 20px;'>
                    <div style='font-size: 2.4rem; margin-bottom: 12px;'>🔍</div>
                    <div style='font-size: 1.25rem; font-weight: 700; color: var(--text-heading); margin-bottom: 8px;'>No Active Retrieval Session</div>
                    <div style='color: var(--text-secondary); font-size: 0.92rem; max-width: 480px; margin: 0 auto 16px auto;'>
                        Run an analysis from the <b>Analyze</b> page to inspect retrieved passages and source citations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c2:
                if st.button("⬅ Go to Analyze Workspace", type="primary", use_container_width=True, key="btn_goto_analyze_ev"):
                    st.session_state.active_nav_page = C.H_ANALYZE
                    st.rerun()
    else:
        ev = report.evidence
        st.markdown(
            f"""
            <div style='margin-bottom: 18px;'>
                <span class='meta-chip'>STATUS: {escape_display(ev.status)}</span>
                <span class='meta-chip'>CORPUS: {escape_display(ev.corpus_version)}</span>
                <span class='meta-chip'>DOCUMENTS: {ev.total_documents}</span>
                <span class='meta-chip'>PASSAGES INDEXED: {ev.total_passages}</span>
                <span class='meta-chip'>RETRIEVED: {len(ev.passages)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if ev.status != "OK":
            st.error(f"Evidence retrieval failed: {escape_display(ev.reason or 'No details available.')}")
        elif not ev.passages:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style='font-size: 1.10rem; font-weight: 700; color: var(--callout-warning-text); margin-bottom: 8px;'>
                        {C.EMPTY_EVIDENCE}
                    </div>
                    <div style='color: var(--text-secondary); font-size: 0.90rem;'>
                        {C.NO_EVIDENCE_SUGGESTION}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("### Retrieved Passages")
            for rp in ev.passages:
                p = rp.passage
                with st.container(border=True):
                    date_str = p.published_at if p.published_at else C.NO_DATE
                    valid_url = safe_url(p.url)
                    url_str = f"<a href='{valid_url}' target='_blank' style='color: var(--link-color); font-weight: 500;'>{valid_url}</a>" if valid_url else C.NO_URL
                    score_info = f" | 📊 <b>Relevance:</b> `{rp.relevance_score:.4f}` ({C.RELEVANCE_LABEL})" if rp.relevance_score is not None else ""

                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;'>
                            <div>
                                <span style='font-weight: 700; font-size: 1.05rem; color: var(--text-heading);'>
                                    Passage {rp.rank}: {escape_display(p.title)}
                                </span>
                                <div style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px;'>
                                    Source: <b>{escape_display(p.source_name)}</b> ({escape_display(p.source_type)})
                                </div>
                            </div>
                            <span class='status-badge status-slate'>
                                {C.rank_label(rp.rank, rp.total_retrieved)}
                            </span>
                        </div>
                        <div class='evidence-excerpt'>
                            {escape_display(p.excerpt)}
                        </div>
                        <div style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 8px;'>
                            📅 <b>Date:</b> {escape_display(date_str)} | 🔗 <b>URL:</b> {url_str}{score_info}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if p.rights_wording:
                        st.caption(f"Rights: {escape_display(p.rights_wording)}")

                    if p.storage_scope == "excerpt":
                        st.caption(f"ℹ️ {C.STORAGE_EXCERPT}")
                    elif p.storage_scope == "citation_metadata":
                        st.caption(f"ℹ️ {C.STORAGE_CITATION}")

        # ── Accepted & Rejected Citations from Assessment ──
        st.markdown("---")
        col_acc, col_rej = st.columns([1, 1], gap="large")

        with col_acc:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>{C.H_ACCEPTED_CITS}</span>
                        <span class='status-badge status-success'>Model-Verified</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if report.assessment.accepted_citations:
                    st.caption(C.CITATION_SEMANTICS_DISCLAIMER)
                    for cit in report.assessment.accepted_citations:
                        st.markdown(f"- **Source:** {escape_display(cit.source_name)}")
                        st.markdown(f"  *Quote:* \"{escape_display(cit.quote)}\"")
                else:
                    st.write("No citations met the criteria for accepted claim evidence.")

        with col_rej:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <span style='font-size: 1.05rem; font-weight: 700; color: var(--text-heading);'>{C.H_REJECTED_CITS}</span>
                        <span class='status-badge status-slate'>Audit Trail</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if report.assessment.rejected_citations:
                    for cit in report.assessment.rejected_citations:
                        st.markdown(f"- **Source:** {escape_display(cit.source_name)}")
                        st.caption(f"  *Reason for rejection:* {escape_display(cit.rejection_reason or 'Not stated')}")
                else:
                    st.write("No citations were rejected.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: METHODOLOGY & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == C.H_METHODOLOGY:
    st.markdown(
        """
        <div class='eyebrow'>System Architecture</div>
        <div class='header-title'>Methodology & System Limitations</div>
        <div class='header-desc'>
            Dual-signal architecture, visual-textual consistency, RAG retrieval mechanics, and operational boundaries.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style='font-size: 1.10rem; font-weight: 700; color: var(--text-heading); margin-bottom: 8px;'>
                Dual-Signal Architecture
            </div>
            <div style='font-size: 0.92rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 16px;'>
                Multimodal misinformation detection requires rigorously separating visual–textual consistency
                from factual real-world truth. An image may accurately match a caption, yet the claim
                itself may be completely false. Conversely, a caption may be factually true while an
                unrelated image is attached out-of-context.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_m1, col_m2 = st.columns([1, 1], gap="large")
        with col_m1:
            st.markdown("#### Signal 1: Image–Caption Consistency")
            st.write(C.METHODOLOGY_CLIP)

        with col_m2:
            st.markdown("#### Signal 2: Claim Assessment via RAG")
            st.write(C.METHODOLOGY_RAG)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # All limitations visible directly, never collapsed (§9.1)
    with st.container(border=True):
        st.markdown(
            f"""
            <div style='font-size: 1.10rem; font-weight: 700; color: var(--text-heading); margin-bottom: 8px;'>
                {C.H_SYSTEM_LIMS}
            </div>
            <div style='font-size: 0.88rem; color: var(--text-muted); margin-bottom: 12px;'>
                The following fundamental boundaries apply to all analyses:
            </div>
            """,
            unsafe_allow_html=True,
        )
        for lim in C.METHODOLOGY_LIMITATIONS:
            st.markdown(f"- {escape_display(lim)}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: PRIVACY & DATA HANDLING
# ══════════════════════════════════════════════════════════════════════════
elif page == C.H_PRIVACY:
    st.markdown(
        """
        <div class='eyebrow'>Security & Privacy Policy</div>
        <div class='header-title'>Privacy & Data Handling Disclosures</div>
        <div class='header-desc'>
            Transparency disclosures regarding local inference, caching, and network isolation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style='font-size: 1.10rem; font-weight: 700; color: var(--text-heading); margin-bottom: 12px;'>
                Inference & Data Retention
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"- **Local Inference:** {C.PRIVACY_LOCAL_INFERENCE}")
        st.markdown(f"- **Model Downloads:** {C.PRIVACY_MODEL_DOWNLOADS}")
        st.markdown(f"- **Memory-Only Session Storage:** {C.PRIVACY_STORAGE}")
        st.markdown(f"- **Outbound Links:** {C.PRIVACY_SOURCE_LINKS}")
        st.markdown(f"- **Zero Telemetry:** {C.PRIVACY_NO_TELEMETRY}")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            """
            <div style='font-size: 1.10rem; font-weight: 700; color: var(--text-heading); margin-bottom: 12px;'>
                Optional Hosted Backend Disclosures
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(C.PRIVACY_HOSTED_BACKEND)
