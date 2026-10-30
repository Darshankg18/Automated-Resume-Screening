"""UI Cards, Badges, and CSS Styling for Streamlit Frontend.

Provides an ultra-stylish, modern executive glassmorphism design system
with Plus Jakarta Sans typography, glowing accents, and smooth micro-interactions.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_styles():
    """Inject polished, modern executive dark-glass styles into Streamlit."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global App Background & Typography */
    .stApp {
        background-color: #07090e;
        background-image: 
            radial-gradient(circle at 12% 16%, rgba(99, 102, 241, 0.14) 0%, transparent 40%),
            radial-gradient(circle at 88% 82%, rgba(168, 85, 247, 0.12) 0%, transparent 42%),
            radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 55%);
        background-attachment: fixed;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #e2e8f0;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    p, span, label {
        color: #cbd5e1;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94a3b8;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(99, 102, 241, 0.12);
        color: #ffffff !important;
    }

    /* Main Hero Header */
    .main-header {
        position: relative;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.15);
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4, #10b981);
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc;
        margin-bottom: 12px;
    }
    .pulsing-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }
    .main-header h1 {
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.03em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 14.5px;
        line-height: 1.5;
        margin: 0;
        max-width: 900px;
    }

    /* Metric Cards */
    .metric-card {
        position: relative;
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        margin-bottom: 14px;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.5), 0 0 18px rgba(99, 102, 241, 0.2);
    }
    .metric-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.8), rgba(168, 85, 247, 0.4));
    }
    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-sub {
        color: #64748b;
        font-size: 12px;
        font-weight: 500;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Score Badges */
    .score-badge-high {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13.5px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .score-badge-mid {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13.5px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
    }
    .score-badge-low {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #fb7185;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 13.5px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(244, 63, 94, 0.2);
    }

    /* Skill Tags */
    .skill-tag-matched {
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid rgba(14, 165, 233, 0.35);
        color: #38bdf8;
        padding: 4px 11px;
        border-radius: 7px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 6px 3px 0;
        display: inline-block;
        transition: all 0.2s ease;
    }
    .skill-tag-matched:hover {
        background: rgba(14, 165, 233, 0.25);
        transform: translateY(-1px);
    }
    .skill-tag-missing {
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: #fb7185;
        padding: 4px 11px;
        border-radius: 7px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 6px 3px 0;
        display: inline-block;
        transition: all 0.2s ease;
    }
    .skill-tag-missing:hover {
        background: rgba(244, 63, 94, 0.22);
        transform: translateY(-1px);
    }

    /* Ethical & Responsible AI Banner */
    .ethical-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 4px solid #6366f1;
        padding: 16px 22px;
        border-radius: 12px;
        margin: 18px 0 24px 0;
        color: #cbd5e1;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
    }
    .ethical-banner b {
        color: #ffffff;
    }

    /* Glass Surface Box */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px 26px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }

    /* Streamlit Interactive Components Overrides */
    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(30, 41, 59, 0.7) !important;
        color: #f8fafc !important;
        padding: 9px 18px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.25) !important;
        background: rgba(45, 55, 72, 0.85) !important;
    }
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.45) !important;
    }
    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        box-shadow: 0 8px 28px rgba(139, 92, 246, 0.6) !important;
        transform: translateY(-2px) scale(1.01);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.65);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-family: 'Plus Jakarta Sans', sans-serif;
        padding: 0 20px;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.3)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(99, 102, 241, 0.45) !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25);
    }

    /* Inputs & Select Boxes */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 10px !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        background-color: rgba(17, 24, 39, 0.75) !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }
    textarea {
        background-color: rgba(17, 24, 39, 0.75) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }

    /* Dataframe Tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }

    /* Native st.metric */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Alert / Callout Overrides */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Render a clean, ultra-stylish glowing executive metric card."""
    sub_html = f'<div class="metric-sub"><span>✦</span> {subtext}</div>' if subtext else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-top-bar"></div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_ethical_disclaimer():
    """Render the standard responsible AI disclaimer with modern styling."""
    st.markdown("""
    <div class="ethical-banner">
        <b>⚖️ Responsible AI Notice:</b> Candidate match scores represent qualification alignment against the active job description keywords and experience criteria. 
        <b>Match scores must never be treated as a probability of hiring success or an automated employment decision.</b> Sensitive personal attributes (gender, age, race, religion) are strictly excluded from ranking features.
    </div>
    """, unsafe_allow_html=True)
