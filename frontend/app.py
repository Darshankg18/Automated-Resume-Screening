"""Streamlit Web Application for AI-Powered Resume Screening and Ranking System with Explainable AI.

Person 4 Deliverable:
- Dashboard
- Job Description Upload
- Multi-Resume Upload (PDF, DOCX, TXT)
- Ranked Candidate Table with search & filtering
- Candidate Details & SHAP Explainability (Waterfall & Bar charts, positive/negative drivers)
- Bias Mitigation, Group Fairness Auditing, Proxy Detection, and Evaluation Metrics
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frontend.components.cards import (
    inject_custom_styles,
    render_ethical_disclaimer,
    render_metric_card,
)
from frontend.components.charts import (
    create_feature_contribution_chart,
    create_group_fairness_bar_chart,
    create_score_distribution_chart,
    create_waterfall_chart,
)

# API Configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & XAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_styles()


# ---------------------------------------------------------
# Helper Functions: API Communication with Local Fallback
# ---------------------------------------------------------
def check_api_health(api_url: str) -> Tuple[bool, Dict[str, Any]]:
    """Check if FastAPI backend is responsive."""
    try:
        res = requests.get(f"{api_url}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, {}


# Initialize Session State
if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "📊 Dashboard"
if "selected_candidate_id" not in st.session_state:
    st.session_state["selected_candidate_id"] = None
if "local_jd" not in st.session_state:
    st.session_state["local_jd"] = None
if "local_resumes" not in st.session_state:
    st.session_state["local_resumes"] = {}
if "local_rankings" not in st.session_state:
    st.session_state["local_rankings"] = []
if "local_eval_dataset" not in st.session_state:
    st.session_state["local_eval_dataset"] = None
if "local_demo_groups" not in st.session_state:
    st.session_state["local_demo_groups"] = {}


# ---------------------------------------------------------
# Sidebar Navigation & System Controls
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 14px 16px; background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8)); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #6366f1, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 18px; box-shadow: 0 0 12px rgba(99, 102, 241, 0.5);">🎯</div>
        <div>
            <h2 style="margin: 0; font-size: 16.5px; color: #ffffff; font-weight: 800; letter-spacing: -0.02em;">AEGIS SCREENER</h2>
            <p style="margin: 1px 0 0 0; font-size: 11px; color: #94a3b8; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;">Explainable AI & Governance</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# API Status Badge
api_online, health_data = check_api_health(st.session_state["api_url"])
if api_online:
    shap_tag = "Ready" if health_data.get("shap_ready") else "Fallback"
    st.sidebar.success(f"🟢 **API Online** (SHAP: {shap_tag})")
else:
    st.sidebar.info("💡 **Local Direct Mode** (FastAPI offline, running via direct engine)")

# Navigation Menu
pages = [
    "📊 Dashboard",
    "📄 Upload Job Description",
    "📁 Upload Resumes",
    "🏆 Ranked Candidates",
    "🔍 Candidate Explanation (SHAP)",
    "⚖️ Fairness & Evaluation",
]

selected_page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0
)
st.session_state["current_page"] = selected_page

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")

# Preload All Sample Data Button
if st.sidebar.button("📦 Load Sample Data (6 Resumes + JD)", use_container_width=True):
    from backend.services.resume_parser import resume_parser
    from backend.services.ranking_service import ranking_service

    # 1. Load Senior Python JD
    jd_path = os.path.join(REPO_ROOT, "sample_data", "job_descriptions", "senior_python_dev.txt")
    if os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        parsed_jd = ranking_service.parse_job_description(jd_text, jd_id="sample_jd_01")
        st.session_state["local_jd"] = parsed_jd

    # 2. Load 6 Sample Resumes
    resumes_dir = os.path.join(REPO_ROOT, "sample_data", "resumes")
    loaded_count = 0
    if os.path.exists(resumes_dir):
        for fname in os.listdir(resumes_dir):
            if fname.endswith((".txt", ".docx")):
                fpath = os.path.join(resumes_dir, fname)
                with open(fpath, "rb") as f:
                    file_bytes = f.read()
                cand_id = f"cand_{fname[:10]}_{loaded_count+1}"
                try:
                    parsed_cand = resume_parser.parse_resume(file_bytes, fname, cand_id)
                    st.session_state["local_resumes"][cand_id] = parsed_cand
                    loaded_count += 1
                except Exception:
                    pass

    # 3. Compute Rankings
    if st.session_state["local_jd"] and st.session_state["local_resumes"]:
        ranks = ranking_service.rank_candidates(
            list(st.session_state["local_resumes"].values()),
            st.session_state["local_jd"]
        )
        st.session_state["local_rankings"] = ranks

    st.sidebar.success(f"Loaded JD and {loaded_count} sample resumes!")
    st.rerun()

# Clear Session Button
if st.sidebar.button("🔄 Reset All Data", use_container_width=True):
    st.session_state["local_jd"] = None
    st.session_state["local_resumes"] = {}
    st.session_state["local_rankings"] = []
    st.session_state["selected_candidate_id"] = None
    st.sidebar.warning("Session cleared.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity AI • Team Person 4: Explainability, Fairness, API & Frontend")


# ---------------------------------------------------------
# SCREEN 1: DASHBOARD
# ---------------------------------------------------------
if selected_page == "📊 Dashboard":
    st.markdown("""
    <div class="main-header">
        <div class="header-badge">
            <span class="pulsing-dot"></span>
            EXPLAINABLE AI ENGINE • PRODUCTION READY
        </div>
        <h1>AI-Powered Resume Screening & Explainability Studio</h1>
        <p>Calibrated candidate qualification matching powered by <b>SHAP TreeExplainer</b> feature attributions, proactive demographic proxy detection, EEOC four-fifths fairness auditing, and multi-metric ranking validation.</p>
    </div>
    """, unsafe_allow_html=True)

    render_ethical_disclaimer()

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_res = len(st.session_state["local_resumes"])
        render_metric_card("Total Resumes Uploaded", f"{total_res}", "Active in session")
    with c2:
        jd_status = "Active" if st.session_state["local_jd"] else "Not Set"
        render_metric_card("Job Description Status", jd_status, st.session_state["local_jd"]["title"] if st.session_state["local_jd"] else "Upload a JD to begin")
    with c3:
        total_ranked = len(st.session_state["local_rankings"])
        render_metric_card("Candidates Processed", f"{total_ranked}", "Scored with ML model")
    with c4:
        top_score = f"{st.session_state['local_rankings'][0]['match_score']:.1f}/100" if st.session_state["local_rankings"] else "N/A"
        render_metric_card("Highest Match Score", top_score, "Top candidate alignment")

    st.markdown("### 📈 Screening Overview & Distribution")

    col_chart, col_summary = st.columns([3, 2])
    with col_chart:
        if st.session_state["local_rankings"]:
            scores = [r["match_score"] for r in st.session_state["local_rankings"]]
            fig = create_score_distribution_chart(scores)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No candidates ranked yet. Navigate to **Upload Job Description** and **Upload Resumes** or click **'Load Sample Data'** in the sidebar.")

    with col_summary:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #ffffff; font-size: 15px; font-weight: 700;">🏛️ Modular Pipeline Architecture</h4>
                <span style="font-size: 11px; background: rgba(99, 102, 241, 0.2); color: #a5b4fc; padding: 2px 8px; border-radius: 9999px; font-weight: 700;">PERSON 1-4</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="padding: 9px 12px; background: rgba(30, 41, 59, 0.45); border-radius: 9px; border-left: 3px solid #38bdf8;">
                    <b style="color: #38bdf8; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;">Person 1 • Parser Engine</b><br>
                    <span style="font-size: 12px; color: #cbd5e1;">Multi-format text extraction (PDF, DOCX, TXT) and skill taxonomy identification.</span>
                </div>
                <div style="padding: 9px 12px; background: rgba(30, 41, 59, 0.45); border-radius: 9px; border-left: 3px solid #818cf8;">
                    <b style="color: #818cf8; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;">Person 2 • NLP & Features</b><br>
                    <span style="font-size: 12px; color: #cbd5e1;">TF-IDF cosine similarity, keyword overlap, experience calibration.</span>
                </div>
                <div style="padding: 9px 12px; background: rgba(30, 41, 59, 0.45); border-radius: 9px; border-left: 3px solid #c084fc;">
                    <b style="color: #c084fc; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;">Person 3 • Ranking Model</b><br>
                    <span style="font-size: 12px; color: #cbd5e1;">Calibrated Random Forest regression model predicting 0-100 match score.</span>
                </div>
                <div style="padding: 9px 12px; background: rgba(30, 41, 59, 0.45); border-radius: 9px; border-left: 3px solid #34d399;">
                    <b style="color: #34d399; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em;">Person 4 • XAI & Governance</b><br>
                    <span style="font-size: 12px; color: #cbd5e1;">SHAP TreeExplainer, labeled fallbacks, EEOC 80% audit & Streamlit dashboard.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Fast Track Workflow")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-top: 12px;">
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="width: 24px; height: 24px; border-radius: 50%; background: #6366f1; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">1</span>
                <b style="color: #ffffff; font-size: 14px;">Upload JD</b>
            </div>
            <span style="font-size: 12.5px; color: #94a3b8;">Provide job specifications or pick a preset role to set the baseline criteria.</span>
        </div>
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="width: 24px; height: 24px; border-radius: 50%; background: #8b5cf6; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">2</span>
                <b style="color: #ffffff; font-size: 14px;">Upload Resumes</b>
            </div>
            <span style="font-size: 12.5px; color: #94a3b8;">Batch ingest candidate resumes (PDF, DOCX, TXT) with automatic extraction.</span>
        </div>
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="width: 24px; height: 24px; border-radius: 50%; background: #ec4899; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">3</span>
                <b style="color: #ffffff; font-size: 14px;">Rank Candidates</b>
            </div>
            <span style="font-size: 12.5px; color: #94a3b8;">Run feature extraction and ML ranking model to calculate match scores.</span>
        </div>
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="width: 24px; height: 24px; border-radius: 50%; background: #06b6d4; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">4</span>
                <b style="color: #ffffff; font-size: 14px;">Inspect SHAP</b>
            </div>
            <span style="font-size: 12.5px; color: #94a3b8;">Analyze individual score waterfalls, positive drivers, and missing skills.</span>
        </div>
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="width: 24px; height: 24px; border-radius: 50%; background: #10b981; color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700;">5</span>
                <b style="color: #ffffff; font-size: 14px;">Audit Fairness</b>
            </div>
            <span style="font-size: 12.5px; color: #94a3b8;">Audit EEOC 80% rule, check proxy markers, and validate test benchmarks.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# SCREEN 2: UPLOAD JOB DESCRIPTION
# ---------------------------------------------------------
elif selected_page == "📄 Upload Job Description":
    st.markdown("## 📄 Job Description Upload & Review")
    st.markdown("Provide the target job description to establish the qualification baseline.")

    from backend.services.ranking_service import ranking_service

    col_upload, col_sample = st.columns([3, 2])

    with col_upload:
        uploaded_jd = st.file_uploader("Upload Job Description (PDF or TXT)", type=["txt", "pdf"])

        if uploaded_jd:
            from backend.services.resume_parser import resume_parser
            file_bytes = uploaded_jd.read()
            try:
                raw_text = resume_parser.extract_text_from_bytes(file_bytes, uploaded_jd.name)
                cleaned_text = resume_parser.clean_text(raw_text)
                parsed_jd = ranking_service.parse_job_description(cleaned_text, jd_id="uploaded_jd")
                st.session_state["local_jd"] = parsed_jd
                st.session_state["local_rankings"] = []  # Reset old rankings
                st.success(f"✅ Successfully processed '{uploaded_jd.name}'!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    with col_sample:
        st.markdown("#### ⚡ Or Load a Pre-configured Sample JD")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("Senior Python Dev", use_container_width=True):
                path = os.path.join(REPO_ROOT, "sample_data", "job_descriptions", "senior_python_dev.txt")
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                st.session_state["local_jd"] = ranking_service.parse_job_description(text, jd_id="sample_senior_python")
                st.session_state["local_rankings"] = []
                st.success("Loaded Senior Python Dev JD!")
                st.rerun()

        with c_btn2:
            if st.button("Lead Data Scientist", use_container_width=True):
                path = os.path.join(REPO_ROOT, "sample_data", "job_descriptions", "data_scientist.txt")
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                st.session_state["local_jd"] = ranking_service.parse_job_description(text, jd_id="sample_data_scientist")
                st.session_state["local_rankings"] = []
                st.success("Loaded Lead Data Scientist JD!")
                st.rerun()

    st.markdown("---")

    if st.session_state["local_jd"]:
        jd = st.session_state["local_jd"]
        st.markdown(f"### 📋 Active Job Description: **{jd['title']}**")

        c_info1, c_info2, c_info3 = st.columns(3)
        c_info1.metric("Word Count", jd["word_count"])
        c_info2.metric("Extracted Required Skills", len(jd["required_skills"]))
        c_info3.metric("Expected Experience", f"{jd['expected_years']} Years")

        st.markdown("#### 🔑 Extracted Key Skills:")
        if jd["required_skills"]:
            skills_html = "".join([f'<span class="skill-tag-matched">{s}</span>' for s in jd["required_skills"]])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.info("No specific technical skills matched from taxonomy; ranking will utilize TF-IDF and keyword overlap.")

        st.markdown("#### 📝 Full Job Description Text (Review & Edit):")
        edited_text = st.text_area("Extracted Job Description Text", value=jd["raw_text"], height=280)
        if st.button("Update Job Description Text"):
            st.session_state["local_jd"] = ranking_service.parse_job_description(edited_text, jd_id=jd["jd_id"])
            st.session_state["local_rankings"] = []
            st.success("Job description updated!")
            st.rerun()
    else:
        st.info("No job description is currently active. Upload a file above or select a sample JD.")


# ---------------------------------------------------------
# SCREEN 3: UPLOAD RESUMES
# ---------------------------------------------------------
elif selected_page == "📁 Upload Resumes":
    st.markdown("## 📁 Upload Candidate Resumes")
    st.markdown("Upload multiple candidate resumes in **PDF**, **DOCX**, or **TXT** format.")

    from backend.services.resume_parser import resume_parser
    from backend.services.ranking_service import ranking_service

    uploaded_files = st.file_uploader(
        "Select Resumes (Multiple files supported)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        success_count = 0
        for uf in uploaded_files:
            try:
                file_bytes = uf.read()
                cand_id = f"cand_{uf.name.replace(' ', '_')}"
                if cand_id not in st.session_state["local_resumes"]:
                    parsed = resume_parser.parse_resume(file_bytes, uf.name, cand_id)
                    st.session_state["local_resumes"][cand_id] = parsed
                    success_count += 1
            except Exception as e:
                st.error(f"Error parsing '{uf.name}': {e}")

        if success_count > 0:
            st.success(f"Processed {success_count} new resume(s)!")

    # Current Resumes Table
    st.markdown("---")
    st.markdown(f"### 📋 Uploaded Resumes in Session ({len(st.session_state['local_resumes'])})")

    if st.session_state["local_resumes"]:
        res_data = []
        for cid, r in st.session_state["local_resumes"].items():
            res_data.append({
                "Candidate ID": cid,
                "Candidate Name": r.get("candidate_name", "Candidate"),
                "Filename": r.get("filename", "unknown"),
                "Experience": f"{r.get('experience_years', 1.0)} yrs",
                "Degree": r.get("education", "Bachelor's"),
                "Extracted Skills Count": len(r.get("skills", [])),
                "Sample Skills": ", ".join(r.get("skills", [])[:4]) or "None"
            })
        st.dataframe(pd.DataFrame(res_data), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Rank Candidates Against Active Job Description", type="primary", use_container_width=True):
            if not st.session_state["local_jd"]:
                st.error("⚠️ Please upload or set a Job Description first in Screen 2.")
            else:
                with st.spinner("Executing feature engineering and ML ranking pipeline..."):
                    ranks = ranking_service.rank_candidates(
                        list(st.session_state["local_resumes"].values()),
                        st.session_state["local_jd"]
                    )
                    st.session_state["local_rankings"] = ranks
                    st.session_state["current_page"] = "🏆 Ranked Candidates"
                    st.success("Ranking complete!")
                    st.rerun()
    else:
        st.info("No resumes uploaded yet. Upload files above or click 'Load Sample Data' in the sidebar.")


# ---------------------------------------------------------
# SCREEN 4: RANKED CANDIDATE TABLE
# ---------------------------------------------------------
elif selected_page == "🏆 Ranked Candidates":
    st.markdown("## 🏆 Ranked Candidate Results")
    render_ethical_disclaimer()

    if not st.session_state["local_rankings"]:
        st.warning("⚠️ No ranked candidate results available. Please ensure you have uploaded a Job Description and Resumes, then click 'Rank Candidates'.")
    else:
        rankings = st.session_state["local_rankings"]
        jd = st.session_state["local_jd"]

        st.markdown(f"**Target Role:** {jd['title'] if jd else 'Role'} • **Total Evaluated:** {len(rankings)} candidates")

        # Filtering and Search Controls
        c_filter1, c_filter2 = st.columns([2, 1])
        with c_filter1:
            search_query = st.text_input("🔍 Search by Candidate Name, ID, or Skill:", placeholder="e.g. Alice, Python, SQL")
        with c_filter2:
            min_score = st.slider("Filter by Minimum Score:", min_value=0.0, max_value=100.0, value=0.0, step=5.0)

        # Apply filters
        filtered_rankings = []
        for r in rankings:
            if r["match_score"] < min_score:
                continue
            if search_query:
                q = search_query.lower()
                cand_name = r.get("candidate_name", "").lower()
                cand_id = r.get("candidate_id", "").lower()
                skills = " ".join(r.get("matched_skills", [])).lower()
                if q not in cand_name and q not in cand_id and q not in skills:
                    continue
            filtered_rankings.append(r)

        # Table Display
        display_data = []
        for r in filtered_rankings:
            display_data.append({
                "Rank": f"#{r['rank']}",
                "Candidate ID": r["candidate_id"],
                "Candidate Name": r.get("candidate_name", "Candidate"),
                "Match Score": f"{r['match_score']:.1f} / 100",
                "Matched Skills": ", ".join(r["matched_skills"][:4]) or "None",
                "Missing Skills": ", ".join(r["missing_skills"][:3]) or "None",
                "Experience": f"{r['experience_years']} yrs",
                "Education": r["education_level"],
                "Explanation": "✅ Available",
            })

        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Select Candidate for Detailed SHAP Explanation")

        candidate_options = {
            f"Rank #{r['rank']} - {r['candidate_name']} ({r['candidate_id']}) — Score: {r['match_score']:.1f}": r["candidate_id"]
            for r in filtered_rankings
        }

        selected_label = st.selectbox("Choose Candidate to Inspect:", list(candidate_options.keys()))
        if st.button("View Candidate Details & SHAP Explanation", type="primary"):
            st.session_state["selected_candidate_id"] = candidate_options[selected_label]
            st.session_state["current_page"] = "🔍 Candidate Explanation (SHAP)"
            st.rerun()


# ---------------------------------------------------------
# SCREEN 5: CANDIDATE DETAILS AND EXPLANATION (SHAP)
# ---------------------------------------------------------
elif selected_page == "🔍 Candidate Explanation (SHAP)":
    st.markdown("## 🔍 Explainable AI: Candidate Match Analysis")

    from backend.services.explainability import explainability_service

    if not st.session_state["local_rankings"]:
        st.warning("⚠️ No ranked candidate data found. Please rank candidates first.")
    else:
        rankings = st.session_state["local_rankings"]
        candidate_ids = [r["candidate_id"] for r in rankings]

        # Candidate dropdown
        curr_id = st.session_state.get("selected_candidate_id") or candidate_ids[0]
        if curr_id not in candidate_ids:
            curr_id = candidate_ids[0]

        cand_choice_labels = {
            r["candidate_id"]: f"Rank #{r['rank']} - {r['candidate_name']} (Score: {r['match_score']:.1f})"
            for r in rankings
        }

        chosen_id = st.selectbox(
            "Select Candidate to Explain:",
            candidate_ids,
            index=candidate_ids.index(curr_id),
            format_func=lambda cid: cand_choice_labels[cid]
        )
        st.session_state["selected_candidate_id"] = chosen_id

        # Toggle for Fallback mode comparison
        c_toggle, c_btn_back = st.columns([3, 1])
        with c_toggle:
            force_fallback = st.checkbox("🧪 Force Transparent Fallback Attribution (Compare against SHAP)", value=False)
        with c_btn_back:
            if st.button("⬅️ Back to Ranked Table"):
                st.session_state["current_page"] = "🏆 Ranked Candidates"
                st.rerun()

        # Compute explanation
        cand_profile = st.session_state["local_resumes"][chosen_id]
        jd_profile = st.session_state["local_jd"]

        explanation = explainability_service.explain_candidate(
            cand_profile, jd_profile, force_fallback=force_fallback
        )

        st.markdown("---")

        # Method indicator banner
        if explanation["is_shap_explanation"]:
            st.success(f"⚡ **Method:** {explanation['explanation_method']} — Genuine Shapley additive values computed from Random Forest scoring model.")
        else:
            st.warning(f"⚠️ **Method:** {explanation['explanation_method']} — Explicit fallback attribution relative to feature baselines.")

        # Top summary cards
        c_score, c_base, c_skills, c_exp = st.columns(4)
        c_score.metric("Overall Match Score", f"{explanation['match_score']:.1f}/100")
        c_base.metric("Expected Base Score", f"{explanation['base_score']:.1f}/100", "Baseline reference")
        c_skills.metric("Matched Skills Count", len(explanation["matched_skills"]), f"{len(explanation['missing_skills'])} missing")
        c_exp.metric("Observed Experience", f"{cand_profile.get('experience_years', 1.0)} Years", f"Target: {jd_profile.get('expected_years', 3.0)} yrs")

        # Human-Readable Explanation Box
        st.markdown("### 💬 Human-Readable Explanation")
        summary_bullets = explanation["human_readable_summary"]
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #8b5cf6; margin-bottom: 24px;">
        """ + "".join([f"<p style='margin: 8px 0; color: #f1f5f9; font-size: 14.5px; line-height: 1.6;'><span style='color: #a78bfa; font-weight: bold;'>▸</span> {b}</p>" for b in summary_bullets]) + "</div>", unsafe_allow_html=True)

        # Matched vs Missing Skills breakdown
        col_matched, col_missing = st.columns(2)
        with col_matched:
            st.markdown("#### ✅ Skills Matched Against Job Description:")
            if explanation["matched_skills"]:
                matched_html = "".join([f'<span class="skill-tag-matched">{s}</span>' for s in explanation["matched_skills"]])
                st.markdown(matched_html, unsafe_allow_html=True)
            else:
                st.write("No matching technical skills identified.")

        with col_missing:
            st.markdown("#### ❌ Missing Required Skills:")
            if explanation["missing_skills"]:
                missing_html = "".join([f'<span class="skill-tag-missing">{s}</span>' for s in explanation["missing_skills"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.write("None! All listed required skills were matched.")

        st.markdown("---")
        st.markdown("### 📊 Feature Contribution Visualizations")

        tab_waterfall, tab_bar, tab_table = st.tabs(["Waterfall Chart", "Horizontal Contribution Bar Chart", "Detailed Feature Table"])

        c_data = explanation["chart_data"]

        with tab_waterfall:
            fig_wf = create_waterfall_chart(
                feature_names=c_data["features"],
                contributions=c_data["contributions"],
                base_score=c_data["base_score"],
                final_score=c_data["final_score"],
                method_title=f"Score Waterfall Decomposition ({'SHAP' if explanation['is_shap_explanation'] else 'Fallback'})"
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with tab_bar:
            fig_bar = create_feature_contribution_chart(
                feature_names=c_data["features"],
                contributions=c_data["contributions"],
                base_score=c_data["base_score"],
                final_score=c_data["final_score"],
                method_title=f"Feature Attributions (Positive vs. Negative Drivers)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_table:
            contrib_rows = []
            for c in explanation["feature_contributions"]:
                contrib_rows.append({
                    "Feature": c["display_name"],
                    "Observed Value": c["value"],
                    "Score Impact (Pts)": f"{c['contribution']:+.2f}",
                    "Impact Direction": c["impact"].title(),
                    "Explanation Description": c["description"]
                })
            st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True)

        # Positive & Negative Drivers
        st.markdown("---")
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("#### 🟢 Positive Score Drivers:")
            if explanation["positive_contributions"]:
                for p in explanation["positive_contributions"]:
                    st.write(f"• {p}")
            else:
                st.write("No substantial positive drivers detected.")

        with col_neg:
            st.markdown("#### 🔴 Negative Score Drivers:")
            if explanation["negative_contributions"]:
                for n in explanation["negative_contributions"]:
                    st.write(f"• {n}")
            else:
                st.write("No substantial penalties detected.")


# ---------------------------------------------------------
# SCREEN 6: FAIRNESS AND EVALUATION
# ---------------------------------------------------------
elif selected_page == "⚖️ Fairness & Evaluation":
    st.markdown("## ⚖️ Bias Mitigation, Group Fairness & Model Evaluation")
    render_ethical_disclaimer()

    from backend.services.fairness import fairness_service
    from backend.services.evaluation import evaluation_service

    tab_fairness, tab_eval, tab_limitations = st.tabs([
        "Group Fairness & Proxy Audit",
        "Model Evaluation Metrics",
        "Responsible AI Limitations"
    ])

    with tab_fairness:
        st.markdown("### 🛡️ Demographic Parity & Disparate Impact Auditing")
        st.write(
            "Under regulatory frameworks (e.g. EEOC 80% rule), group fairness metrics evaluate whether selection rates "
            "are equitable across protected cohorts. **Demographic data is never inferred from candidate names or photos.**"
        )

        cutoff = st.slider("Selection Score Cutoff Threshold:", min_value=50.0, max_value=90.0, value=70.0, step=5.0)

        # Check if demographic evaluation data is present
        demo_available = len(st.session_state["local_demo_groups"]) > 0

        c_load_demo, _ = st.columns([2, 3])
        with c_load_demo:
            if not demo_available:
                if st.button("📊 Load Lawful Demographic Benchmark Cohort for Audit Demo"):
                    benchmark_path = os.path.join(REPO_ROOT, "sample_data", "benchmark_eval_data.json")
                    with open(benchmark_path, "r", encoding="utf-8") as f:
                        bdata = json.load(f)
                    demo_map = {c["candidate_id"]: c["demographic_group"] for c in bdata["candidates"]}
                    st.session_state["local_demo_groups"] = demo_map
                    st.session_state["local_eval_dataset"] = bdata
                    st.success("Loaded demographic evaluation cohort!")
                    st.rerun()

        # Build candidate score list for fairness check
        if demo_available and st.session_state["local_eval_dataset"]:
            cand_scores = [
                {"candidate_id": c["candidate_id"], "match_score": c["predicted_score"]}
                for c in st.session_state["local_eval_dataset"]["candidates"]
            ]
            demo_data = st.session_state["local_demo_groups"]
        elif st.session_state["local_rankings"]:
            cand_scores = [
                {"candidate_id": r["candidate_id"], "match_score": r["match_score"]}
                for r in st.session_state["local_rankings"]
            ]
            demo_data = st.session_state["local_demo_groups"]
        else:
            cand_scores = []
            demo_data = {}

        fairness_report = fairness_service.evaluate_group_fairness(
            candidate_scores=cand_scores,
            demographic_data=demo_data,
            selection_threshold=cutoff
        )

        if not fairness_report["evaluation_data_available"]:
            st.info(f"ℹ️ **Status: {fairness_report['status']}**")
            st.write(fairness_report["note"])
            for w in fairness_report["warnings"]:
                st.warning(f"⚠️ {w}")
        else:
            st.success(f"✅ **Status:** {fairness_report['status']} (Evaluated on {len(cand_scores)} candidates)")

            # Metric cards
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                disp_impact = fairness_report["disparate_impact_ratio"]
                passed = fairness_report["four_fifths_rule_passed"]
                status_str = "PASS (≥ 80%)" if passed else "AUDIT NEEDED (< 80%)"
                render_metric_card("Disparate Impact Ratio", f"{disp_impact:.2f}", status_str)

            with c_f2:
                parity_diff = fairness_report["demographic_parity_difference"]
                render_metric_card("Demographic Parity Diff", f"{parity_diff:.2%}", "Spread between highest and lowest rates")

            with c_f3:
                render_metric_card("Cutoff Threshold", f"{cutoff} pts", "Minimum score for selection")

            # Chart
            grp_names = [g["group_name"] for g in fairness_report["group_metrics"]]
            sel_rates = [g["selection_rate"] for g in fairness_report["group_metrics"]]
            fig_grp = create_group_fairness_bar_chart(grp_names, sel_rates)
            st.plotly_chart(fig_grp, use_container_width=True)

            # Table of groups
            st.markdown("#### Cohort Selection Rates Table:")
            st.dataframe(pd.DataFrame(fairness_report["group_metrics"]), use_container_width=True)

        # Proxy Detection Audit on Resumes
        st.markdown("---")
        st.markdown("### 🔎 Resume Proxy Feature Audit")
        st.write("Scans uploaded resumes for indirect demographic proxies (e.g., graduation year markers, affinity clubs):")

        flagged_count = 0
        proxy_rows = []
        for cid, cand in st.session_state["local_resumes"].items():
            audit = fairness_service.detect_proxy_features(cand["raw_text"])
            if audit["potential_proxies_detected"]:
                flagged_count += 1
                for cat, items in audit["categories"].items():
                    proxy_rows.append({
                        "Candidate ID": cid,
                        "Candidate Name": cand.get("candidate_name", "Candidate"),
                        "Proxy Category": cat,
                        "Detected Clues": ", ".join(items)
                    })

        if proxy_rows:
            st.warning(f"⚠️ Flagged {flagged_count} candidate(s) with potential demographic proxies.")
            st.dataframe(pd.DataFrame(proxy_rows), use_container_width=True)
            st.caption("Notice: Scoring algorithms DO NOT ingest these proxy features into the ranking model.")
        else:
            st.info("✅ No obvious demographic proxy markers detected in currently loaded resumes.")

    with tab_eval:
        st.markdown("### 📊 Model Evaluation Metrics (Classification & Ranking)")
        st.write("Distinguishes classification metrics (binary shortlist accuracy) from ranking metrics (ordering quality).")

        eval_data_present = st.session_state["local_eval_dataset"] is not None

        if not eval_data_present:
            st.info("ℹ️ **Status: Evaluation data required**")
            st.write("No ground-truth benchmark dataset is currently loaded. Ground-truth evaluation metrics require actual hiring decision labels.")
            if st.button("📥 Load Synthetic Ground-Truth Benchmark Dataset (16 Samples)"):
                benchmark_path = os.path.join(REPO_ROOT, "sample_data", "benchmark_eval_data.json")
                with open(benchmark_path, "r", encoding="utf-8") as f:
                    bdata = json.load(f)
                st.session_state["local_eval_dataset"] = bdata
                st.session_state["local_demo_groups"] = {c["candidate_id"]: c["demographic_group"] for c in bdata["candidates"]}
                st.success("Loaded ground-truth benchmark dataset!")
                st.rerun()
        else:
            bdata = st.session_state["local_eval_dataset"]
            eval_res = evaluation_service.compute_metrics(
                candidate_eval_items=bdata["candidates"],
                k=5,
                threshold=70.0,
                dataset_name=bdata.get("dataset_name")
            )

            st.success(f"✅ Evaluated on **{eval_res['total_samples']} candidates** from '{eval_res['dataset_name']}'")

            # Classification Section
            st.markdown("#### 1. Classification Metrics (Cutoff: 70 Score Points)")
            cls_m = eval_res["classification_metrics"]
            c_p, c_r, c_f1, c_auc = st.columns(4)
            c_p.metric("Precision", f"{cls_m['precision']:.3f}", "Selected that are truly relevant")
            c_r.metric("Recall", f"{cls_m['recall']:.3f}", "Truly relevant that were selected")
            c_f1.metric("F1-Score", f"{cls_m['f1_score']:.3f}", "Harmonic mean of P & R")
            c_auc.metric("ROC-AUC", f"{cls_m['roc_auc']:.3f}" if cls_m['roc_auc'] else "N/A", "Discrimination capability")

            st.markdown("---")
            # Ranking Section
            st.markdown("#### 2. Information Retrieval & Ranking Metrics (Top-5 Cutoff)")
            rnk_m = eval_res["ranking_metrics"]
            c_ndcg, c_patk, c_mrr = st.columns(3)
            c_ndcg.metric("NDCG@5", f"{rnk_m['ndcg_at_k']:.3f}", "Graded relevance discount gain")
            c_patk.metric("Precision@5", f"{rnk_m['precision_at_k']:.3f}", "Proportion of relevant in top 5")
            c_mrr.metric("MRR", f"{rnk_m['mean_reciprocal_rank']:.3f}", "Reciprocal rank of 1st relevant")

            st.markdown("#### 📌 Metric Interpretation Guide:")
            for note in eval_res["notes"]:
                st.write(f"• {note}")

    with tab_limitations:
        st.markdown("### ⚠️ Responsible AI Principles & System Limitations")
        st.write("Responsible recruitment AI requires transparent documentation of architectural and societal limitations:")

        limits = fairness_service.get_fairness_limitations()
        for idx, lim in enumerate(limits, 1):
            title, desc = lim.split(":", 1) if ":" in lim else (f"Limitation {idx}", lim)
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(244, 63, 94, 0.25); border-left: 4px solid #f43f5e; border-radius: 12px; padding: 14px 20px; margin-bottom: 14px; backdrop-filter: blur(14px); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);">
                <b style="color: #fb7185; font-size: 14.5px;">{title}</b><br>
                <span style="color: #cbd5e1; font-size: 13.5px; line-height: 1.6; display: block; margin-top: 5px;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        #### Core Ethical Guardrails Enforced:
        1. **Strict Feature Restriction:** Protected characteristics (gender, caste, religion, date of birth) are entirely absent from candidate feature vectors.
        2. **No Automated Inferences:** The engine never uses computer vision or name heuristics to infer gender, race, or religion.
        3. **No Automated Disqualification:** The system assists human review through transparent match scoring and SHAP explanations; candidates are not autonomously rejected.
        4. **Transparent Fallback Attribution:** When SHAP is unavailable, feature attributions are clearly designated as fallback rather than misrepresenting genuine SHAP outputs.
        """)
