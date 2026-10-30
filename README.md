<<<<<<< HEAD
# AI-Powered Resume Screening and Ranking System with Explainable AI

**Person 4 Implementation: Explainability, Fairness, API & Frontend Engineer**

A complete, production-ready system for automated resume screening and candidate ranking featuring **Explainable AI (SHAP)**, **Responsible AI & Bias Mitigation**, **FastAPI REST APIs**, and an interactive **Streamlit Dashboard**.

---

## 🏛️ Team Architecture & Person 4 Scope

The system is structured with clean modular adapters across 4 teammates:
- **Person 1 (Resume Parsing):** Text extraction from PDF, DOCX, and TXT; skill taxonomy parsing; experience estimation (`backend/services/resume_parser.py`).
- **Person 2 (NLP & Feature Engineering):** Required skills extraction, lexical keyword overlap, TF-IDF cosine similarity, education weighting (`backend/services/ranking_service.py`).
- **Person 3 (Matching Model & Ranking):** Interpretable Random Forest regression model predicting 0-100 match score (`backend/services/ranking_service.py`).
- **Person 4 (Explainability, Fairness, API & Frontend):** 
  - **Phase 8:** SHAP TreeExplainer & transparent attribution fallback with waterfall and bar charts (`backend/services/explainability.py`).
  - **Phase 9:** Bias mitigation, proxy feature detection, demographic masking, and group parity metrics (`backend/services/fairness.py`).
  - **Phase 10:** FastAPI REST backend endpoints, Pydantic validation, safe file handling (`backend/main.py`, `backend/api/routes.py`).
  - **Phase 11:** Classification and ranking evaluation metrics (`backend/services/evaluation.py`).
  - **Frontend:** 6-screen responsive Streamlit dashboard with interactive Plotly visualizations (`frontend/app.py`).

---

## 🌟 Key Features by Phase

### Phase 8: Explainable AI Using SHAP
- **SHAP TreeExplainer:** Computes exact Shapley additive feature values ($\phi_i$) decomposing candidate scores relative to baseline expectation ($E[f(x)]$).
- **Transparent Fallback Attribution:** If SHAP is unavailable or in fallback mode, computes transparent baseline attributions and **clearly labels it as a fallback rather than claiming genuine SHAP output**.
- **Human-Readable Explanations:** Automatically generates recruitment summaries (e.g., *"Strong match: Python and SQL found. Positive contribution: Relevant work experience (5.0 years). Missing requirement: Docker. Overall match score: 85.0/100."*).
- **Interactive Visualizations:** Waterfall decomposition charts and horizontal positive/negative contribution charts using Plotly.

### Phase 9: Bias Mitigation and Fairness Checks
- **Strict Exclusion:** Protected characteristics (gender, religion, caste, race, date of birth) are excluded from scoring features.
- **Demographic Masking:** Redacts names, honorifics (Mr., Ms.), gendered pronouns, and sensitive biographical headings.
- **Proxy Detection:** Audits text for indirect proxies (affinity student groups, graduation era markers that correlate with age).
- **Group Parity Auditing:** Computes **Demographic Parity Difference** and **Disparate Impact Ratio (EEOC 80% rule)** when lawful demographic evaluation data is present.
- **Ethical Guardrails:** Demographic attributes are **never** inferred from names or photos; candidates are never automatically rejected based on demographic proxies.
- **Documented Limitations:** Removes demographic fields alone does not guarantee a bias-free system due to historical training bias and contextual factors.

### Phase 10: FastAPI Backend
- `GET /health` — System status, API version, and SHAP readiness.
- `POST /upload-jd` — Upload TXT/PDF job description with validation.
- `POST /upload-resumes` — Multi-file resume upload (PDF, DOCX, TXT) with error isolation.
- `POST /rank-candidates` — Ranks candidates by calibrated 0-100 match score.
- `GET /candidate/{candidate_id}/explanation` — SHAP feature contributions, waterfall data, and human summary.
- `GET /fairness-report` — Demographic parity metrics, proxy detection flags, and ethical limitations.
- `GET /evaluation` — Precision, Recall, F1, ROC-AUC, NDCG@K, Precision@K, MRR (returns *"Evaluation data required"* when test labels are absent).
- `POST /load-benchmark` — Loads standardized 16-candidate evaluation dataset for demonstration.
- `POST /reset-session` — Clears temporary session state.
- Interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### Phase 11: Evaluation Metrics
- **Classification Metrics:** Precision, Recall, F1-Score, ROC-AUC at configurable decision thresholds.
- **Ranking Metrics:** Normalized Discounted Cumulative Gain (NDCG@K), Precision@K, Mean Reciprocal Rank (MRR).
- **Integrity Guardrail:** If no ground-truth test labels are loaded, displays *"Evaluation data required"* rather than fabricating fake results.

---

## 📁 Project Structure

```
pp1/
├── backend/
│   ├── main.py                  # FastAPI application entrypoint & CORS
│   ├── schemas.py               # Pydantic v2 schemas for all requests & responses
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # API endpoints (/upload-jd, /rank, /explanation, etc.)
│   └── services/
│       ├── __init__.py
│       ├── resume_parser.py     # Person 1: Text extraction (PDF/DOCX/TXT) & skill parsing
│       ├── ranking_service.py   # Person 2 & 3: Feature engineering & Random Forest model
│       ├── explainability.py    # Phase 8: SHAP TreeExplainer & labeled fallback
│       ├── fairness.py          # Phase 9: Demographic masking, proxy flags, group parity
│       └── evaluation.py        # Phase 11: Precision, Recall, F1, ROC-AUC, NDCG@K, MRR
├── frontend/
│   ├── app.py                   # Streamlit multi-screen web dashboard
│   └── components/
│       ├── __init__.py
│       ├── cards.py             # UI cards, badges, and responsive CSS
│       └── charts.py            # Plotly waterfall and contribution charts
├── sample_data/
│   ├── benchmark_eval_data.json # 16-candidate ground truth benchmark dataset
│   ├── job_descriptions/
│   │   ├── senior_python_dev.txt
│   │   └── data_scientist.txt
│   └── resumes/
│       ├── candidate_1_alice_smith_senior_dev.txt
│       ├── candidate_2_bob_jones_mid_dev.txt
│       ├── candidate_2_bob_jones_mid_dev.docx
│       ├── candidate_3_charlie_patel_data_scientist.txt
│       ├── candidate_4_diana_kim_junior_dev.txt
│       ├── candidate_5_edward_stone_cloud_architect.txt
│       └── candidate_6_fiona_taylor_marketing_unrelated.txt
├── tests/
│   ├── test_api.py              # TestClient tests for all REST endpoints
│   ├── test_parser.py           # Parsing, PDF/DOCX, corrupted file handling
│   ├── test_ranking.py          # Feature extraction, scoring monotonicity
│   ├── test_explainability.py   # SHAP explainer & fallback verification
│   ├── test_fairness.py         # Masking, proxy audit, 80% rule checks
│   ├── test_evaluation.py       # Classification & ranking metric formulas
│   └── test_e2e.py              # End-to-end recruitment workflow test
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .env.example                 # Environment configuration template
└── .gitignore                   # Git ignore rules
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- Windows / macOS / Linux

### 1. Clone & Install Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd pp1

# Optional: Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Option A: Run Backend & Frontend in Separate Terminals

#### Terminal 1 — Start the FastAPI Backend:
```bash
uvicorn backend.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

#### Terminal 2 — Start the Streamlit Frontend:
```bash
streamlit run frontend/app.py
```
- Open `http://localhost:8501` in your browser.

### Option B: Standalone Frontend Mode
The Streamlit frontend is designed with **dual connectivity**:
- If the FastAPI backend is running on port 8000, it makes REST API requests.
- If the backend is offline, it runs seamlessly in **Local Engine Mode** using the underlying service classes directly.

---

## 🖥️ Streamlit Screen Walkthrough

| Screen | Purpose |
| :--- | :--- |
| **1. 📊 Dashboard** | High-level recruitment metrics, candidate score distribution chart, and system architecture summary. |
| **2. 📄 Upload Job Description** | Upload TXT or PDF job description, review extracted required skills and expected experience, or 1-click load pre-set sample roles. |
| **3. 📁 Upload Resumes** | Batch upload multiple resumes (PDF, DOCX, TXT) with error validation, review extracted candidates, and click **"Rank Candidates"**. |
| **4. 🏆 Ranked Candidates** | Sortable candidate ranking table with search/filtering, match scores, matched skills, and disclaimer against automated hiring. |
| **5. 🔍 Candidate Explanation (SHAP)** | SHAP feature attributions, interactive waterfall chart, horizontal bar chart, positive/negative drivers, and human-readable explanation. Includes toggle to test fallback mode. |
| **6. ⚖️ Fairness & Evaluation** | Group fairness audit (Demographic Parity & 80% rule), resume proxy feature audit, and classification/ranking evaluation metrics on benchmark data. |

---

## 🧪 Running Automated Tests

Run the full automated test suite with pytest:

```bash
python -m pytest tests -v
```

### Test Coverage Highlights:
- `test_parser.py` — Valid & empty text, PDF extraction, DOCX parsing, unsupported formats, and corrupted files.
- `test_ranking.py` — Job description parsing, feature calculations, score monotonicity, descending rank order.
- `test_explainability.py` — SHAP TreeExplainer, fallback mode attribution labeling, and human-readable explanation content.
- `test_fairness.py` — Demographic masking, proxy detection, group fairness formulas, demographic data unavailable guardrail.
- `test_evaluation.py` — Missing evaluation labels guardrail (*"Evaluation data required"*), Precision, Recall, F1, ROC-AUC, NDCG@K, Precision@K, and MRR.
- `test_api.py` — All FastAPI endpoints with `TestClient`.
- `test_e2e.py` — Complete end-to-end recruitment screening workflow from upload to evaluation.

---

## ⚖️ Responsible AI Principles & Ethical Notice

1. **Assisted Human Decision-Making:** Match scores reflect qualification alignment against job description keywords and experience criteria. They must **never** be treated as a probability of hiring success or used for automated employment decisions.
2. **Fairness through Unawareness Limitation:** Removing demographic identifiers (names, gender) does not eliminate bias because remaining text features can act as subtle proxies.
3. **Transparent Fallbacks:** When a compatible SHAP model is unavailable, feature attributions are clearly labeled as fallback estimations rather than misrepresenting genuine SHAP outputs.
=======
# Automated Resume Screening and Candidate Ranking System

An AI-driven recruitment tool that screens resumes, compares them with a job description, ranks candidates by suitability, and explains each recommendation. Built with Machine Learning and NLP (not an LLM or an agent).

## Problem

Organizations receive thousands of resumes per job opening. Manual screening is slow, inconsistent, and prone to bias, and keyword filters miss the meaning behind a candidate's experience. This system parses resumes, understands skills and experience semantically, ranks candidates, and gives transparent reasons for its decisions.

## Features

- Resume parsing (PDF and DOCX): name, education, skills, experience, certifications, projects, achievements
- NLP processing: tokenization, stop-word removal, lemmatization, NER, skill and experience extraction
- Feature engineering: TF-IDF, word and sentence embeddings (SBERT), skill/experience/education match scores
- Candidate matching: cosine similarity, Euclidean distance, skill overlap
- ML models compared: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- Ranking: match score (0-100), rank, match percentage, shortlist recommendation
- Explainable AI (SHAP): why a candidate was selected, strengths, missing skills
- Bias mitigation: gender-neutral evaluation and fairness checks
- FastAPI backend with a Streamlit frontend

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Parsing | pdfplumber, python-docx |
| NLP | spaCy, NLTK |
| Embeddings | sentence-transformers (SBERT) |
| ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Fairness | fairlearn |
| API / UI | FastAPI, uvicorn, Streamlit |
| Data | pandas, NumPy |

## Project Structure

```
Automated-Resume-Screening/
├── data/
│   ├── raw_resumes/            # resume files (not committed)
│   ├── job_descriptions/
│   └── candidate_features/     # numeric feature CSVs
├── parsers/                    # resume parsing
├── nlp/                        # NLP processing
├── features/                   # feature engineering
├── matching/                   # similarity_matcher.py
├── models/                     # train_model.py, ranking.py
├── explainability/             # SHAP explanations
├── api/                        # FastAPI backend
├── notebooks/
├── requirements.txt
└── README.md
```

## Setup

Python 3.11 is recommended, since some libraries do not yet support newer versions.

```bash
git clone https://github.com/Darshankg18/Automated-Resume-Screening.git
cd Automated-Resume-Screening

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Run the candidate matching engine:

```bash
python matching/similarity_matcher.py
```

Train and compare all five models:

```bash
python models/train_model.py
```

## Pipeline

```
Resume (PDF/DOCX) -> Parsing -> NLP processing -> Feature engineering
-> Matching engine -> ML model -> Ranking -> Explanation -> API / UI
```

## Current Results

Evaluated on a 60-candidate synthetic feature dataset (80/20 train/test split):

| Metric | Logistic Regression | Random Forest | SVM | XGBoost | LightGBM |
|---|---|---|---|---|---|
| Accuracy | 0.833 | 0.833 | 0.833 | 0.833 | 0.750 |
| Precision | 0.778 | 0.857 | 0.857 | 0.857 | 0.833 |
| Recall | 1.000 | 0.857 | 0.857 | 0.857 | 0.714 |
| F1-Score | 0.875 | 0.857 | 0.857 | 0.857 | 0.769 |

Logistic Regression has the best F1-score on this small dataset. Results will be re-evaluated on real resume-derived features.

## Expected Output (per candidate)

```json
{
  "candidate_name": "John Doe",
  "match_score": 92.5,
  "rank": 1,
  "skills_matched": ["Python", "Machine Learning", "SQL"],
  "missing_skills": ["Docker"],
  "experience_match": "High",
  "recommendation": "Strongly Recommended",
  "explanation": "Candidate matches 92% of job requirements with strong ML and NLP experience."
}
```

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC, processing time, fairness score.

## Team Roles

| Member | Responsibility |
|---|---|
| Person 1 | Data collection and resume parsing |
| Person 2 | NLP processing and feature engineering |
| Person 3 | Matching engine, ML models, ranking |
| Person 4 | Explainability, fairness, API, frontend |

## Status

- [x] Environment setup
- [x] Candidate matching engine (cosine similarity, Euclidean distance)
- [x] Five ML models trained and compared
- [ ] Ranking system (in progress)
- [ ] Resume parsing and NLP features
- [ ] Explainability and bias mitigation
- [ ] API and frontend

## Responsible Use

This is an academic project. It is not audited for real hiring decisions, and a human recruiter should always make the final call. Real resumes contain personal data and are excluded from the repository via `.gitignore`.
>>>>>>> a2775bbfbb3973b998110cb7229ec8adbcd3bb87
