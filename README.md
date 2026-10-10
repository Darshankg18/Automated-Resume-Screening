Automated Resume Screening and Candidate Ranking System

An AI-driven recruitment tool that screens resumes, compares them with a job description, ranks candidates by suitability, and explains each recommendation. Built with Machine Learning and NLP (not an LLM or an agent).

Problem

Organizations receive thousands of resumes per job opening. Manual screening is slow, inconsistent, and prone to bias, and keyword filters miss the meaning behind a candidate's experience. This system parses resumes, understands skills and experience semantically, ranks candidates, and gives transparent reasons for its decisions.

Features
Resume parsing (PDF and DOCX): name, education, skills, experience, certifications, projects, achievements
NLP processing: tokenization, stop-word removal, lemmatization, NER, skill and experience extraction
Feature engineering: TF-IDF, word and sentence embeddings (SBERT), skill/experience/education match scores
Candidate matching: cosine similarity, Euclidean distance, skill overlap
ML models compared: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
Ranking: match score (0-100), rank, match percentage, shortlist recommendation
Explainable AI (SHAP): why a candidate was selected, strengths, missing skills
Bias mitigation: gender-neutral evaluation and fairness checks
FastAPI backend with a Streamlit frontend
Tech Stack
Area	Tools
Language	Python 3.11
Parsing	pdfplumber, python-docx
NLP	spaCy, NLTK
Embeddings	sentence-transformers (SBERT)
ML	scikit-learn, XGBoost, LightGBM
Explainability	SHAP
Fairness	fairlearn
API / UI	FastAPI, uvicorn, Streamlit
Data	pandas, NumPy
Project Structure
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
Setup

Python 3.11 is recommended, since some libraries do not yet support newer versions.

bash
git clone https://github.com/Darshankg18/Automated-Resume-Screening.git
cd Automated-Resume-Screening

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
Usage

Run the candidate matching engine:

bash
python matching/similarity_matcher.py

Train and compare all five models:

bash
python models/train_model.py
Pipeline
Resume (PDF/DOCX) -> Parsing -> NLP processing -> Feature engineering
-> Matching engine -> ML model -> Ranking -> Explanation -> API / UI
Current Results

Evaluated on a 60-candidate synthetic feature dataset (80/20 train/test split):

Metric	Logistic Regression	Random Forest	SVM	XGBoost	LightGBM
Accuracy	0.833	0.833	0.833	0.833	0.750
Precision	0.778	0.857	0.857	0.857	0.833
Recall	1.000	0.857	0.857	0.857	0.714
F1-Score	0.875	0.857	0.857	0.857	0.769

Logistic Regression has the best F1-score on this small dataset. Results will be re-evaluated on real resume-derived features.

Expected Output (per candidate)
json
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
Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC, processing time, fairness score.

Team Roles
Member	Responsibility
Person 1	Data collection and resume parsing
Person 2	NLP processing and feature engineering
Person 3	Matching engine, ML models, ranking
Person 4	Explainability, fairness, API, frontend
Status
 Environment setup
 Candidate matching engine (cosine similarity, Euclidean distance)
 Five ML models trained and compared
 Ranking system (in progress)
 Resume parsing and NLP features
 Explainability and bias mitigation
 API and frontend
Responsible Use

This is an academic project. It is not audited for real hiring decisions, and a human recruiter should always make the final call. Real resumes contain personal data and are excluded from the repository via .gitignore.
