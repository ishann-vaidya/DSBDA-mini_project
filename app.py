#!/usr/bin/env python
"""Streamlit dashboard for the AI resume screening project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import confusion_matrix

from dsbdamodel import (
    evaluate_models,
    extract_features,
    load_data,
    preprocess_dataframe,
    preprocess_text,
    split_data,
    train_models,
)


st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path(__file__).with_name("AI_Resume_Screening (1).csv")


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(59,130,246,0.15), transparent 26%),
                linear-gradient(180deg, #0b1220 0%, #101a2f 100%);
            color: #e5eefc;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.84));
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 18px 60px rgba(0,0,0,0.24);
        }
        .chip {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            margin: 0.15rem 0.35rem 0.15rem 0;
            border-radius: 999px;
            background: rgba(59,130,246,0.14);
            border: 1px solid rgba(96,165,250,0.24);
            color: #dbeafe;
            font-size: 0.78rem;
        }
        .glass {
            background: rgba(15,23,42,0.76);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 18px;
            padding: 1rem 1rem 0.75rem 1rem;
        }
        h1, h2, h3, h4, p, label, div {
            color: #e5eefc !important;
        }
        .small-note {
            color: #a5b4fc !important;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def encode_education(education: str) -> int:
    edu_order = {
        "high school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5,
        "doctorate": 5,
    }
    education_lower = str(education).lower()
    for key, score in edu_order.items():
        if key in education_lower:
            return score
    return 2


@st.cache_data(show_spinner=False)
def load_and_prepare_data() -> pd.DataFrame:
    df = load_data(str(DATA_PATH))
    return preprocess_dataframe(df)


@st.cache_resource(show_spinner=False)
def train_pipeline():
    df = load_and_prepare_data().copy()
    X, y, tfidf, scaler = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    return df, X_test, y_test, tfidf, scaler, models, results


def concept_status() -> list[tuple[str, str, str]]:
    return [
        ("Text preprocessing", "Implemented", "Lowercasing, cleaning, stopword removal, and stemming are active."),
        ("TF-IDF feature extraction", "Implemented", "Skills, certifications, and job role are vectorized."),
        ("Classification", "Implemented", "Naive Bayes and Logistic Regression are trained."),
        ("Feature engineering", "Implemented", "Numeric features plus text features are combined."),
        ("Skills input", "Implemented", "Skills are part of the model input and dashboard form."),
        ("Projects input", "Implemented", "Projects Count is used as a numeric feature."),
        ("CGPA", "Missing in CSV", "The provided dataset does not contain a CGPA column."),
    ]


def build_candidate_features(
    skills: str,
    certifications: str,
    job_role: str,
    experience_years: float,
    salary_expectation: float,
    projects_count: int,
    education: str,
    tfidf,
    scaler,
):
    raw_text = f"{skills} {certifications} {job_role}".strip()
    cleaned = preprocess_text(raw_text)
    x_text = tfidf.transform([cleaned])
    x_numeric = scaler.transform([[experience_years, salary_expectation, projects_count, encode_education(education)]])
    return raw_text, hstack([x_text, csr_matrix(x_numeric)])


def top_keywords(text: str, tfidf, limit: int = 8) -> list[str]:
    cleaned = preprocess_text(text)
    x_text = tfidf.transform([cleaned])
    scores = x_text.toarray().ravel()
    if not np.any(scores):
        return []
    features = np.array(tfidf.get_feature_names_out())
    top_indexes = np.argsort(scores)[::-1][:limit]
    return [features[i] for i in top_indexes if scores[i] > 0]


def fig_from_confusion_matrix(y_true, y_pred, title: str, figsize=(4.0, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Hired", "Hired"],
        yticklabels=["Not Hired", "Hired"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_experience_distribution(df: pd.DataFrame, figsize=(4.2, 3.1)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df,
        x="Experience (Years)",
        hue="label",
        bins=15,
        palette={0: "#f87171", 1: "#34d399"},
        alpha=0.72,
        ax=ax,
    )
    ax.set_title("Experience vs Hiring Outcome")
    ax.set_xlabel("Experience (Years)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def fig_projects_vs_salary(df: pd.DataFrame, figsize=(4.2, 3.1)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = df["label"].map({0: "#ef4444", 1: "#22c55e"})
    ax.scatter(
        df["Projects Count"],
        df["Salary Expectation ($)"],
        c=colors,
        alpha=0.62,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.set_title("Projects vs Salary Expectation")
    ax.set_xlabel("Projects Count")
    ax.set_ylabel("Salary Expectation ($)")
    fig.tight_layout()
    return fig


def fig_hiring_by_education(df: pd.DataFrame, figsize=(4.2, 3.1)):
    fig, ax = plt.subplots(figsize=figsize)
    edu_hire = df.groupby("Education")["label"].mean().sort_values(ascending=False)
    edu_hire.plot(kind="bar", color="#6366f1", edgecolor="white", ax=ax)
    ax.set_title("Hiring Rate by Education Level")
    ax.set_ylabel("Hire Rate")
    ax.set_xlabel("Education")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    return fig


df, X_test, y_test, tfidf, scaler, models, results = train_pipeline()
best_model_name = max(results, key=lambda name: results[name]["accuracy"])


QUESTION_ORDER = [
    "skills",
    "certifications",
    "job_role",
    "experience_years",
    "salary_expectation",
    "projects_count",
    "education",
    "cgpa",
    "model_name",
]


DEFAULT_FORM_STATE = {
    "skills": "Python, Machine Learning, NLP, SQL, Data Visualization",
    "certifications": "AWS Certified, TensorFlow Developer",
    "job_role": "Data Scientist",
    "experience_years": 3.0,
    "salary_expectation": 85000,
    "projects_count": 6,
    "education": sorted(df["Education"].dropna().astype(str).unique().tolist())[0],
    "cgpa": 8.2,
    "model_name": best_model_name,
}


if "step_index" not in st.session_state:
    st.session_state.step_index = 0

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "resume_result" not in st.session_state:
    st.session_state.resume_result = None

for key, value in DEFAULT_FORM_STATE.items():
    st.session_state.setdefault(key, value)


def reset_flow() -> None:
    st.session_state.step_index = 0
    st.session_state.show_results = False
    st.session_state.resume_result = None


def current_step_label() -> str:
    labels = {
        "skills": "Skills",
        "certifications": "Certifications",
        "job_role": "Target job role",
        "experience_years": "Experience (Years)",
        "salary_expectation": "Salary expectation ($)",
        "projects_count": "Projects Count",
        "education": "Education",
        "cgpa": "CGPA preview",
        "model_name": "Choose model",
    }
    return labels[QUESTION_ORDER[st.session_state.step_index]]


def render_step_input() -> None:
    step_key = QUESTION_ORDER[st.session_state.step_index]

    st.subheader(f"Step {st.session_state.step_index + 1} of {len(QUESTION_ORDER)}")
    st.caption("Answer one question at a time. Your final screen will show the hired / not hired prediction.")

    with st.container():
        if step_key == "skills":
            st.session_state.skills = st.text_area("Skills", value=st.session_state.skills, height=100)
        elif step_key == "certifications":
            st.session_state.certifications = st.text_input("Certifications", value=st.session_state.certifications)
        elif step_key == "job_role":
            st.session_state.job_role = st.text_input("Target job role", value=st.session_state.job_role)
        elif step_key == "experience_years":
            st.session_state.experience_years = st.slider("Experience (Years)", 0.0, 15.0, float(st.session_state.experience_years), 0.5)
        elif step_key == "salary_expectation":
            st.session_state.salary_expectation = st.number_input(
                "Salary expectation ($)",
                min_value=0,
                value=int(st.session_state.salary_expectation),
                step=1000,
            )
        elif step_key == "projects_count":
            st.session_state.projects_count = st.slider("Projects Count", 0, 20, int(st.session_state.projects_count))
        elif step_key == "education":
            st.session_state.education = st.selectbox(
                "Education",
                sorted(df["Education"].dropna().astype(str).unique().tolist()),
                index=sorted(df["Education"].dropna().astype(str).unique().tolist()).index(st.session_state.education),
            )
        elif step_key == "cgpa":
            st.session_state.cgpa = st.slider("CGPA preview", 0.0, 10.0, float(st.session_state.cgpa), 0.1)
        elif step_key == "model_name":
            st.session_state.model_name = st.selectbox(
                "Model",
                list(models.keys()),
                index=list(models.keys()).index(st.session_state.model_name),
            )

    left_col, right_col = st.columns([1, 1])
    with left_col:
        if st.session_state.step_index > 0 and st.button("Back", use_container_width=True):
            st.session_state.step_index -= 1
            st.rerun()
    with right_col:
        button_label = "Predict" if st.session_state.step_index == len(QUESTION_ORDER) - 1 else "Next"
        if st.button(button_label, type="primary", use_container_width=True):
            if st.session_state.step_index < len(QUESTION_ORDER) - 1:
                st.session_state.step_index += 1
                st.rerun()

            selected_model = models[st.session_state.model_name]
            raw_text, candidate_features = build_candidate_features(
                st.session_state.skills,
                st.session_state.certifications,
                st.session_state.job_role,
                float(st.session_state.experience_years),
                float(st.session_state.salary_expectation),
                int(st.session_state.projects_count),
                st.session_state.education,
                tfidf,
                scaler,
            )
            prediction = int(selected_model.predict(candidate_features)[0])
            probabilities = selected_model.predict_proba(candidate_features)[0]
            keywords = top_keywords(raw_text, tfidf)

            st.session_state.resume_result = {
                "prediction": prediction,
                "probabilities": probabilities,
                "keywords": keywords,
                "raw_text": raw_text,
                "model_name": st.session_state.model_name,
            }
            st.session_state.show_results = True
            st.session_state.step_index = 0
            st.rerun()


def render_result_page() -> None:
    result = st.session_state.resume_result
    if not result:
        st.warning("No prediction found yet. Start the questions to generate one.")
        return

    selected_model = models[result["model_name"]]
    raw_text, candidate_features = build_candidate_features(
        st.session_state.skills,
        st.session_state.certifications,
        st.session_state.job_role,
        float(st.session_state.experience_years),
        float(st.session_state.salary_expectation),
        int(st.session_state.projects_count),
        st.session_state.education,
        tfidf,
        scaler,
    )

    st.subheader("Prediction result")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Prediction", "HIRED" if result["prediction"] == 1 else "NOT HIRED")
    result_col2.metric("Model used", result["model_name"])
    result_col3.metric("CGPA preview", f"{st.session_state.cgpa:.1f}/10")

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    details_col1, details_col2 = st.columns([1, 1])
    with details_col1:
        st.write("**Skills**")
        st.write(st.session_state.skills)
        st.write("**Certifications**")
        st.write(st.session_state.certifications)
        st.write("**Target job role**")
        st.write(st.session_state.job_role)
    with details_col2:
        st.write("**Experience (Years)**")
        st.write(st.session_state.experience_years)
        st.write("**Salary expectation ($)**")
        st.write(st.session_state.salary_expectation)
        st.write("**Projects Count**")
        st.write(st.session_state.projects_count)
        st.write("**Education**")
        st.write(st.session_state.education)

    probability_table = pd.DataFrame(
        {
            "Outcome": ["Not Hired", "Hired"],
            "Probability": np.round(result["probabilities"], 4),
        }
    )
    st.dataframe(probability_table, width="stretch", hide_index=True)

    st.write("Top extracted keywords:")
    st.write(", ".join(result["keywords"]) if result["keywords"] else "No strong keyword matches found.")

    with st.expander("Show model confidence and diagnostics", expanded=False):
        row1_left, row1_right = st.columns(2)
        with row1_left:
            st.pyplot(fig_from_confusion_matrix(y_test, results[best_model_name]["y_pred"], f"{best_model_name} confusion matrix"))
        with row1_right:
            st.pyplot(fig_experience_distribution(df))

        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.pyplot(fig_hiring_by_education(df))
        with row2_right:
            st.pyplot(fig_projects_vs_salary(df))

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict another resume", use_container_width=True):
        reset_flow()
        st.rerun()


st.markdown(
    """
    <div class="hero">
        <h1>Resume Screening AI Dashboard</h1>
        <p class="small-note">A compact hiring signal lab built on TF-IDF, text preprocessing, classification, and numeric feature fusion.</p>
        <div>
            <span class="chip">Naive Bayes</span>
            <span class="chip">Logistic Regression</span>
            <span class="chip">TF-IDF</span>
            <span class="chip">Resume Keywords</span>
            <span class="chip">Feature Engineering</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Resumes", f"{len(df):,}")
# col2.metric("Shortlist rate", f"{df['label'].mean() * 100:.1f}%")
# col3.metric("Best model", best_model_name)
# col4.metric("Best accuracy", f"{results[best_model_name]['accuracy'] * 100:.1f}%")

if st.session_state.show_results:
    render_result_page()
else:
    st.info("Use the flow below. Each click moves you to the next question until the final prediction screen appears.")
    render_step_input()
