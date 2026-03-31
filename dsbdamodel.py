#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[3]:


from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[4]:


nltk.download('stopwords', quiet=True)


# In[5]:


# STEP 1: LOAD DATASET
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
 
    print(f"Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
 
    # ── Encode label: hire/hired/selected-style decisions → 1, otherwise 0 ────────
    print(f"\nRecruiter Decision values: {df['Recruiter Decision'].unique()}")
    positive_decisions = {'hire', 'hired', 'selected', 'yes', 'approved'}
    df['label'] = df['Recruiter Decision'].apply(
        lambda x: 1 if str(x).strip().lower() in positive_decisions else 0
    )

    if df['label'].nunique() < 2:
        raise ValueError(
            "After encoding 'Recruiter Decision', only one class was found. "
            "Check that the CSV contains both positive and negative outcomes."
        )
    print(f"\nLabel distribution:\n{df['label'].value_counts()}\n")
    return df


# In[6]:


# STEP 2: TEXT PREPROCESSING
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
 
def preprocess_text(text: str) -> str:
    """Lowercase → remove special chars → remove stopwords → stem"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)
 
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # ── Combine all text columns into one ────────────────────
    # Skills + Certifications + Job Role → rich text signal
    df['combined_text'] = (
        df['Skills'].fillna('') + ' ' +
        df['Certifications'].fillna('') + ' ' +
        df['Job Role'].fillna('')
    )
    df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
 
    # ── Encode Education as ordinal number ───────────────────
    edu_order = {
        "high school": 1,
        "associate": 2,
        "bachelor's": 3,
        "bachelor": 3,
        "master's": 4,
        "master": 4,
        "phd": 5,
        "doctorate": 5
    }
    def encode_education(val):
        if not isinstance(val, str):
            return 2
        val_lower = val.strip().lower()
        for key, score in edu_order.items():
            if key in val_lower:
                return score
        return 2  # fallback
 
    df['education_encoded'] = df['Education'].apply(encode_education)
 
    # ── Fill missing numerics with median ────────────────────
    for col in ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count']:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        df[col] = numeric_col.fillna(numeric_col.median())
 
    print("Preprocessing done")
    print(f"   Sample cleaned text: {df['cleaned_text'].iloc[0][:100]}...")
    print(f"   Education sample  : {df['Education'].iloc[0]} → {df['education_encoded'].iloc[0]}\n")
    return df


# In[7]:


# ============================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================
# Features used:
#   TF-IDF (Skills + Certifications + Job Role) → 500 cols
#   Experience (Years)                           → 1 col
#   Salary Expectation ($)                       → 1 col
#   Projects Count                               → 1 col
#   Education (encoded)                          → 1 col
#                                          TOTAL = 504 features
 
NUMERIC_COLS = [
    'Experience (Years)',
    'Salary Expectation ($)',
    'Projects Count',
    'education_encoded'
]
 
def extract_features(df: pd.DataFrame):
    # TF-IDF on combined text
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df['cleaned_text'])
 
    # Scale numeric features to [0, 1]
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[NUMERIC_COLS])
    X_numeric_sparse = csr_matrix(X_numeric)
 
    # Stack text + numeric
    X = hstack([X_text, X_numeric_sparse])
    y = df['label'].values
 
    print(f"Feature matrix shape: {X.shape}")
    print(f"   TF-IDF: 500 | Numeric: {len(NUMERIC_COLS)} | Total: {X.shape[1]}\n")
    return X, y, tfidf, scaler


# In[8]:


# STEP 4: TRAIN / TEST SPLIT
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


# In[9]:


# STEP 5: TRAIN MODELS
def train_models(X_train, y_train):
    models = {
        "Naive Bayes":         MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"Trained: {name}")
    print()
    return trained


# In[10]:


# STEP 6: EVALUATE
def evaluate_models(models: dict, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': acc, 'y_pred': y_pred}
 
        print(f"{'='*50}")
        print(f"{name}  —  Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred,
                                    target_names=['Not Hired', 'Hired']))
    return results


# In[11]:


# STEP 7: VISUALIZATIONS
def plot_results(models: dict, results: dict, y_test, df: pd.DataFrame):
 
    # ── Confusion matrices ───────────────────────────────────
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, _) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, results[name]['y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=['Not Hired', 'Hired']).plot(
            ax=ax, colorbar=False, cmap='Blues'
        )
        ax.set_title(f'{name}\nAccuracy: {results[name]["accuracy"]*100:.1f}%')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150)
    plt.show()
    print("Saved confusion_matrices.png")
 
    # ── Experience distribution by hiring decision ───────────
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='Experience (Years)', hue='label', bins=15,
                 palette={0: '#ef4444', 1: '#22c55e'}, alpha=0.7)
    plt.title('Experience (Years) — Hired vs Not Hired')
    plt.legend(labels=['Not Hired', 'Hired'])
    plt.tight_layout()
    plt.savefig('experience_distribution.png', dpi=150)
    plt.show()
    print("Saved experience_distribution.png")
 
    # ── Projects Count vs Salary scatter ─────────────────────
    plt.figure(figsize=(7, 4))
    colors = df['label'].map({0: '#ef4444', 1: '#22c55e'})
    plt.scatter(df['Projects Count'], df['Salary Expectation ($)'],
                c=colors, alpha=0.5, edgecolors='white', linewidths=0.3)
    plt.xlabel('Projects Count')
    plt.ylabel('Salary Expectation ($)')
    plt.title('Projects vs Salary — Hired (green) / Not Hired (red)')
    plt.tight_layout()
    plt.savefig('projects_vs_salary.png', dpi=150)
    plt.show()
    print("Saved projects_vs_salary.png")
 
    # ── Hiring rate by Education level ───────────────────────
    edu_hire = df.groupby('Education')['label'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    edu_hire.plot(kind='bar', color='#6366f1', edgecolor='white')
    plt.title('Hiring Rate by Education Level')
    plt.ylabel('Hire Rate')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('hiring_by_education.png', dpi=150)
    plt.show()
    print("Saved hiring_by_education.png\n")


# In[12]:


# STEP 8: PREDICT A NEW RESUME
def predict_resume(
    skills: str,
    certifications: str,
    job_role: str,
    experience_years: float,
    salary_expectation: float,
    projects_count: int,
    education: str,
    model,
    tfidf: TfidfVectorizer,
    scaler: MinMaxScaler
):
    """Feed a new candidate's details → get hiring prediction"""
 
    raw_text = skills + ' ' + certifications + ' ' + job_role
    cleaned = preprocess_text(raw_text)
 
    edu_order = {"high school": 1, "associate": 2, "bachelor": 3,
                 "master": 4, "phd": 5, "doctorate": 5}
    edu_score = 2
    for key, score in edu_order.items():
        if key in education.lower():
            edu_score = score
            break
 
    X_text = tfidf.transform([cleaned])
    X_numeric = scaler.transform([[experience_years, salary_expectation,
                                    projects_count, edu_score]])
    X = hstack([X_text, csr_matrix(X_numeric)])
 
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
 
    label = "HIRED" if prediction == 1 else "NOT HIRED"
    print(f"\n{'='*45}")
    print(f"  Candidate  : {job_role}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {max(proba)*100:.1f}%")
    print(f"{'='*45}\n")
    return prediction


# In[14]:


if __name__ == "__main__":
 
    # ── 1. Load
    df = load_data(r"C:\Users\Ishan N Vaidya\Desktop\DSBDA Mini Project\AI_Resume_Screening (1).csv")
 
    # ── 2. Preprocess
    df = preprocess_dataframe(df)
 
    # ── 3. Features
    X, y, tfidf, scaler = extract_features(df)
 
    # ── 4. Split
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    # ── 5. Train
    models = train_models(X_train, y_train)
 
    # ── 6. Evaluate
    results = evaluate_models(models, X_test, y_test)
 
    # ── 7. Visualize
    plot_results(models, results, y_test, df)
 
    # ── 8. Test with a sample candidate
    best_model = models["Logistic Regression"]
 
    predict_resume(
        skills="Python, Machine Learning, Deep Learning, SQL",
        certifications="AWS Certified, TensorFlow Developer",
        job_role="Data Scientist",
        experience_years=3,
        salary_expectation=85000,
        projects_count=6,
        education="Master's",
        model=best_model,
        tfidf=tfidf,
        scaler=scaler
    )


# In[ ]:




