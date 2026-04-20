# AI Resume Screening Project

This project reads resume-style data from a CSV file, trains two machine learning models to predict whether a candidate is likely to be hired or not hired, and includes an interactive Streamlit dashboard for demos.

## What the program does

1. Loads the CSV dataset.
2. Cleans text fields like Skills, Certifications, and Job Role.
3. Converts text into features the computer can understand.
4. Uses numerical details like experience, salary expectation, projects, and education.
5. Trains two models.
6. Evaluates the models and creates charts.
7. Tests one sample candidate at the end.
8. Launches a Streamlit dashboard for interactive scoring and visualization.

## Models used

- Naive Bayes: a simple and fast model that works well with text data.
- Logistic Regression: a model that learns patterns and predicts the chance of a candidate being hired.

Both models learn from the dataset labels in the Recruiter Decision column.

## Input data

The script uses this file:

- AI_Resume_Screening (1).csv

Important columns used by the script:

- Skills
- Experience (Years)
- Education
- Certifications
- Job Role
- Recruiter Decision
- Salary Expectation ($)
- Projects Count
- AI Score (0-100)

Note: the provided CSV does not include a `CGPA` column. The dashboard shows CGPA as a preview input, but it is not used as a trained feature because the dataset does not contain it.

## Output files

When you run the script, it creates plot images like:

- confusion_matrices.png
- experience_distribution.png
- projects_vs_salary.png
- hiring_by_education.png

These are output files, so they do not need to be pushed to GitHub.

## Requirements

Install these Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- scipy
- streamlit

## How to run in a virtual environment

Open a terminal in the project folder and run:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python dsbdamodel.py
```

If the virtual environment already exists, just activate it and run the script:

```bash
source .venv/Scripts/activate
python dsbdamodel.py
```

## Streamlit dashboard

Run the dashboard with:

```bash
streamlit run app.py
```

The dashboard includes:

- model comparison and confusion matrices
- interactive resume scoring
- keyword extraction from the resume text
- experience, education, and projects visualizations
- a concept coverage panel that flags the missing CGPA field

## Notes

- The first run may download the NLTK stopwords list.
- If you change the CSV file, make sure the Recruiter Decision column still uses values like Hire and Reject.
- The script prints the model results in the terminal and saves the charts as PNG files.
