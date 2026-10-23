import numpy as np
import pandas as pd

from src.data_preprocessing import (
    FEATURES, ROLES, SIZES, clean_data, map_job_role, parse_salary_estimate,
)
from src.predict import predict_with_interval
from src.train_model import main as train_main


def make_raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    titles = ["Data Scientist", "Senior Data Analyst", "Machine Learning Engineer",
              "Software Engineer II", "DevOps Engineer", "Marketing Manager"]
    lo = rng.integers(40, 120, n)
    return pd.DataFrame({
        "Job Title": rng.choice(titles, n),
        "Salary Estimate": [f"${a}K-${a + 30}K (Glassdoor est.)" for a in lo],
        "Rating": rng.choice([-1, 3.2, 3.8, 4.5], n),
        "Location": rng.choice(["Austin, TX", "New York, NY", "Remote", "Boise, ID"], n),
        "Size": rng.choice(SIZES + ["Unknown"], n),
        "Founded": rng.choice([-1, 1990, 2005, 2012], n),
    })


def test_parse_salary():
    assert parse_salary_estimate("$53K-$91K (Glassdoor est.)")[:2] == (53.0, 91.0)
    lo, hi, hourly = parse_salary_estimate("Employer Provided Salary:$25-$35 Per Hour")
    assert hourly == 1 and round(lo) == 52 and round(hi) == 73


def test_missing_salary_is_nan():
    lo, hi, _ = parse_salary_estimate("-1")
    assert np.isnan(lo) and np.isnan(hi)


def test_missing_salary_marker_is_nan():
    lo, hi, _ = parse_salary_estimate("-1")
    assert np.isnan(lo) and np.isnan(hi)


def test_role_mapping():
    assert map_job_role("Senior Data Scientist") == "Data Scientist"
    assert map_job_role("ML Engineer") == "ML Engineer"
    assert map_job_role("Pastry Chef") == "Other Role"


def test_clean_data():
    df = clean_data(make_raw())
    assert set(FEATURES) <= set(df.columns)
    assert df["Rating"].between(1, 5).all()
    assert df["Company_Age"].ge(0).all()
    assert set(df["Job_Role"]) <= set(ROLES)


def test_train_and_predict(tmp_path):
    csv = tmp_path / "d.csv"
    make_raw().to_csv(csv, index=False)
    model_path = tmp_path / "m.pkl"
    metrics = train_main(str(csv), str(model_path), str(tmp_path / "m.json"))
    assert metrics["n_rows"] > 100

    import joblib
    model = joblib.load(model_path)
    row = {"Job_Role": "Data Scientist", "Size": SIZES[3], "Rating": 4.0,
           "State": "CA", "Company_Age": 10}
    mean, low, high = predict_with_interval(model, row)
    assert low <= mean <= high
