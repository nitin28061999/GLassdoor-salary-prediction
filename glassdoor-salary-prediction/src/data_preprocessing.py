"""Cleaning and feature extraction for the Glassdoor jobs dataset.

Works with both the raw Kaggle export (``Salary Estimate`` text column) and the
pre-cleaned version that already carries ``avg_salary`` (in thousands of USD).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REF_YEAR = 2017  # dataset snapshot year; Company_Age = REF_YEAR - Founded

ROLES = [
    "Data Scientist",
    "Data Analyst",
    "ML Engineer",
    "Software Engineer",
    "DevOps Engineer",
    "Other Role",
]
TOP_STATES = ["CA", "NY", "TX", "MA", "VA", "IL", "WA"]
STATES = TOP_STATES + ["Other"]
SIZES = [
    "1 to 50 employees",
    "51 to 200 employees",
    "201 to 500 employees",
    "501 to 1000 employees",
    "1001 to 5000 employees",
    "5001 to 10000 employees",
    "10000+ employees",
]

CAT_FEATURES = ["Job_Role", "Size", "State"]
NUM_FEATURES = ["Rating", "Company_Age"]
FEATURES = ["Job_Role", "Size", "Rating", "State", "Company_Age"]
TARGET = "avg_salary"

# Order matters: first match wins.
_ROLE_RULES = [
    ("ML Engineer", r"machine learning|\bml\b|\bai\b|deep learning|\bnlp\b|computer vision"),
    ("Data Scientist", r"data scien"),
    ("Data Analyst", r"data analy|business analy|\bbi\b|analytics"),
    ("DevOps Engineer", r"devops|site reliability|\bsre\b|cloud engineer|infrastructure"),
    ("Software Engineer", r"software|developer|full.?stack|back.?end|front.?end|engineer"),
]


def _find_col(df: pd.DataFrame, *candidates: str):
    lookup = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    return None


def parse_salary_estimate(value) -> tuple:
    """Parse text like ``$53K-$91K (Glassdoor est.)`` -> (min_k, max_k, hourly).

    Hourly rates are converted to annual (x2080 hours). Values are in $K.
    """
    if pd.isna(value) or "$" not in str(value):  # '-1' marks a missing salary
        return np.nan, np.nan, 0
    text = str(value).lower()
    if text.strip() in {"-1", ""} or not re.search(r"\d", text):
        return np.nan, np.nan, 0  # -1 is the raw dataset's "missing" marker
    hourly = int("per hour" in text)
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("employer provided salary:", "").replace("per hour", "")
    text = text.replace(",", "")
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return np.nan, np.nan, hourly
    lo, hi = min(nums), max(nums)
    if hourly:
        lo, hi = lo * 2080 / 1000, hi * 2080 / 1000
    elif "k" not in text and lo > 1000:  # plain dollars
        lo, hi = lo / 1000, hi / 1000
    return lo, hi, hourly


def map_job_role(title) -> str:
    t = str(title).lower()
    for role, pattern in _ROLE_RULES:
        if re.search(pattern, t):
            return role
    return "Other Role"


def _extract_state(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, "job_state", "state")
    if col is not None:
        raw = df[col].astype(str).str.strip()
    else:
        loc = _find_col(df, "Location")
        if loc is None:
            return pd.Series("Other", index=df.index)
        raw = df[loc].astype(str).str.split(",").str[-1].str.strip()
    return raw.where(raw.str.fullmatch(r"[A-Z]{2}"), "Other")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready dataframe (FEATURES + TARGET + optional EDA columns)."""
    df = df.copy()

    sal = _find_col(df, TARGET)
    if sal is None:
        est = _find_col(df, "Salary Estimate")
        if est is None:
            raise ValueError("Need either an 'avg_salary' or 'Salary Estimate' column.")
        parsed = df[est].apply(parse_salary_estimate)
        df["min_salary"] = parsed.str[0]
        df["max_salary"] = parsed.str[1]
        df[TARGET] = (df["min_salary"] + df["max_salary"]) / 2
    else:
        df[TARGET] = pd.to_numeric(df[sal], errors="coerce")

    title = _find_col(df, "Job Title", "job_title", "job_simp")
    df["Job_Role"] = df[title].apply(map_job_role) if title else "Other Role"

    size = _find_col(df, "Size")
    df["Size"] = df[size].astype(str).str.strip() if size else np.nan
    df.loc[~df["Size"].isin(SIZES), "Size"] = np.nan

    rating = _find_col(df, "Rating")
    df["Rating"] = pd.to_numeric(df[rating], errors="coerce") if rating else np.nan
    df.loc[df["Rating"] <= 0, "Rating"] = np.nan  # -1 means missing in raw data

    founded = _find_col(df, "Founded")
    founded_s = (
        pd.to_numeric(df[founded], errors="coerce")
        if founded
        else pd.Series(np.nan, index=df.index)
    )
    founded_s = founded_s.where(founded_s > 1000)
    df["Company_Age"] = (REF_YEAR - founded_s).clip(lower=0)

    state_raw = _extract_state(df)
    df["State"] = state_raw.where(state_raw.isin(TOP_STATES), "Other")

    df = df.dropna(subset=[TARGET, "Size"])
    df = df[df[TARGET] > 0].copy()
    df["Rating"] = df["Rating"].fillna(df["Rating"].median())
    df["Company_Age"] = df["Company_Age"].fillna(df["Company_Age"].median())

    loc = _find_col(df, "Location")
    if loc:
        df["City"] = df[loc].astype(str).str.split(",").str[0].str.strip()
    keep = FEATURES + [TARGET] + (["City"] if loc else [])
    for extra in ("Sector", "Industry"):
        col = _find_col(df, extra)
        if col and col not in keep:
            keep.append(col)
    return df[keep].reset_index(drop=True)


def load_and_clean(path: str) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))
