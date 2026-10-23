"""Eight EDA charts.  Run `python -m src.eda` to save PNGs to reports/figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import SIZES, TARGET, load_and_clean

sns.set_theme(style="whitegrid")
LABEL = "Average salary ($K)"


def salary_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[TARGET], bins=30, kde=True, ax=ax, color="#4C72B0")
    med = df[TARGET].median()
    ax.axvline(med, color="red", ls="--", label=f"Median {med:.0f}K")
    ax.set(title="Salary distribution", xlabel=LABEL)
    ax.legend()
    return fig


def salary_by_role(df):
    order = df.groupby("Job_Role")[TARGET].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=df, x="Job_Role", y=TARGET, order=order, ax=ax)
    ax.set(title="Salary by job role", xlabel="", ylabel=LABEL)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return fig


def salary_by_size(df):
    order = [s for s in SIZES if s in set(df["Size"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df, x="Size", y=TARGET, order=order, errorbar=None, ax=ax, color="#55A868")
    ax.set(title="Mean salary by company size", xlabel="", ylabel=LABEL)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return fig


def salary_by_state(df):
    order = df.groupby("State")[TARGET].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="State", y=TARGET, order=order, errorbar=None, ax=ax, color="#C44E52")
    ax.set(title="Mean salary by state", xlabel="State", ylabel=LABEL)
    return fig


def rating_vs_salary(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x="Rating", y=TARGET, alpha=0.4, ax=ax)
    sns.regplot(data=df, x="Rating", y=TARGET, scatter=False, color="red", ax=ax)
    ax.set(title="Company rating vs salary", ylabel=LABEL)
    return fig


def age_vs_salary(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x="Company_Age", y=TARGET, hue="Job_Role", alpha=0.5, ax=ax)
    ax.set(title="Company age vs salary", xlabel="Company age (years)", ylabel=LABEL)
    ax.legend(fontsize=7, loc="upper right")
    return fig


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df[["Rating", "Company_Age", TARGET]].corr(), annot=True,
                cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Numeric correlations")
    return fig


def salary_by_city(df, min_postings=15):
    """Median salary for the cities with enough postings (brief: SF vs Austin vs NY)."""
    counts = df["City"].value_counts()
    cities = counts[counts >= min_postings].index
    sub = df[df["City"].isin(cities)]
    order = sub.groupby("City")[TARGET].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=sub, x="City", y=TARGET, order=order, estimator="median",
                errorbar=None, ax=ax, color="#8172B3")
    ax.set(title=f"Median salary by city (cities with {min_postings}+ postings)",
           xlabel="", ylabel=LABEL)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig


PLOTS = {
    "Salary distribution": salary_distribution,
    "By job role": salary_by_role,
    "By company size": salary_by_size,
    "By state": salary_by_state,
    "By city": salary_by_city,
    "Rating vs salary": rating_vs_salary,
    "Company age vs salary": age_vs_salary,
    "Correlation heatmap": correlation_heatmap,
}

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/glassdoor_jobs.csv")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    data = load_and_clean(a.data)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name, fn in PLOTS.items():
        fig = fn(data)
        path = Path(a.out) / (name.lower().replace(" ", "_") + ".png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)
