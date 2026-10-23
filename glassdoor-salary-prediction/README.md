# 💼 Glassdoor Tech Salary Predictor & AI Career Coach

Predict tech salaries from job role, company size, rating, state and company age, explore what drives pay, and ask a Gemini-powered career coach. Built with pandas, scikit-learn and Streamlit.

> Data: Glassdoor postings from 2017–2018. Estimates are a historical snapshot, not current market rates.

## Features
- **Cleaning pipeline** (`src/data_preprocessing.py`): parses salary text and hourly pay into annual $K, maps titles to roles, handles missing rating/founded values.
- **Model** (`src/train_model.py`): one-hot encoding + Random Forest in a single sklearn `Pipeline`, with hold-out and 5-fold CV metrics saved to `models/metrics.json`.
- **Uncertainty band** (`src/predict.py`): P10–P90 of individual tree predictions instead of a fixed ±%.
- **EDA** (`src/eda.py`): 8 Matplotlib/Seaborn charts (distribution, role, company size, state, rating, company age, city, correlations), shown in the app and exportable as PNGs.
- **Gemini coach**: optional context from your latest prediction, per-session call limit.

## Results (real data, 732 cleaned postings)
| Model | 5-fold CV R² | CV MAE |
|---|---|---|
| Mean baseline | 0.00 | $29.6K |
| Ridge | 0.35 | $23.0K |
| **Random Forest (deployed)** | **0.53** | **$17.7K** |

Key findings (full write-up in `notebooks/eda_and_modeling.ipynb`): Data Scientist and ML Engineer postings pay a median of about $114K versus about $63K for Data Analyst; California has the highest state median (about $121K); San Francisco is about $139K versus about $98K for New York; company size shows no clean trend; rating and company age are only weakly related to salary.

**Limitations:** ~730 postings, mostly data-science searches, from 2017–2018. Salaries are Glassdoor estimates, and there are no experience or skill features. Only 4 DevOps postings exist, so that option is flagged as unreliable in the app. Austin has only 8 postings, too few to compare.

## Project structure
```
├── app.py                  # Streamlit app (3 tabs)
├── notebooks/eda_and_modeling.ipynb  # executed analysis, findings, model comparison
├── src/                    # preprocessing, training, prediction, EDA
├── reports/figures/        # exported charts (python -m src.eda)
├── tests/                  # pytest suite (runs in CI)
├── models/model.pkl        # trained pipeline (committed for deployment)
├── data/glassdoor_jobs.csv # Glassdoor dataset
├── .streamlit/             # config + secrets template
└── requirements.txt        # scikit-learn pinned to the training version
```

## Run locally
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
python -m src.train_model          # trains and writes models/model.pkl
python -m src.eda                  # optional: saves charts to reports/figures
pytest -q
streamlit run app.py
```
Add your key to `.env` (`GEMINI_API_KEY=...`) or `.streamlit/secrets.toml`.

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (make sure `models/model.pkl` is committed).
2. share.streamlit.io → **New app** → pick the repo, branch, `app.py`.
3. **Advanced settings**: choose the same Python version you trained with, and paste `GEMINI_API_KEY = "..."` under Secrets.
4. Deploy.

If the app errors on model load, the cause is almost always a scikit-learn version mismatch: retrain in the same environment as `requirements.txt` and commit the new `model.pkl`. The committed model was trained with scikit-learn 1.8.0; if you retrain on another version, update the pin in `requirements.txt`.

## License
MIT
