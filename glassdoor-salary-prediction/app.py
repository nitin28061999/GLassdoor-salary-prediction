import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st

from src.data_preprocessing import ROLES, SIZES, STATES, load_and_clean
from src.eda import PLOTS
from src.predict import predict_with_interval

MODEL_PATH = Path("models/model.pkl")
DATA_PATH = Path("data/glassdoor_jobs.csv")
MAX_AI_CALLS = 10  # per browser session, protects the API quota on a public app

st.set_page_config(page_title="Glassdoor Salary Predictor & AI Coach", page_icon="💼", layout="wide")
st.title("💼 Glassdoor Tech Salary Predictor & AI Career Coach")
st.caption(
    "Trained on Glassdoor job postings from 2017–2018. Estimates reflect that snapshot, "
    "not current market rates."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_data():
    return load_and_clean(str(DATA_PATH)) if DATA_PATH.exists() else None


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file locally
        key = None
    return key or os.getenv("GEMINI_API_KEY")


model = load_model()
tab_pred, tab_eda, tab_ai = st.tabs(["📊 Salary Predictor", "📈 Market Insights", "🤖 AI Career Coach"])

# ---------------------------------------------------------------- Predictor
with tab_pred:
    st.subheader("Predict a tech salary range")
    c1, c2 = st.columns(2)
    with c1:
        job_role = st.selectbox("Job role", ROLES)
        company_size = st.selectbox("Company size", SIZES, index=3)
        rating = st.slider("Company rating", 1.0, 5.0, 3.8, 0.1)
    with c2:
        state = st.selectbox("State", STATES, help="'Other' covers all remaining states.")
        company_age = st.number_input("Company age (years)", 0, 200, 15)

    if job_role == "DevOps Engineer":
        st.warning("Only 4 DevOps postings exist in the training data, so this estimate is unreliable.")

    if st.button("Predict salary", type="primary"):
        if model is None:
            st.error("Model file `models/model.pkl` is missing. Run `python -m src.train_model`.")
        else:
            row = {"Job_Role": job_role, "Size": company_size, "Rating": rating,
                   "State": state, "Company_Age": company_age}
            mean_k, low_k, high_k = predict_with_interval(model, row)
            st.session_state["last_prediction"] = (row, mean_k, low_k, high_k)
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated average", f"${mean_k:,.0f}K")
            m2.metric("Likely low (P10)", f"${low_k:,.0f}K")
            m3.metric("Likely high (P90)", f"${high_k:,.0f}K")
            st.caption("Range = 10th–90th percentile of the Random Forest's individual trees.")

# ---------------------------------------------------------------- EDA
with tab_eda:
    data = load_data()
    if data is None:
        st.info("Add `data/glassdoor_jobs.csv` to the repo to enable the market-insights charts.")
    else:
        st.subheader("What drives tech salaries?")
        choice = st.selectbox("Chart", list(PLOTS))
        fig = PLOTS[choice](data)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"{len(data):,} cleaned postings.")

# ---------------------------------------------------------------- Gemini
with tab_ai:
    st.subheader("AI career advisor (Gemini)")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Not stored. Get one free at aistudio.google.com.")
    query = st.text_area("Ask about market trends, negotiation, or skills:")
    use_pred = st.checkbox("Include my latest prediction as context",
                           value="last_prediction" in st.session_state,
                           disabled="last_prediction" not in st.session_state)

    if st.button("Ask Gemini"):
        calls = st.session_state.get("ai_calls", 0)
        if not api_key:
            st.error("Please provide a Gemini API key.")
        elif not query.strip():
            st.warning("Please type a question.")
        elif calls >= MAX_AI_CALLS:
            st.warning("Session limit reached. Refresh the page to start a new session.")
        else:
            context = ""
            if use_pred and "last_prediction" in st.session_state:
                row, mean_k, low_k, high_k = st.session_state["last_prediction"]
                context = (f"\nCandidate context: {row}. Model estimate "
                           f"${mean_k:,.0f}K (range ${low_k:,.0f}K–${high_k:,.0f}K, 2017–18 data).\n")
            prompt = (
                "You are an expert tech career advisor and compensation analyst. "
                "Answer clearly and concisely. Note when advice depends on current market data."
                f"{context}\nQuestion: {query}"
            )
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                with st.spinner("Thinking..."):
                    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                st.session_state["ai_calls"] = calls + 1
                st.markdown("### Advisor insights")
                st.write(resp.text)
            except Exception as e:
                st.error(f"Gemini request failed: {e}")
