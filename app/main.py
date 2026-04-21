"""
main.py — TrialLens Streamlit app
Run: streamlit run app/main.py
"""

import json
from pathlib import Path

import streamlit as st

from components.ocr_pipeline import extract_lab_data
from components.trial_retriever import retrieve_trials
from components.gemma_inference import match_and_summarize
from components.formatter import format_results

# Load settings
SETTINGS = json.loads(Path("SETTINGS.json").read_text())

# Load secrets from Streamlit Cloud or environment
import streamlit as _st_check
try:
    if hasattr(_st_check, 'secrets') and "ANTHROPIC_API_KEY" in _st_check.secrets:
        SETTINGS["ANTHROPIC_API_KEY"] = _st_check.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass
if not SETTINGS.get("ANTHROPIC_API_KEY"):
    import os
    SETTINGS["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

st.set_page_config(
    page_title="TrialLens",
    page_icon="🔬",
    layout="centered",
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("## 🔬 TrialLens")
st.markdown(
    "Find clinical trials that match *you* — explained in plain language, in your language."
)
st.divider()

# ── Language selector ────────────────────────────────────────────────────────
language = st.selectbox(
    "Output language",
    options=["English", "Português", "Español", "Français"],
    index=0,
)
lang_code = {"English": "en", "Português": "pt", "Español": "es", "Français": "fr"}[language]

# ── Patient input ────────────────────────────────────────────────────────────
st.markdown("### Your situation")

diagnosis = st.text_area(
    "Describe your diagnosis",
    placeholder="e.g. Stage 2 HER2-positive breast cancer, diagnosed 3 months ago. "
    "I am 42 years old, female, located in São Paulo, Brazil.",
    height=120,
)

lab_image = st.file_uploader(
    "Upload a lab report (optional — photo or scan)",
    type=["jpg", "jpeg", "png", "pdf"],
    help="TrialLens will read your lab values to improve matching. "
    "Your data never leaves this app.",
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=42)
with col2:
    sex = st.selectbox("Sex", ["Female", "Male", "Other / Prefer not to say"])

country = st.text_input("Country", value="Brazil")

# ── Run button ───────────────────────────────────────────────────────────────
if st.button("Find matching trials →", type="primary", use_container_width=True):
    if not diagnosis.strip():
        st.warning("Please describe your diagnosis above.")
        st.stop()

    with st.status("Reading your lab report...", expanded=True) as status:
        lab_data = {}
        if lab_image:
            st.write("Extracting values from your lab report...")
            lab_data = extract_lab_data(lab_image, SETTINGS)
            if lab_data:
                st.write(f"Found: {', '.join(lab_data.keys())}")
        else:
            st.write("No lab report uploaded — matching on diagnosis text only.")

        st.write("Searching 450,000+ clinical trials...")
        candidate_trials = retrieve_trials(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            country=country,
            lab_data=lab_data,
            settings=SETTINGS,
        )
        st.write(f"Found {len(candidate_trials)} candidate trials. Analyzing eligibility...")

        results = match_and_summarize(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            country=country,
            lab_data=lab_data,
            trials=candidate_trials,
            lang_code=lang_code,
            settings=SETTINGS,
        )
        status.update(label="Done!", state="complete")

    # ── Results ──────────────────────────────────────────────────────────────
    st.markdown(f"### Trials matched for you ({len(results)})")
    st.caption(
        "TrialLens is not a medical advisor. Always discuss results with your doctor."
    )

    for i, trial in enumerate(results, 1):
        with st.expander(f"**{i}. {trial['title']}**  —  {trial['match_label']}", expanded=(i == 1)):
            st.markdown(f"**Why you may qualify:** {trial['match_reason']}")
            st.markdown(f"**What this trial is about:** {trial['plain_summary']}")

            if trial.get("key_criteria"):
                st.markdown("**Key eligibility points:**")
                for point in trial["key_criteria"]:
                    st.markdown(f"- {point}")

            st.markdown(f"**Phase:** {trial.get('phase', 'N/A')}  |  "
                        f"**Status:** {trial.get('status', 'N/A')}  |  "
                        f"**Location:** {trial.get('location', 'N/A')}")
            st.markdown(f"[View on ClinicalTrials.gov ↗]({trial['url']})")

    # ── Doctor Q&A list ───────────────────────────────────────────────────────
    if results:
        st.divider()
        st.markdown("### Questions to ask your doctor")
        st.caption("Print or share this list before your next appointment.")
        questions = results[0].get("doctor_questions", [])
        for q in questions:
            st.markdown(f"- {q}")

        st.download_button(
            "Download as text",
            data="\n".join(f"- {q}" for q in questions),
            file_name="triallens_questions.txt",
            mime="text/plain",
        )