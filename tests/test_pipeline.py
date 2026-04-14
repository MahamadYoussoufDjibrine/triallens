"""
tests/test_pipeline.py
Core unit tests for the TrialLens pipeline.
Run: python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def settings():
    return {
        "RAW_DATA_DIR": "data/raw",
        "PROCESSED_DATA_DIR": "data/processed",
        "VECTORSTORE_DIR": "data/vectorstore",
        "MODEL_DIR": "models/lora_weights",
        "OLLAMA_MODEL": "gemma4:27b",
        "OLLAMA_OCR_MODEL": "gemma4:4b",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "CLINICALTRIALS_BASE_URL": "https://clinicaltrials.gov/api/v2",
        "MAX_TRIALS_TO_RETRIEVE": 10,
        "TOP_K_VECTOR_RESULTS": 5,
        "SUPPORTED_LANGUAGES": ["en", "pt", "es", "fr"],
    }


@pytest.fixture
def sample_trial():
    return {
        "nct_id": "NCT12345678",
        "title": "Phase 2 Study of Trastuzumab in HER2-Positive Breast Cancer",
        "summary": "This study evaluates trastuzumab efficacy in patients with HER2-positive breast cancer.",
        "status": "RECRUITING",
        "phase": ["PHASE2"],
        "conditions": ["Breast Cancer", "HER2-Positive Breast Cancer"],
        "eligibility_raw": "Inclusion Criteria:\n- Age 18-75\n- HER2-positive confirmed\n- ECOG 0-1\n\nExclusion Criteria:\n- Prior trastuzumab therapy\n- Active CNS metastases",
        "eligibility_structured": {
            "inclusion": ["Age 18-75", "HER2-positive confirmed by IHC or FISH", "ECOG performance status 0-1"],
            "exclusion": ["Prior trastuzumab therapy", "Active CNS metastases"]
        },
        "age_range": {"min_age": "18 Years", "max_age": "75 Years", "sex": "FEMALE", "std_ages": ["ADULT", "OLDER_ADULT"]},
        "locations": [{"country": "Brazil", "city": "São Paulo", "facility": "Hospital das Clínicas"}],
        "url": "https://clinicaltrials.gov/study/NCT12345678",
        "interventions": [{"type": "Drug", "name": "Trastuzumab"}],
    }


@pytest.fixture
def sample_patient():
    return {
        "diagnosis": "Stage 2 HER2-positive breast cancer, diagnosed 3 months ago",
        "age": 42,
        "sex": "Female",
        "country": "Brazil",
        "lab_data": {"her2_status": 3.0, "ecog_score": 1.0},
    }


# ─────────────────────────────────────────────
# preprocess_trials tests
# ─────────────────────────────────────────────

class TestPreprocess:
    def test_extract_eligibility_sections_inclusion_exclusion(self):
        from scripts.preprocess_trials import extract_eligibility_sections
        raw = """Inclusion Criteria:
- Age 18 or older
- Confirmed HER2-positive

Exclusion Criteria:
- Prior chemotherapy within 6 months
- Active infection"""
        result = extract_eligibility_sections(raw)
        assert len(result["inclusion"]) == 2
        assert len(result["exclusion"]) == 2
        assert "Age 18 or older" in result["inclusion"]

    def test_extract_eligibility_empty_text(self):
        from scripts.preprocess_trials import extract_eligibility_sections
        result = extract_eligibility_sections("")
        assert result == {"inclusion": [], "exclusion": []}

    def test_extract_eligibility_no_sections(self):
        from scripts.preprocess_trials import extract_eligibility_sections
        result = extract_eligibility_sections("Patients must be over 18 years old.")
        # No section headers — returns empty (stricter parsing)
        assert isinstance(result["inclusion"], list)
        assert isinstance(result["exclusion"], list)

    def test_process_study_returns_none_for_missing_nct(self):
        from scripts.preprocess_trials import process_study
        result = process_study({"protocolSection": {}})
        assert result is None

    def test_process_study_valid(self, sample_trial):
        from scripts.preprocess_trials import process_study
        # Build a raw study object matching the API shape
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test Trial"},
                "descriptionModule": {"briefSummary": "A test study."},
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n- Age 18+\n\nExclusion Criteria:\n- Pregnancy",
                    "minimumAge": "18 Years",
                    "maximumAge": "N/A",
                    "sex": "ALL",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL"},
                "conditionsModule": {"conditions": ["Cancer"]},
                "armsInterventionsModule": {"interventions": []},
                "contactsLocationsModule": {"locations": []},
            }
        }
        result = process_study(study)
        assert result is not None
        assert result["nct_id"] == "NCT12345678"
        assert result["status"] == "RECRUITING"
        assert len(result["eligibility_structured"]["inclusion"]) > 0


# ─────────────────────────────────────────────
# formatter tests
# ─────────────────────────────────────────────

class TestFormatter:
    def test_format_results_fills_missing_fields(self):
        from app.components.formatter import format_results
        results = [{"nct_id": "NCT123", "title": "Test"}]
        out = format_results(results, "en")
        assert out[0]["match_label"] == "POSSIBLE"
        assert out[0]["match_reason"] != ""
        assert isinstance(out[0]["key_criteria"], list)
        assert len(out[0]["doctor_questions"]) == 5

    def test_format_results_portuguese_questions(self):
        from app.components.formatter import format_results
        results = [{}]
        out = format_results(results, "pt")
        # Portuguese questions should not be in English
        q = out[0]["doctor_questions"][0]
        assert "diagnóstico" in q or "qualificar" in q or "tratamento" in q

    def test_format_results_preserves_existing_fields(self):
        from app.components.formatter import format_results
        results = [{
            "nct_id": "NCT999",
            "title": "Real Trial",
            "match_label": "LIKELY",
            "match_reason": "Patient matches all criteria.",
            "key_criteria": ["Must be HER2+", "Age 18-70"],
            "plain_summary": "A trial about breast cancer.",
            "doctor_questions": ["What is the dosage?"],
        }]
        out = format_results(results, "en")
        assert out[0]["match_label"] == "LIKELY"
        assert out[0]["match_reason"] == "Patient matches all criteria."
        assert out[0]["key_criteria"] == ["Must be HER2+", "Age 18-70"]


# ─────────────────────────────────────────────
# gemma_inference tests (mocked — no Ollama needed)
# ─────────────────────────────────────────────

class TestGemmaInference:
    def test_build_prompt_contains_patient_info(self, sample_patient):
        from app.components.gemma_inference import build_prompt
        prompt = build_prompt(
            diagnosis=sample_patient["diagnosis"],
            age=sample_patient["age"],
            sex=sample_patient["sex"],
            country=sample_patient["country"],
            lab_data=sample_patient["lab_data"],
            trials=[],
            lang_code="en",
        )
        assert "42" in prompt
        assert "Brazil" in prompt
        assert "HER2-positive" in prompt
        assert "her2_status" in prompt

    def test_build_prompt_portuguese_instruction(self, sample_patient):
        from app.components.gemma_inference import build_prompt
        prompt = build_prompt(
            diagnosis=sample_patient["diagnosis"],
            age=42, sex="Female", country="Brazil",
            lab_data={}, trials=[], lang_code="pt",
        )
        assert "Português" in prompt

    @patch("app.components.gemma_inference.ollama")
    def test_match_and_summarize_handles_json_response(self, mock_ollama, settings, sample_patient, sample_trial):
        from app.components.gemma_inference import match_and_summarize

        mock_response = {
            "message": {
                "content": json.dumps({
                    "trials": [{
                        "nct_id": "NCT12345678",
                        "title": "Phase 2 Study of Trastuzumab",
                        "match_label": "LIKELY",
                        "match_reason": "Patient is HER2-positive and within age range.",
                        "key_criteria": ["HER2-positive", "Age 18-75", "ECOG 0-1"],
                        "plain_summary": "This trial tests trastuzumab for HER2+ breast cancer.",
                        "phase": "PHASE2",
                        "status": "RECRUITING",
                        "location": "São Paulo, Brazil",
                        "url": "https://clinicaltrials.gov/study/NCT12345678",
                    }],
                    "doctor_questions": [
                        "Do I qualify for this trial?",
                        "What are the side effects?",
                        "How often do I need to visit?",
                        "Will this affect my current treatment?",
                        "What happens if I withdraw?",
                    ]
                })
            }
        }
        mock_ollama.chat.return_value = mock_response

        results = match_and_summarize(
            diagnosis=sample_patient["diagnosis"],
            age=42, sex="Female", country="Brazil",
            lab_data={},
            trials=[sample_trial],
            lang_code="en",
            settings=settings,
        )

        assert len(results) == 1
        assert results[0]["match_label"] == "LIKELY"
        assert results[0]["nct_id"] == "NCT12345678"
        assert len(results[0]["doctor_questions"]) == 5

    @patch("app.components.gemma_inference.ollama")
    def test_match_and_summarize_fallback_on_bad_json(self, mock_ollama, settings, sample_trial):
        from app.components.gemma_inference import match_and_summarize

        mock_ollama.chat.return_value = {"message": {"content": "not valid json at all"}}

        results = match_and_summarize(
            diagnosis="breast cancer", age=42, sex="Female", country="Brazil",
            lab_data={}, trials=[sample_trial], lang_code="en", settings=settings,
        )
        # Should fall back gracefully — not crash
        assert isinstance(results, list)

    @patch("app.components.gemma_inference.ollama")
    def test_results_sorted_likely_first(self, mock_ollama, settings):
        from app.components.gemma_inference import match_and_summarize

        mock_ollama.chat.return_value = {
            "message": {
                "content": json.dumps({
                    "trials": [
                        {"nct_id": "NCT001", "title": "A", "match_label": "UNLIKELY",
                         "match_reason": "", "key_criteria": [], "plain_summary": "",
                         "phase": "", "status": "", "location": "", "url": ""},
                        {"nct_id": "NCT002", "title": "B", "match_label": "LIKELY",
                         "match_reason": "", "key_criteria": [], "plain_summary": "",
                         "phase": "", "status": "", "location": "", "url": ""},
                        {"nct_id": "NCT003", "title": "C", "match_label": "POSSIBLE",
                         "match_reason": "", "key_criteria": [], "plain_summary": "",
                         "phase": "", "status": "", "location": "", "url": ""},
                    ],
                    "doctor_questions": []
                })
            }
        }

        results = match_and_summarize(
            diagnosis="cancer", age=50, sex="Female", country="USA",
            lab_data={}, trials=[{}, {}, {}], lang_code="en", settings=settings,
        )
        labels = [r["match_label"] for r in results]
        assert labels == ["LIKELY", "POSSIBLE", "UNLIKELY"]


# ─────────────────────────────────────────────
# ClinicalTrials.gov API integration test
# (uses real network — skipped in CI if offline)
# ─────────────────────────────────────────────

class TestClinicalTrialsAPI:
    @pytest.mark.network
    def test_api_returns_results_for_breast_cancer(self):
        import requests
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "query.cond": "breast cancer",
                "filter.overallStatus": "RECRUITING",
                "pageSize": 5,
                "format": "json",
            },
            timeout=15,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "studies" in data
        assert len(data["studies"]) > 0
        # Each study should have an NCT ID
        for study in data["studies"]:
            nct = study["protocolSection"]["identificationModule"]["nctId"]
            assert nct.startswith("NCT")
