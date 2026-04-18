"""
gemma_inference.py
Sends candidate trials + patient profile to Gemma 4B (via Ollama) for
eligibility matching, plain-language summarization, and Q&A list generation.
Optimized for 16GB RAM / local CPU inference.
"""

import json
import re

import ollama


SYSTEM_PROMPT = """You are TrialLens, a clinical trial navigator.
Help patients understand if they may qualify for clinical trials.
Always explain in plain, simple language. Never use medical jargon.
Respond ONLY with valid JSON — no extra text, no markdown."""


def build_prompt(diagnosis: str, age: int, sex: str, country: str,
                 lab_data: dict, trials: list[dict], lang_code: str) -> str:

    lang_instruction = {
        "en": "Respond in English.",
        "pt": "Responda em Português.",
        "es": "Responde en Español.",
        "fr": "Répondez en Français.",
    }.get(lang_code, "Respond in English.")

    # Only use top 3 trials and trim eligibility text aggressively for 4B model
    trials_str = ""
    for i, t in enumerate(trials[:3], 1):
        eligibility = t.get("eligibility_raw", "")
        if isinstance(eligibility, dict):
            inc = " | ".join(t.get("eligibility_structured", {}).get("inclusion", [])[:4])
            eligibility = f"Inclusion: {inc}"
        # Hard cap at 400 chars per trial to fit in 4B context
        trials_str += (
            f"\nTrial {i}: {t.get('title', 'Unknown')[:80]} ({t.get('nct_id', '')})\n"
            f"Eligibility: {str(eligibility)[:400]}\n"
            f"URL: {t.get('url', '')}\n"
        )

    return f"""{lang_instruction}

Patient: {diagnosis[:200]}. Age: {age}. Sex: {sex}. Country: {country}.

Analyze these 3 clinical trials for this patient.
Return ONLY this JSON (no other text):
{{
  "trials": [
    {{
      "nct_id": "NCT...",
      "title": "short title",
      "match_label": "LIKELY or POSSIBLE or UNLIKELY",
      "match_reason": "1-2 plain sentences why",
      "key_criteria": ["criterion 1", "criterion 2"],
      "plain_summary": "1 sentence about the trial",
      "phase": "Phase 1/2/3",
      "status": "RECRUITING",
      "location": "country",
      "url": "https://..."
    }}
  ],
  "doctor_questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?",
    "Question 4?",
    "Question 5?"
  ]
}}

Trials:
{trials_str}"""


def match_and_summarize(
    diagnosis: str,
    age: int,
    sex: str,
    country: str,
    lab_data: dict,
    trials: list[dict],
    lang_code: str,
    settings: dict,
) -> list[dict]:
    """
    Run Gemma 4B locally via Ollama to match and summarize trials.
    Optimized for 16GB RAM — processes 3 trials, small context window.
    """
    model = settings.get("OLLAMA_MODEL", "gemma3:4b")
    prompt = build_prompt(diagnosis, age, sex, country, lab_data, trials, lang_code)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.1,
                "num_ctx": 4096,
                "num_predict": 1024,
            },
        )

        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Find JSON object even if there's extra text around it
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group()

        parsed = json.loads(raw)
        result_trials = parsed.get("trials", [])
        doctor_questions = parsed.get("doctor_questions", [])

        if result_trials and doctor_questions:
            result_trials[0]["doctor_questions"] = doctor_questions

        order = {"LIKELY": 0, "POSSIBLE": 1, "UNLIKELY": 2}
        result_trials.sort(key=lambda t: order.get(t.get("match_label", "UNLIKELY"), 2))

        return result_trials

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}\nRaw response: {raw[:300]}")
        return _fallback_results(trials)

    except Exception as e:
        print(f"Ollama inference error: {e}")
        return _fallback_results(trials)


def _fallback_results(trials: list[dict]) -> list[dict]:
    """Return basic trial info when Ollama inference fails."""
    return [
        {
            "nct_id": t.get("nct_id", ""),
            "title": t.get("title", "Unknown trial"),
            "match_label": "POSSIBLE",
            "match_reason": "Automatic analysis unavailable. Please review eligibility criteria manually.",
            "key_criteria": [],
            "plain_summary": str(t.get("eligibility_raw", ""))[:200],
            "phase": str(t.get("phase", "N/A")),
            "status": t.get("status", "N/A"),
            "location": "",
            "url": t.get("url", ""),
            "doctor_questions": [
                "Do I qualify for any clinical trials given my diagnosis?",
                "What are the main risks and benefits of joining a trial?",
                "How would a trial affect my current treatment?",
                "Will I need to travel frequently?",
                "What happens to my care if I withdraw from the trial?",
            ],
        }
        for t in trials[:3]
    ]
