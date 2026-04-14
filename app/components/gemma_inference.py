"""
gemma_inference.py
Sends candidate trials + patient profile to Gemma 4 27B (via Ollama) for
eligibility matching, plain-language summarization, and Q&A list generation.
Uses Gemma 4's thinking mode for transparent reasoning.
"""

import json
import re

import ollama


SYSTEM_PROMPT = """You are TrialLens, an expert clinical trial navigator.
Your job is to help patients understand whether they may qualify for clinical trials,
and to explain complex medical eligibility criteria in plain, compassionate language.

Rules:
- Always be honest about uncertainty. Say "you may qualify" not "you qualify".
- Explain medical jargon in simple terms.
- Be warm and supportive — patients reading this are often scared.
- Never give definitive medical advice. Always recommend consulting their doctor.
- Respond in the language specified by the user."""


def build_prompt(diagnosis: str, age: int, sex: str, country: str,
                 lab_data: dict, trials: list[dict], lang_code: str) -> str:

    lab_str = ""
    if lab_data:
        lab_str = "\nLab values:\n" + "\n".join(f"  - {k}: {v}" for k, v in lab_data.items())

    trials_str = ""
    for i, t in enumerate(trials[:10], 1):
        eligibility = t.get("eligibility_raw", t.get("eligibility_structured", "Not available"))
        if isinstance(eligibility, dict):
            inc = "\n".join(f"  - {c}" for c in eligibility.get("inclusion", [])[:8])
            exc = "\n".join(f"  - {c}" for c in eligibility.get("exclusion", [])[:5])
            eligibility = f"Inclusion:\n{inc}\nExclusion:\n{exc}"
        trials_str += f"\n--- Trial {i}: {t.get('title', 'Unknown')} ({t.get('nct_id', '')}) ---\n"
        trials_str += f"Status: {t.get('status', 'N/A')} | Phase: {t.get('phase', 'N/A')}\n"
        trials_str += f"URL: {t.get('url', '')}\n"
        trials_str += f"Eligibility criteria:\n{str(eligibility)[:1500]}\n"

    lang_instruction = {
        "en": "Respond in English.",
        "pt": "Responda em Português.",
        "es": "Responde en Español.",
        "fr": "Répondez en Français.",
    }.get(lang_code, "Respond in English.")

    return f"""Patient profile:
- Diagnosis: {diagnosis}
- Age: {age}
- Sex: {sex}
- Country: {country}{lab_str}

{lang_instruction}

Here are {len(trials[:10])} candidate clinical trials. For each one:
1. Assess whether this patient likely qualifies (LIKELY / POSSIBLE / UNLIKELY)
2. Explain why in 2-3 plain sentences
3. List 2-3 key eligibility points in simple language
4. Write a 1-paragraph plain-language summary of what the trial is about

After analyzing all trials, provide a list of 5 questions the patient should ask their doctor.

Respond ONLY with a valid JSON object in this format:
{{
  "trials": [
    {{
      "nct_id": "NCT...",
      "title": "...",
      "match_label": "LIKELY / POSSIBLE / UNLIKELY",
      "match_reason": "...",
      "key_criteria": ["...", "...", "..."],
      "plain_summary": "...",
      "phase": "...",
      "status": "...",
      "location": "...",
      "url": "..."
    }}
  ],
  "doctor_questions": ["...", "...", "...", "...", "..."]
}}

Trials to analyze:
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
    Run Gemma 4 27B to match and summarize trials for a patient.
    Returns a list of enriched trial dicts sorted by match quality.
    """
    model = settings.get("OLLAMA_MODEL", "gemma4:27b")

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
                "num_ctx": 16384,
            },
        )

        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        # Handle thinking mode output (Gemma 4 wraps reasoning in <think> tags)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        parsed = json.loads(raw)
        result_trials = parsed.get("trials", [])
        doctor_questions = parsed.get("doctor_questions", [])

        # Attach doctor questions to the first result
        if result_trials:
            result_trials[0]["doctor_questions"] = doctor_questions

        # Sort: LIKELY first, then POSSIBLE, then UNLIKELY
        order = {"LIKELY": 0, "POSSIBLE": 1, "UNLIKELY": 2}
        result_trials.sort(key=lambda t: order.get(t.get("match_label", "UNLIKELY"), 2))

        return result_trials

    except json.JSONDecodeError as e:
        print(f"JSON parse error from Gemma: {e}")
        # Return raw trial metadata without enrichment as fallback
        return [
            {
                "nct_id": t.get("nct_id", ""),
                "title": t.get("title", "Unknown trial"),
                "match_label": "POSSIBLE",
                "match_reason": "Could not analyze eligibility automatically. Please review manually.",
                "key_criteria": [],
                "plain_summary": t.get("eligibility_raw", "")[:300],
                "url": t.get("url", ""),
                "doctor_questions": [],
            }
            for t in trials[:5]
        ]
    except Exception as e:
        print(f"Gemma inference error: {e}")
        return []
