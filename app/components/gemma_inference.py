"""
gemma_inference.py - HF Inference API version, optimized for demo.
"""
import json
import os
import re
import requests

SYSTEM_PROMPT = """You are TrialLens, a clinical trial navigator.
Help patients understand if they may qualify for clinical trials.
Explain in plain simple language. No medical jargon.
Respond ONLY with valid JSON — no extra text before or after."""

# Using a reliable, always-available HF model
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"


def build_prompt(diagnosis, age, sex, country, trials, lang_code):
    lang_instruction = {
        "en": "Respond in English.",
        "pt": "Responda em Português.",
        "es": "Responde en Español.",
        "fr": "Répondez en Français.",
    }.get(lang_code, "Respond in English.")

    trials_str = ""
    for i, t in enumerate(trials[:3], 1):
        eligibility = t.get("eligibility_raw", "")
        if isinstance(eligibility, dict):
            inc = " | ".join(t.get("eligibility_structured", {}).get("inclusion", [])[:3])
            eligibility = f"Inclusion: {inc}"
        trials_str += (
            f"\nTrial {i}: {t.get('title', 'Unknown')[:80]} ({t.get('nct_id', '')})\n"
            f"Eligibility: {str(eligibility)[:300]}\n"
            f"URL: {t.get('url', '')}\n"
        )

    return f"""<s>[INST] {lang_instruction}

Patient: {diagnosis[:150]}. Age: {age}. Sex: {sex}. Country: {country}.

Analyze these 3 clinical trials for this patient.
Return ONLY this JSON (no other text whatsoever):
{{
  "trials": [
    {{
      "nct_id": "NCT...",
      "title": "trial title here",
      "match_label": "LIKELY",
      "match_reason": "1-2 plain sentences explaining why patient may qualify.",
      "key_criteria": ["Must be HER2-positive", "Age 18-75"],
      "plain_summary": "One plain sentence about what this trial tests.",
      "phase": "Phase 2",
      "status": "RECRUITING",
      "location": "United States",
      "url": "https://clinicaltrials.gov/study/NCT..."
    }}
  ],
  "doctor_questions": [
    "Do I qualify for any clinical trials given my diagnosis?",
    "What are the risks of joining a trial?",
    "How would a trial affect my current treatment?",
    "Will I need to travel for the trial?",
    "What happens if I need to leave the trial early?"
  ]
}}

Trials:
{trials_str}
[/INST]"""


def _call_hf_api(prompt, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 900,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "")
    return str(data)


def _call_ollama(prompt, model):
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_ctx": 4096, "num_predict": 900},
    )
    return response["message"]["content"].strip()


def _parse_response(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found. Raw: {raw[:200]}")
    parsed = json.loads(json_match.group())
    return parsed.get("trials", []), parsed.get("doctor_questions", [])


def match_and_summarize(diagnosis, age, sex, country, lab_data, trials, lang_code, settings):
    hf_token = settings.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    ollama_model = settings.get("OLLAMA_MODEL", "gemma3:4b")
    prompt = build_prompt(diagnosis, age, sex, country, trials, lang_code)
    raw = ""
    try:
        if hf_token:
            print("Using HF Inference API...")
            raw = _call_hf_api(prompt, hf_token)
            print(f"HF response received ({len(raw)} chars)")
        else:
            print("No HF_TOKEN — using local Ollama...")
            raw = _call_ollama(prompt, ollama_model)

        result_trials, doctor_questions = _parse_response(raw)

        if result_trials and doctor_questions:
            result_trials[0]["doctor_questions"] = doctor_questions

        order = {"LIKELY": 0, "POSSIBLE": 1, "UNLIKELY": 2}
        result_trials.sort(key=lambda t: order.get(t.get("match_label", "UNLIKELY"), 2))
        return result_trials

    except Exception as e:
        print(f"Inference error: {e}")
        if raw:
            print(f"Raw (first 400): {raw[:400]}")
        return _fallback_results(trials)


def _fallback_results(trials):
    return [
        {
            "nct_id": t.get("nct_id", ""),
            "title": t.get("title", "Unknown trial"),
            "match_label": "POSSIBLE",
            "match_reason": "Please review eligibility criteria with your doctor.",
            "key_criteria": [],
            "plain_summary": str(t.get("eligibility_raw", ""))[:150],
            "phase": str(t.get("phase", "N/A")),
            "status": t.get("status", "N/A"),
            "location": "",
            "url": t.get("url", ""),
            "doctor_questions": [
                "Do I qualify for any clinical trials given my diagnosis?",
                "What are the main risks of joining a trial?",
                "How would a trial affect my current treatment?",
                "Will I need to travel frequently?",
                "What happens if I need to leave the trial?",
            ],
        }
        for t in trials[:3]
    ]
