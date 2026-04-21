"""
gemma_inference.py
Cloud: Anthropic API (fast, reliable)
Local: Ollama fallback
"""
import json
import os
import re
import requests


def build_messages(diagnosis, age, sex, country, trials, lang_code):
    lang = {"en":"English","pt":"Portuguese","es":"Spanish","fr":"French"}.get(lang_code,"English")

    trials_str = ""
    for i, t in enumerate(trials[:2], 1):
        elig = t.get("eligibility_raw","")
        if isinstance(elig, dict):
            elig = " | ".join(t.get("eligibility_structured",{}).get("inclusion",[])[:3])
        phase = t.get("phase","N/A")
        if isinstance(phase, list):
            phase = phase[0].replace("PHASE","Phase ").title() if phase else "N/A"
        locs = t.get("locations",[])
        location = locs[0].get("country","United States") if locs else "United States"
        trials_str += (
            f"Trial {i}:\n"
            f"  NCT ID: {t.get('nct_id','')}\n"
            f"  Title: {t.get('title','')[:80]}\n"
            f"  Phase: {phase}\n"
            f"  Location: {location}\n"
            f"  URL: {t.get('url','')}\n"
            f"  Eligibility: {str(elig)[:250]}\n\n"
        )

    system = """You are TrialLens, a clinical trial navigator. 
You help cancer patients understand if they may qualify for clinical trials.
Always explain in plain, compassionate language. No medical jargon.
You MUST respond with valid JSON only — no prose, no markdown, no explanation."""

    user = f"""Analyze these clinical trials for a patient. Respond in {lang}.

Patient:
- Diagnosis: {diagnosis[:150]}
- Age: {age}, Sex: {sex}, Country: {country}

{trials_str}

Respond with ONLY this JSON (fill in all fields with real content):
{{
  "trials": [
    {{
      "nct_id": "use the real NCT ID from above",
      "title": "use the real title from above",
      "match_label": "LIKELY or POSSIBLE or UNLIKELY",
      "match_reason": "2-3 sentences in {lang} explaining why this patient may or may not qualify",
      "key_criteria": ["real criterion 1 from eligibility text", "real criterion 2"],
      "plain_summary": "1 sentence in {lang} describing what this trial is testing",
      "phase": "use the real phase from above",
      "status": "RECRUITING",
      "location": "use the real location from above",
      "url": "use the real URL from above"
    }}
  ],
  "doctor_questions": [
    "Specific question 1 about this trial in {lang}?",
    "Specific question 2 about eligibility in {lang}?",
    "Specific question 3 about side effects in {lang}?",
    "Specific question 4 about logistics in {lang}?",
    "Specific question 5 about what happens next in {lang}?"
  ]
}}"""

    return system, user


def _call_anthropic(system, user, api_key):
    """Call Anthropic API — fast and reliable for cloud deployment."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1200,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def _call_ollama(system, user, model):
    """Local Ollama fallback."""
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1, "num_ctx": 3072, "num_predict": 900},
    )
    return resp["message"]["content"].strip()


def _parse_response(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for m in re.finditer(r'\{', raw):
        candidate = raw[m.start():]
        depth = 0
        end = 0
        for i, ch in enumerate(candidate):
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            if depth == 0 and i > 0:
                end = i + 1
                break
        if not end:
            continue
        try:
            parsed = json.loads(candidate[:end])
            if 'trials' in parsed and parsed['trials']:
                return parsed.get("trials", []), parsed.get("doctor_questions", [])
        except:
            continue
    raise ValueError(f"No valid JSON. Raw: {raw[:200]}")


def match_and_summarize(diagnosis, age, sex, country, lab_data, trials, lang_code, settings):
    anthropic_key = settings.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    ollama_model = settings.get("OLLAMA_MODEL", "gemma3:4b")
    raw = ""

    system, user = build_messages(diagnosis, age, sex, country, trials, lang_code)

    # Try Anthropic API first (works on Streamlit Cloud)
    if anthropic_key:
        try:
            print("Using Anthropic API (Claude Haiku)...")
            raw = _call_anthropic(system, user, anthropic_key)
            print(f"Anthropic: {len(raw)} chars | {raw[:60]}")
            result_trials, doctor_questions = _parse_response(raw)
            if result_trials:
                if doctor_questions:
                    result_trials[0]["doctor_questions"] = doctor_questions
                _sort(result_trials)
                print(f"Anthropic success: {len(result_trials)} trial(s)")
                return result_trials
        except Exception as e:
            print(f"Anthropic failed: {e}")

    # Ollama fallback (works locally)
    try:
        print("Using local Ollama...")
        raw = _call_ollama(system, user, ollama_model)
        print(f"Ollama: {len(raw)} chars | {raw[:60]}")
        result_trials, doctor_questions = _parse_response(raw)
        if result_trials:
            if doctor_questions:
                result_trials[0]["doctor_questions"] = doctor_questions
            _sort(result_trials)
            print(f"Ollama success: {len(result_trials)} trial(s)")
            return result_trials
    except Exception as e:
        print(f"Ollama failed: {e}")
        if raw:
            print(f"Raw: {raw[:300]}")

    return _fallback_results(trials)


def _sort(result_trials):
    order = {"LIKELY": 0, "POSSIBLE": 1, "UNLIKELY": 2}
    result_trials.sort(key=lambda t: order.get(t.get("match_label", "UNLIKELY"), 2))


def _fallback_results(trials):
    return [{
        "nct_id": t.get("nct_id", ""),
        "title": t.get("title", "Unknown trial"),
        "match_label": "POSSIBLE",
        "match_reason": "Your diagnosis may match this trial. Please discuss with your doctor.",
        "key_criteria": [],
        "plain_summary": "See ClinicalTrials.gov for full trial details.",
        "phase": str(t.get("phase", "N/A")),
        "status": t.get("status", "N/A"),
        "location": "",
        "url": t.get("url", ""),
        "doctor_questions": [
            "Do I qualify for any clinical trials given my diagnosis?",
            "What are the main risks of joining a clinical trial?",
            "How would participating affect my current treatment?",
            "How often would I need to visit the trial site?",
            "What happens to my care if I withdraw from the trial?",
        ],
    } for t in trials[:3]]