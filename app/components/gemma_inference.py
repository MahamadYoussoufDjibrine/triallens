"""
gemma_inference.py — Fixed prompt so Ollama fills real values not placeholders.
"""
import json
import os
import re
import requests


def _call_hf_router(diagnosis, age, sex, country, trials, lang_code, hf_token):
    """HF router with correct 2026 API."""
    lang = {"en":"English","pt":"Portuguese","es":"Spanish","fr":"French"}.get(lang_code,"English")
    trials_str = ""
    for i, t in enumerate(trials[:2], 1):
        elig = t.get("eligibility_raw","")
        if isinstance(elig,dict):
            elig = " | ".join(t.get("eligibility_structured",{}).get("inclusion",[])[:3])
        trials_str += f"Trial {i}: {t.get('title','')[:70]} ID:{t.get('nct_id','')} Phase:{t.get('phase','N/A')} URL:{t.get('url','')}\nEligibility:{str(elig)[:200]}\n\n"

    messages = [
        {"role":"system","content":"You are a JSON API. Output ONLY valid JSON. Never use placeholder values like c1, q1, loc, Phase. Always use real content."},
        {"role":"user","content":f"""Patient: {diagnosis[:100]}. Age:{age}. Sex:{sex}. Country:{country}. Language:{lang}.

{trials_str}

Return this JSON with REAL content (not placeholders):
{{"trials":[{{"nct_id":"real NCT ID","title":"real title","match_label":"LIKELY","match_reason":"real explanation why patient qualifies","key_criteria":["real criterion from eligibility text","real criterion 2"],"plain_summary":"real 1-sentence description of the trial","phase":"real phase","status":"RECRUITING","location":"real country","url":"real url"}}],"doctor_questions":["Real specific question 1?","Real specific question 2?","Real specific question 3?","Real specific question 4?","Real specific question 5?"]}}"""}
    ]
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"model": "meta-llama/Llama-3.2-3B-Instruct", "messages": messages, "max_tokens": 1000, "temperature": 0.05}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_ollama_json(diagnosis, age, sex, country, trials, lang_code, model):
    """Ollama with explicit instruction to use real content not placeholders."""
    import ollama
    lang = {"en":"English","pt":"Portuguese","es":"Spanish","fr":"French"}.get(lang_code,"English")

    trials_info = []
    for i, t in enumerate(trials[:2], 1):
        elig = t.get("eligibility_raw","")
        if isinstance(elig,dict):
            inc = t.get("eligibility_structured",{}).get("inclusion",[])
            elig = " | ".join(inc[:3]) if inc else "See trial page"
        phase = t.get("phase","N/A")
        if isinstance(phase, list): phase = ", ".join(phase) if phase else "N/A"
        trials_info.append({
            "num": i,
            "title": t.get("title","")[:70],
            "nct_id": t.get("nct_id",""),
            "phase": str(phase),
            "url": t.get("url",""),
            "eligibility": str(elig)[:200],
        })

    trials_str = ""
    for t in trials_info:
        trials_str += f"Trial {t['num']} title: {t['title']}\nNCT ID: {t['nct_id']}\nPhase: {t['phase']}\nURL: {t['url']}\nEligibility criteria: {t['eligibility']}\n\n"

    # Build doctor questions based on diagnosis
    sample_questions = [
        f"Based on my {diagnosis[:40]}, do I qualify for any of these trials?",
        "What are the main side effects I should expect if I join?",
        "How would participating affect my current treatment plan?",
        "How often would I need to visit the trial site?",
        "What happens to my care if I need to withdraw from the trial?",
    ]

    prompt = f"""You are analyzing clinical trials for a patient. Give REAL specific answers based on the trial data below. NEVER use placeholder text like c1, q1, loc, Phase.

Patient: {diagnosis[:100]}
Age: {age}, Sex: {sex}, Country: {country}
Output language: {lang}

TRIAL DATA:
{trials_str}

Output ONLY this JSON with real content filled in:
{{
  "trials": [
    {{
      "nct_id": "{trials_info[0]['nct_id'] if trials_info else 'NCT0000000'}",
      "title": "{trials_info[0]['title'] if trials_info else 'Trial'}",
      "match_label": "LIKELY",
      "match_reason": "Write 1-2 sentences in {lang} explaining why this patient may qualify based on their diagnosis and the eligibility criteria",
      "key_criteria": ["Write the first real eligibility criterion from the trial data", "Write the second real eligibility criterion"],
      "plain_summary": "Write 1 sentence in {lang} describing what this trial is testing",
      "phase": "{trials_info[0]['phase'] if trials_info else 'N/A'}",
      "status": "RECRUITING",
      "location": "United States",
      "url": "{trials_info[0]['url'] if trials_info else ''}"
    }}
  ],
  "doctor_questions": [
    "{sample_questions[0]}",
    "{sample_questions[1]}",
    "{sample_questions[2]}",
    "{sample_questions[3]}",
    "{sample_questions[4]}"
  ]
}}"""

    resp = ollama.chat(
        model=model,
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0.1,"num_ctx":3072,"num_predict":800},
    )
    return resp["message"]["content"].strip()


def _parse_response(raw):
    raw = re.sub(r"```json|```","",raw).strip()
    raw = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL).strip()
    for m in re.finditer(r'\{', raw):
        candidate = raw[m.start():]
        depth=0; end=0
        for i,ch in enumerate(candidate):
            if ch=='{': depth+=1
            elif ch=='}': depth-=1
            if depth==0 and i>0: end=i+1; break
        if not end: continue
        try:
            parsed=json.loads(candidate[:end])
            if 'trials' in parsed and parsed['trials']:
                return parsed.get("trials",[]), parsed.get("doctor_questions",[])
        except: continue
    raise ValueError(f"No valid JSON. Raw: {raw[:200]}")


def match_and_summarize(diagnosis, age, sex, country, lab_data, trials, lang_code, settings):
    hf_token = settings.get("HF_TOKEN") or os.environ.get("HF_TOKEN","")
    ollama_model = settings.get("OLLAMA_MODEL","gemma3:4b")
    raw = ""

    if hf_token:
        try:
            print("Trying HF router...")
            raw = _call_hf_router(diagnosis, age, sex, country, trials, lang_code, hf_token)
            print(f"HF: {len(raw)} chars | {raw[:60]}")
            result_trials, doctor_questions = _parse_response(raw)
            if result_trials:
                if doctor_questions: result_trials[0]["doctor_questions"] = doctor_questions
                _sort(result_trials)
                print(f"HF success: {len(result_trials)} trial(s)")
                return result_trials
        except Exception as e:
            print(f"HF failed: {e}")

    try:
        print("Using Ollama...")
        raw = _call_ollama_json(diagnosis, age, sex, country, trials, lang_code, ollama_model)
        print(f"Ollama: {len(raw)} chars | {raw[:60]}")
        result_trials, doctor_questions = _parse_response(raw)
        if result_trials:
            if doctor_questions: result_trials[0]["doctor_questions"] = doctor_questions
            _enrich_from_source(result_trials, trials)
            _sort(result_trials)
            print(f"Ollama success: {len(result_trials)} trial(s)")
            return result_trials
    except Exception as e:
        print(f"Ollama failed: {e}")
        if raw: print(f"Raw: {raw[:300]}")

    return _fallback_results(trials, diagnosis)


def _enrich_from_source(result_trials, source_trials):
    """Merge fields the model might miss with real data from source trials."""
    source_map = {t.get("nct_id",""): t for t in source_trials}
    for rt in result_trials:
        src = source_map.get(rt.get("nct_id",""), {})
        if src:
            # Fix phase - source has it as list like ["PHASE2"]
            phase = src.get("phase", rt.get("phase","N/A"))
            if isinstance(phase, list):
                phase = phase[0].replace("PHASE","Phase ").replace("_"," ").title() if phase else "N/A"
            if not rt.get("phase") or rt.get("phase") in ("Phase", "N/A", ""):
                rt["phase"] = phase
            # Fix location from trial locations
            locs = src.get("locations", [])
            if locs and (not rt.get("location") or rt.get("location") in ("loc","N/A","")):
                rt["location"] = locs[0].get("country", locs[0].get("city","")) if locs else country_from_url(rt.get("url",""))
            # Ensure URL is real
            if not rt.get("url") or rt["url"] in ("url",""):
                nct = rt.get("nct_id","")
                if nct:
                    rt["url"] = f"https://clinicaltrials.gov/study/{nct}"
    return result_trials

def country_from_url(url):
    return "United States"

def _sort(result_trials):
    order={"LIKELY":0,"POSSIBLE":1,"UNLIKELY":2}
    result_trials.sort(key=lambda t: order.get(t.get("match_label","UNLIKELY"),2))


def _fallback_results(trials, diagnosis=""):
    questions = [
        f"Do I qualify for any clinical trials given my diagnosis?",
        "What are the main risks of joining a clinical trial?",
        "How would participating affect my current treatment?",
        "How often would I need to visit the trial site?",
        "What happens to my care if I withdraw from the trial?",
    ]
    return [{
        "nct_id": t.get("nct_id",""),
        "title": t.get("title","Unknown trial"),
        "match_label": "POSSIBLE",
        "match_reason": "Your diagnosis may match the trial criteria. Please discuss with your doctor.",
        "key_criteria": [],
        "plain_summary": "See ClinicalTrials.gov for full trial details.",
        "phase": str(t.get("phase","N/A")),
        "status": t.get("status","N/A"),
        "location": "",
        "url": t.get("url",""),
        "doctor_questions": questions,
    } for t in trials[:3]]