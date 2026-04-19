"""
gemma_inference.py
Primary: HF router.huggingface.co (OpenAI-compatible, correct 2026 API)
Fallback: Local Ollama
"""
import json
import os
import re
import requests


def build_messages(diagnosis, age, sex, country, trials, lang_code):
    lang = {"en":"English","pt":"Portuguese","es":"Spanish","fr":"French"}.get(lang_code,"English")

    trials_str = ""
    for i, t in enumerate(trials[:2], 1):
        elig = t.get("eligibility_raw", "")
        if isinstance(elig, dict):
            elig = " | ".join(t.get("eligibility_structured",{}).get("inclusion",[])[:3])
        trials_str += (
            f"Trial {i}: {t.get('title','')[:70]} (ID:{t.get('nct_id','')})\n"
            f"Eligibility: {str(elig)[:200]}\n"
            f"URL: {t.get('url','')}\n\n"
        )

    system = "You are a JSON API. Output ONLY valid JSON. No prose. No explanation. No markdown."
    user = f"""Patient: {diagnosis[:100]}. Age:{age}. Sex:{sex}. Country:{country}. Respond in {lang}.

Trials:
{trials_str}

Return ONLY this JSON object:
{{"trials":[{{"nct_id":"...","title":"...","match_label":"LIKELY","match_reason":"plain reason in {lang}","key_criteria":["c1","c2"],"plain_summary":"1 sentence in {lang}","phase":"Phase 2","status":"RECRUITING","location":"country","url":"..."}}],"doctor_questions":["q1 in {lang}","q2","q3","q4","q5"]}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_hf_router(messages, hf_token):
    """Use the correct 2026 HF router API (OpenAI-compatible)."""
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.05,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama_json(diagnosis, age, sex, country, trials, lang_code, model):
    """Ollama with a tight JSON-focused prompt."""
    import ollama
    lang = {"en":"English","pt":"Portuguese","es":"Spanish","fr":"French"}.get(lang_code,"English")
    trials_str = ""
    for i, t in enumerate(trials[:2], 1):
        elig = t.get("eligibility_raw","")
        if isinstance(elig,dict):
            elig = " | ".join(t.get("eligibility_structured",{}).get("inclusion",[])[:2])
        trials_str += f"T{i}: {t.get('title','')[:60]} ID:{t.get('nct_id','')} Criteria:{str(elig)[:150]} URL:{t.get('url','')}\n"

    prompt = f"""Output JSON only. No text before or after the JSON.

Patient:{diagnosis[:80]} Age:{age} Sex:{sex} Country:{country} Language:{lang}
{trials_str}

JSON:
{{"trials":[{{"nct_id":"ID","title":"title","match_label":"LIKELY","match_reason":"reason","key_criteria":["c1"],"plain_summary":"summary","phase":"Phase","status":"RECRUITING","location":"loc","url":"url"}}],"doctor_questions":["q1","q2","q3","q4","q5"]}}"""

    resp = ollama.chat(
        model=model,
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0.05,"num_ctx":2048,"num_predict":600},
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

    # Try HF router first
    if hf_token:
        try:
            print("Trying HF router (Llama-3.2-3B)...")
            messages = build_messages(diagnosis, age, sex, country, trials, lang_code)
            raw = _call_hf_router(messages, hf_token)
            print(f"HF response: {len(raw)} chars | Preview: {raw[:80]}")
            result_trials, doctor_questions = _parse_response(raw)
            if result_trials:
                if doctor_questions: result_trials[0]["doctor_questions"] = doctor_questions
                order={"LIKELY":0,"POSSIBLE":1,"UNLIKELY":2}
                result_trials.sort(key=lambda t: order.get(t.get("match_label","UNLIKELY"),2))
                print(f"HF success: {len(result_trials)} trial(s)")
                return result_trials
        except Exception as e:
            print(f"HF router failed: {e}")

    # Ollama fallback
    try:
        print("Using local Ollama...")
        raw = _call_ollama_json(diagnosis, age, sex, country, trials, lang_code, ollama_model)
        print(f"Ollama response: {len(raw)} chars | Preview: {raw[:80]}")
        result_trials, doctor_questions = _parse_response(raw)
        if result_trials:
            if doctor_questions: result_trials[0]["doctor_questions"] = doctor_questions
            order={"LIKELY":0,"POSSIBLE":1,"UNLIKELY":2}
            result_trials.sort(key=lambda t: order.get(t.get("match_label","UNLIKELY"),2))
            print(f"Ollama success: {len(result_trials)} trial(s)")
            return result_trials
    except Exception as e:
        print(f"Ollama failed: {e}")
        if raw: print(f"Raw: {raw[:300]}")

    print("Both failed — using fallback display")
    return _fallback_results(trials)


def _fallback_results(trials):
    return [{
        "nct_id": t.get("nct_id",""),
        "title": t.get("title","Unknown trial"),
        "match_label": "POSSIBLE",
        "match_reason": "Please review eligibility criteria with your doctor.",
        "key_criteria": [],
        "plain_summary": str(t.get("eligibility_raw",""))[:150],
        "phase": str(t.get("phase","N/A")),
        "status": t.get("status","N/A"),
        "location": "",
        "url": t.get("url",""),
        "doctor_questions": [
            "Do I qualify for any clinical trials given my diagnosis?",
            "What are the main risks of joining a trial?",
            "How would a trial affect my current treatment?",
            "Will I need to travel frequently?",
            "What happens if I need to leave the trial?",
        ],
    } for t in trials[:3]]