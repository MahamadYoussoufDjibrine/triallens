---
license: apache-2.0
base_model: google/gemma-4-27b-it
tags:
  - gemma4
  - clinical-trials
  - health
  - lora
  - unsloth
  - peft
  - medical
  - multilingual
language:
  - en
  - pt
  - es
  - fr
datasets:
  - clinicaltrials-gov
pipeline_tag: text-generation
library_name: transformers
---

# TrialLens — Gemma 4 27B Clinical LoRA

**Plain-language clinical trial matching for cancer patients.**

Fine-tuned from `google/gemma-4-27b-it` using [Unsloth](https://github.com/unslothai/unsloth) LoRA on clinical trial eligibility criteria from ClinicalTrials.gov.

Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) · Health & Sciences Track.

---

## What this model does

Given a patient profile (diagnosis, age, sex, location, lab values) and raw clinical trial eligibility criteria, this model:

1. Assesses whether the patient likely qualifies (`LIKELY / POSSIBLE / UNLIKELY`)
2. Explains eligibility in plain language (no medical jargon)
3. Lists the 3 most important eligibility points simply
4. Summarizes what the trial is actually about
5. Generates doctor Q&A questions for the patient's next appointment

Output is available in **English, Portuguese, Spanish, and French**.

---

## Model details

| Property | Value |
|---|---|
| Base model | `google/gemma-4-27b-it` |
| Fine-tuning method | Unsloth LoRA (r=16, alpha=16) |
| Training data | ClinicalTrials.gov structured eligibility criteria |
| Training steps | 200 (Kaggle T4 GPU) |
| Context window | 256,000 tokens |
| Thinking mode | Enabled (chain-of-thought reasoning) |
| Languages | EN, PT, ES, FR |
| License | Apache 2.0 |

---

## Benchmarks

| Metric | Score | Notes |
|---|---|---|
| Eligibility match precision @3 | TBD | Evaluated against expert annotation |
| OCR accuracy (lab reports) | TBD | Gemma 4 E4B sub-model |
| Multilingual BLEU (PT) | TBD | vs. reference Portuguese translations |
| Inference latency (Ollama, M2) | ~10s | Full 27B, local inference |

*Benchmarks will be updated after training run completes.*

---

## How to use

### With Ollama (recommended — local, private)

```bash
# Pull the base model via Ollama
ollama pull gemma4:27b

# Run the full TrialLens app
git clone https://github.com/MahamadYoussoufDjibrine/triallens
cd triallens
pip install -r requirements.txt
streamlit run app/main.py
```

### With transformers + PEFT (for fine-tuned weights)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "google/gemma-4-27b-it"
lora_weights_id = "triallens/gemma4-27b-clinical-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_weights_id)

prompt = """<start_of_turn>system
You are a clinical trial eligibility expert. Given a patient profile and clinical trial 
eligibility criteria, assess whether the patient may qualify and explain the key criteria 
in plain language.<end_of_turn>
<start_of_turn>user
Patient: {"diagnosis": "Stage 2 HER2-positive breast cancer", "age": 42, "sex": "Female", "country": "Brazil"}
Trial: Phase 2 Study of Trastuzumab in HER2-Positive Breast Cancer
Eligibility criteria:
Inclusion Criteria:
- Age 18-75
- HER2-positive confirmed by IHC or FISH
- ECOG performance status 0-1

Exclusion Criteria:
- Prior trastuzumab therapy
- Active CNS metastases

Respond in English.<end_of_turn>
<start_of_turn>model"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Expected output format

```json
{
  "match_label": "LIKELY",
  "match_reason": "The patient is 42 years old (within the 18-75 age range), and the diagnosis specifies HER2-positive breast cancer, which directly matches the primary inclusion criterion.",
  "key_criteria": [
    "Must have HER2-positive breast cancer confirmed by a lab test",
    "Must not have received trastuzumab before",
    "Must be generally healthy enough to participate (ECOG 0-1)"
  ],
  "plain_summary": "This trial tests whether trastuzumab — a targeted drug that blocks HER2 — works better when given earlier in treatment. Participants receive the drug every 3 weeks for up to 18 cycles.",
  "doctor_questions": [
    "Based on my HER2 status and ECOG score, do I qualify for this trial?",
    "What does 'prior trastuzumab therapy' mean — have I received this?",
    "How would participating affect my current chemotherapy schedule?",
    "Is the trial site accessible from São Paulo?",
    "What happens to my care if I need to withdraw?"
  ]
}
```

---

## Training data

This model was fine-tuned on structured eligibility criteria from [ClinicalTrials.gov](https://clinicaltrials.gov), which provides free public access to 450,000+ clinical trial records.

Training data format: patient profile + raw eligibility criteria → structured assessment JSON.

Ground truth assessments were generated using the base `google/gemma-4-27b-it` model (self-play / distillation) and reviewed for quality before training.

**Data is not redistributed here** — it is freely available from ClinicalTrials.gov. See `scripts/download_trials.py` in the [TrialLens GitHub repo](https://github.com/MahamadYoussoufDjibrine/triallens) for the download script.

---

## Limitations and responsible use

- This model is **not a medical device** and should not be used for clinical diagnosis or treatment decisions.
- Outputs should always be reviewed with a qualified healthcare provider.
- Eligibility assessments are approximate — a patient flagged as "LIKELY" still needs formal screening by the trial coordinator.
- The model may perform less well on rare cancer types or highly complex eligibility criteria with many sub-criteria.
- Multilingual performance is best for Portuguese and Spanish; French output quality may vary.

---

## Citation

```bibtex
@misc{triallens2026,
  title={TrialLens: Plain-language clinical trial matching powered by Gemma 4},
  author={MahamadYoussoufDjibrine and team},
  year={2026},
  url={https://github.com/MahamadYoussoufDjibrine/triallens},
  note={Built for the Gemma 4 Good Hackathon}
}
```

---

## Links

- GitHub: [MahamadYoussoufDjibrine/triallens](https://github.com/MahamadYoussoufDjibrine/triallens)
- Live demo: [triallens.streamlit.app](https://triallens.streamlit.app)
- Kaggle competition: [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
