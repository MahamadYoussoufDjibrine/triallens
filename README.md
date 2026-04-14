# TrialLens 🔬

**Plain-language clinical trial matching for cancer patients — powered by Gemma 4**

> Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) · Health & Sciences Track

---

## The problem

450,000 clinical trials are recruiting patients right now. Most patients never find them — not because the trials don't exist, but because eligibility criteria are written in dense medical jargon that requires a medical degree to parse.

TrialLens bridges that gap. A patient uploads their diagnosis and a photo of their lab report. TrialLens returns a ranked list of matching trials with plain-language summaries, eligibility explanations, and a ready-made list of questions to ask their doctor — in their own language.

---

## Demo

🔗 **Live demo:** [triallens.streamlit.app](https://triallens.streamlit.app) *(deployed during hackathon)*  
🤗 **Model weights:** [huggingface.co/triallens/gemma4-27b-clinical-lora](https://huggingface.co/triallens) *(published after fine-tuning)*  
📹 **Video:** [youtube.com/watch?v=...](https://youtube.com) *(link after recording)*

---

## Architecture

```
User Input (text + lab report image)
        │
        ▼
┌─────────────────────┐
│  Gemma 4 E4B (OCR)  │  ← Extracts structured data from lab report photo
└────────┬────────────┘
         │ structured JSON (diagnosis, biomarkers, age, etc.)
         ▼
┌─────────────────────┐
│  ClinicalTrials.gov │  ← Real-time API query by condition + demographics
│  API + ChromaDB RAG │  ← Vector similarity search over eligibility criteria
└────────┬────────────┘
         │ top-k matching trials
         ▼
┌──────────────────────────────┐
│  Gemma 4 27B (fine-tuned)    │  ← LoRA fine-tuned on eligibility criteria
│  + Thinking mode enabled     │  ← Reasoning chain shown in UI
│  + Function calling          │  ← Queries API, formats output
└────────┬─────────────────────┘
         │
         ▼
Plain-language summaries · Match score · Doctor Q&A list
(Portuguese / English / Spanish / French)
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Main LLM | Gemma 4 27B (LoRA fine-tuned) | 256K context fits full trial docs |
| OCR sub-model | Gemma 4 E4B | Fast, local image parsing |
| Fine-tuning | Unsloth LoRA | 2x faster, qualifies for Unsloth prize |
| Local serving | Ollama | Privacy-first, qualifies for Ollama prize |
| Vector store | ChromaDB | Local, no infra needed |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, free |
| Data source | ClinicalTrials.gov API | 450K+ trials, free, no auth |
| Frontend | Streamlit | Fast to build, deployable for free |
| Fine-tune compute | Kaggle GPU (T4/P100) | Free |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-team/triallens.git
cd triallens
pip install -r requirements.txt
```

### 2. Download & preprocess trial data

```bash
python scripts/download_trials.py --condition "breast cancer" --max 5000
python scripts/preprocess_trials.py
python scripts/build_vectorstore.py
```

### 3. Run the Streamlit app (with Ollama)

```bash
# First, pull the model locally
ollama pull gemma4:27b

# Run the app
streamlit run app/main.py
```

### 4. Fine-tune with Unsloth (optional, Kaggle GPU recommended)

Open `notebooks/finetune_unsloth.ipynb` in Kaggle and follow the instructions.  
Trained weights are published to Hugging Face automatically via the notebook.

---

## Project structure

```
triallens/
├── README.md
├── requirements.txt
├── SETTINGS.json                  # All paths configured here
├── app/
│   ├── main.py                    # Streamlit entrypoint
│   ├── components/
│   │   ├── ocr_pipeline.py        # Lab report image → structured JSON
│   │   ├── trial_retriever.py     # ClinicalTrials.gov API + ChromaDB RAG
│   │   ├── gemma_inference.py     # Gemma 4 27B inference via Ollama
│   │   └── formatter.py          # Multilingual output formatting
├── data/
│   ├── raw/                       # Downloaded trial JSON from API
│   ├── processed/                 # Cleaned eligibility criteria
│   └── vectorstore/               # ChromaDB files
├── models/
│   └── lora_weights/              # Fine-tuned LoRA adapter weights
├── notebooks/
│   ├── finetune_unsloth.ipynb     # Fine-tuning pipeline (run on Kaggle)
│   └── evaluation.ipynb           # Benchmark results
├── scripts/
│   ├── download_trials.py         # Fetch from ClinicalTrials.gov API
│   ├── preprocess_trials.py       # Clean + structure eligibility data
│   └── build_vectorstore.py       # Embed criteria into ChromaDB
└── tests/
    └── test_pipeline.py
```

---

## Evaluation & benchmarks

See `notebooks/evaluation.ipynb` for full results.

| Metric | Score |
|---|---|
| Eligibility match precision (top-3) | TBD after fine-tune |
| OCR accuracy on lab reports | TBD |
| Multilingual output quality (BLEU) | TBD |
| Inference latency (Ollama, local) | TBD |

---

## Team

Built during the Gemma 4 Good Hackathon (April–May 2026).

---

## License

Apache 2.0 — in compliance with Gemma 4 license and hackathon CC-BY 4.0 requirements.
