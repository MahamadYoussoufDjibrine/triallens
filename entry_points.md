# Entry Points — TrialLens

This document lists every command needed to go from zero to a running TrialLens demo.
All paths are relative to the project root. All settings (data dirs, model names) are
controlled by `SETTINGS.json` — edit that file before running anything.

---

## Prerequisites

```bash
# 1. Python 3.11+
python --version

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull Gemma 4 models via Ollama (must be installed first: https://ollama.com)
ollama pull gemma4:27b   # main reasoning model (~16 GB)
ollama pull gemma4:4b    # OCR sub-model (~3 GB)
```

---

## Step 1 — Download trial data

Fetches clinical trials from ClinicalTrials.gov (no API key required).

```bash
# Breast cancer trials (recommended starting point)
python scripts/download_trials.py --condition "breast cancer" --max 5000

# Add more conditions as needed
python scripts/download_trials.py --condition "lung cancer" --max 2000
python scripts/download_trials.py --condition "colorectal cancer" --max 2000

# Output: data/raw/<condition>_RECRUITING.jsonl
```

---

## Step 2 — Preprocess trial data

Cleans raw JSON and extracts structured eligibility criteria.

```bash
python scripts/preprocess_trials.py

# Input:  data/raw/*.jsonl
# Output: data/processed/*.jsonl
```

---

## Step 3 — Build the vector store

Embeds eligibility criteria into ChromaDB for fast RAG retrieval.

```bash
python scripts/build_vectorstore.py

# Input:  data/processed/*.jsonl
# Output: data/vectorstore/ (ChromaDB files)
# Note:   Takes ~5–15 min for 5000 trials on CPU
```

---

## Step 4 — Run the Streamlit demo

```bash
streamlit run app/main.py

# Opens at: http://localhost:8501
# Requires: Ollama running in the background (it starts automatically)
```

---

## Step 5 — Fine-tune with Unsloth (optional, Kaggle GPU recommended)

Open `notebooks/finetune_unsloth.ipynb` in Kaggle:

1. Upload your `data/processed/` files as a Kaggle Dataset
2. Set `RAW_DATA_DIR` in the notebook to point to the uploaded dataset
3. Run all cells top to bottom
4. The final cell pushes LoRA weights to Hugging Face automatically

To use fine-tuned weights locally after training:

```bash
# In SETTINGS.json, update:
# "OLLAMA_MODEL": "triallens/gemma4-27b-clinical-lora"
# Then pull via Ollama or load directly with transformers
```

---

## Step 6 — Run tests

```bash
python -m pytest tests/ -v
```

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (public)
2. Go to share.streamlit.io → New app → point to `app/main.py`
3. Add secrets in Streamlit Cloud dashboard if needed
4. Note: Streamlit Cloud cannot run Ollama — for the live demo, use the Hugging Face
   Inference API or a small cloud VM with Ollama installed

---

## Hardware used during development

| Task | Hardware | Time |
|---|---|---|
| Downloading 5000 trials | Any CPU | ~8 min |
| Building vector store (5000 trials) | Any CPU | ~12 min |
| Fine-tuning (Unsloth LoRA, 200 steps) | Kaggle T4 GPU | ~45 min |
| Inference per query (Gemma 4 27B, Ollama) | M2 MacBook Pro | ~8–12 sec |
| Inference per query (Gemma 4 4B OCR) | Any laptop GPU | ~2–3 sec |
