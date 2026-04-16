# TrialLens: Plain-Language Clinical Trial Matching for Cancer Patients — Powered by Gemma 4

**Track:** Health & Sciences  
**Team:** MahamadYoussoufDjibrine et al.  
**GitHub:** https://github.com/MahamadYoussoufDjibrine/triallens  
**Live Demo:** https://triallens.streamlit.app  
**Model Weights:** https://huggingface.co/triallens/gemma4-27b-clinical-lora  

---

## The Problem

450,000 clinical trials are actively recruiting patients worldwide right now. For many patients — particularly those with cancer — these trials represent access to cutting-edge treatments that are not yet available through standard care. Yet the vast majority of eligible patients never enroll.

The reason is not a lack of trials. It is a language barrier. Eligibility criteria are written by regulatory and medical professionals in dense clinical jargon. A typical set of criteria runs 8–14 pages and includes terms like "ECOG performance status ≤ 2," "prior anthracycline therapy," and "HER2 3+ by IHC or FISH-amplified." A patient without a medical degree has no realistic path to determining whether they qualify — and most cannot afford a patient advocate to interpret the documents on their behalf.

This gap falls hardest on patients in lower-income countries and non-English-speaking communities, where patient navigation services are rare and clinical trial awareness is low.

TrialLens eliminates this barrier.

---

## What We Built

TrialLens is an AI-powered clinical trial navigator that takes a patient's plain-language description of their diagnosis and an optional photo of their lab report, and returns a ranked list of matching trials explained in simple, compassionate language — in English, Portuguese, Spanish, or French.

The core user flow is:

1. Patient types their diagnosis in plain language
2. Patient optionally uploads a photo of their lab report
3. TrialLens extracts structured medical data from the image using Gemma 4 E4B
4. TrialLens searches 450,000+ trials via ClinicalTrials.gov API and ChromaDB vector search
5. Gemma 4 27B (fine-tuned) ranks matches and generates plain-language summaries
6. Patient receives a ranked list of trials with eligibility explanations and a ready-made list of questions for their doctor

The entire pipeline can run locally via Ollama — no patient data leaves the device.

---

## Technical Architecture

### Model Layer

**Gemma 4 27B (fine-tuned)** is the core reasoning model. We fine-tuned it using Unsloth LoRA on structured eligibility criteria from ClinicalTrials.gov — training the model to translate raw medical eligibility language into patient-friendly assessments with match labels (LIKELY / POSSIBLE / UNLIKELY), plain-language explanations, and multilingual output.

Key reasons we chose Gemma 4 27B:
- **256K context window** — allows the full eligibility document of any trial to be passed without chunking or information loss
- **Thinking mode** — we enable Gemma 4's chain-of-thought reasoning, which produces more accurate and explainable eligibility assessments, and displays the reasoning process in the UI for transparency
- **Native multilingual support** — Gemma 4 supports 35+ languages out of the box, allowing us to generate high-quality output in Portuguese, Spanish, and French without separate translation models
- **Function calling** — used for structured JSON output of the eligibility assessment, ensuring consistent formatting across the Streamlit interface

**Gemma 4 E4B** serves as a fast, local OCR sub-model. When a patient uploads a photo of their lab report, E4B extracts structured values (HER2 status, CA-125 levels, ECOG score, etc.) and returns them as JSON. This structured data is then passed to the 27B model to improve eligibility matching accuracy. E4B's multimodal capability — vision + text in under 4B parameters — makes this fast and practical on commodity hardware.

### Fine-Tuning with Unsloth

We fine-tuned using Unsloth LoRA (r=16, alpha=16) on a Kaggle T4 GPU. The training dataset was constructed from ClinicalTrials.gov structured data: for each trial, we created synthetic patient profiles and generated ground-truth eligibility assessments using the base Gemma 4 27B model (a self-play / distillation approach), then reviewed samples for quality before training.

Unsloth's 2x speed improvement over standard PEFT made it feasible to complete a meaningful fine-tuning run within Kaggle's free GPU quota. The LoRA adapter adds eligibility reasoning capability to the base model while keeping the full weight of Gemma 4's multilingual and clinical knowledge intact.

Trained LoRA weights are published to Hugging Face at `triallens/gemma4-27b-clinical-lora` under Apache 2.0.

### Retrieval-Augmented Generation (RAG)

We built a two-stage retrieval system to identify candidate trials before sending them to Gemma 4:

**Stage 1 — Vector search (ChromaDB):** We embed the eligibility criteria of all downloaded trials using `sentence-transformers/all-MiniLM-L6-v2` and store them in a local ChromaDB vector store. When a patient query arrives, we perform cosine similarity search to retrieve the top-20 most semantically relevant trials. This runs entirely locally with no API calls.

**Stage 2 — Live API fallback (ClinicalTrials.gov API v2):** We supplement vector results with a live query to the ClinicalTrials.gov API, filtered by condition, status (RECRUITING), and location. This ensures newly added trials not yet in our local vector store are captured. The API requires no authentication.

Combined, this retrieval approach feeds Gemma 4 27B a curated set of 10–20 candidate trials, dramatically reducing the prompt length needed and improving match precision.

### Local Deployment via Ollama

We serve Gemma 4 27B locally using Ollama, which manages model quantization, memory, and GPU/CPU routing automatically. This means:

- Patient data never leaves the device
- No API costs at inference time
- The application works in low-connectivity environments
- It qualifies for the Ollama Special Technology Prize

### Frontend

The patient-facing interface is built with Streamlit — fast to build, freely deployable on Streamlit Cloud, and simple enough for patients without technical backgrounds to navigate. The interface handles file upload, language selection, result display, and a one-click doctor Q&A download.

---

## Key Technical Challenges and How We Solved Them

**Challenge 1 — OCR accuracy on low-quality lab report photos.** Real patient lab reports come from phone photos taken in poor lighting. We addressed this by preprocessing images (contrast normalization via PIL) before passing to Gemma 4 E4B, and by prompting E4B to omit values it cannot read with confidence rather than hallucinating them.

**Challenge 2 — Eligibility criteria are highly structured but inconsistently formatted.** ClinicalTrials.gov eligibility text has no enforced schema — some records use bullet points, some use numbered lists, some are dense paragraphs. Our `preprocess_trials.py` script normalizes these into inclusion/exclusion arrays using regex-based section detection, with graceful fallback for unstructured formats.

**Challenge 3 — Gemma 4's thinking mode output.** Thinking mode wraps the model's reasoning in `<think>...</think>` tags before the final output. Our inference pipeline strips these tags before JSON parsing, then optionally surfaces the reasoning chain in the UI for transparency — a feature that directly supports the hackathon's Safety & Trust dimension.

**Challenge 4 — Keeping the Streamlit demo fast enough for judges.** Full 27B inference takes 8–12 seconds locally. We addressed this by pre-loading the ChromaDB vector store and embedding model at app startup, and by streaming Ollama responses rather than waiting for the full completion.

---

## Real-World Validation

We tested TrialLens with five patients and caregivers across Brazil, Portugal, and Rwanda using real or realistic diagnosis profiles. Key findings:

- All five testers could identify at least one relevant trial within 60 seconds
- Three testers said they had not previously been aware of any clinical trials relevant to their situation
- The doctor Q&A list was rated as the most useful feature by four of five testers
- Portuguese output quality was rated "clear and natural" by native speakers

---

## Impact and Vision

TrialLens is open source (Apache 2.0), runs on a \$300 laptop, works offline via Ollama, and supports four languages. The fine-tuned model weights are public on Hugging Face. The data pipeline is fully reproducible using free public data.

Our immediate next step is deploying TrialLens in partnership with a cancer patient support organization in West Africa, where clinical trial awareness and access are both critically low and where Gemma 4's multilingual capabilities provide immediate value.

The broader vision: every cancer patient in the world, regardless of language, income, or location, should be able to understand whether a clinical trial exists for them — and should know the right questions to ask their doctor. TrialLens is the bridge.

---

## Links

- **GitHub:** https://github.com/MahamadYoussoufDjibrine/triallens
- **Live Demo:** https://triallens.streamlit.app
- **Model Weights:** https://huggingface.co/triallens/gemma4-27b-clinical-lora
- **Video:** [YouTube link — to be added]

---

*Word count: ~1,180 words (within the 1,500-word limit)*
