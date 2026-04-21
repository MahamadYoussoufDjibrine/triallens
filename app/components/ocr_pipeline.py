"""
ocr_pipeline.py
Extracts structured lab values from an uploaded image.
Uses Ollama locally if available, skips gracefully on Streamlit Cloud.
"""
import base64
import json
import re
from io import BytesIO

from PIL import Image


OCR_PROMPT = """You are a medical document reader. Extract all lab test values from this image.
Return ONLY a valid JSON object with lab test names as keys and their numeric values as values.
Include units in the key name (e.g. "hemoglobin_g_dl", "ca125_u_ml").
If you cannot read a value clearly, omit it.
Example output: {"hemoglobin_g_dl": 11.2, "wbc_k_ul": 6.8, "ca125_u_ml": 43.0}
Return only JSON, nothing else."""


def image_to_base64(image_file) -> str:
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_lab_data(image_file, settings: dict) -> dict:
    """
    Extract lab values from an uploaded image.
    Returns a dict of lab value name → numeric value.
    Falls back to empty dict if Ollama not available (e.g. Streamlit Cloud).
    """
    try:
        import ollama  # Only available locally
    except ImportError:
        print("Ollama not available — skipping OCR (cloud deployment)")
        return {}

    ocr_model = settings.get("OLLAMA_OCR_MODEL", "gemma3:4b")

    try:
        img_b64 = image_to_base64(image_file)
        response = ollama.chat(
            model=ocr_model,
            messages=[
                {
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [img_b64],
                }
            ],
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        lab_data = json.loads(raw)
        return {k: float(v) for k, v in lab_data.items() if isinstance(v, (int, float))}

    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return {}