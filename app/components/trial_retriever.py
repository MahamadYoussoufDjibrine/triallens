"""
trial_retriever.py - ChromaDB RAG + ClinicalTrials.gov API (fields param removed).
"""
import json
from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer

_model_cache = None
_collection_cache = None


def _get_collection(settings):
    global _collection_cache
    if _collection_cache is None:
        vectorstore_dir = settings.get("VECTORSTORE_DIR", "data/vectorstore")
        client = chromadb.PersistentClient(path=vectorstore_dir)
        _collection_cache = client.get_collection("trials")
    return _collection_cache


def _get_model(settings):
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(settings.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return _model_cache


def _vector_search(query, settings, top_k=20):
    model = _get_model(settings)
    collection = _get_collection(settings)
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    trials = []
    for i, nct_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        trials.append({
            "nct_id": nct_id,
            "title": meta.get("title", ""),
            "status": meta.get("status", ""),
            "conditions": meta.get("conditions", ""),
            "min_age": meta.get("min_age", ""),
            "max_age": meta.get("max_age", ""),
            "sex": meta.get("sex", "ALL"),
            "url": meta.get("url", f"https://clinicaltrials.gov/study/{nct_id}"),
            "phase": meta.get("phase", ""),
            "_score": results["distances"][0][i],
        })
    return trials


def _api_search(condition, age, sex, country, max_results=10):
    """Live API search — no fields parameter (caused 400 errors)."""
    # Use only the first part of diagnosis for cleaner search
    clean_condition = condition.split(".")[0][:60]
    params = {
        "query.cond": clean_condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "format": "json",
    }
    try:
        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        studies = resp.json().get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            nct_id = id_mod.get("nctId", "")
            elig = proto.get("eligibilityModule", {}).get("eligibilityCriteria", "")
            results.append({
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "status": proto.get("statusModule", {}).get("overallStatus", ""),
                "eligibility_raw": elig,
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "_source": "api",
            })
        return results
    except Exception as e:
        print(f"API search failed: {e}")
        return []


def retrieve_trials(diagnosis, age, sex, country, lab_data, settings):
    query = f"Patient: {diagnosis}. Age: {age}. Sex: {sex}. Country: {country}."
    if lab_data:
        lab_str = ", ".join(f"{k}={v}" for k, v in list(lab_data.items())[:5])
        query += f" Lab: {lab_str}."

    trials = []

    # Primary: vector store
    try:
        trials = _vector_search(query, settings, top_k=settings.get("TOP_K_VECTOR_RESULTS", 20))
    except Exception as e:
        print(f"Vector search failed: {e}")

    # Supplement with live API
    api_results = _api_search(
        condition=diagnosis,
        age=age,
        sex=sex,
        country=country,
        max_results=settings.get("MAX_TRIALS_TO_RETRIEVE", 10),
    )

    seen = {t["nct_id"] for t in trials}
    for t in api_results:
        if t["nct_id"] not in seen:
            trials.append(t)
            seen.add(t["nct_id"])

    return trials[:settings.get("TOP_K_VECTOR_RESULTS", 20)]
