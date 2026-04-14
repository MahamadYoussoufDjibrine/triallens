"""
trial_retriever.py
Retrieves candidate trials using ChromaDB vector similarity + ClinicalTrials.gov API.
"""

import json
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer


_model_cache: SentenceTransformer | None = None
_collection_cache = None


def _get_collection(settings: dict):
    global _collection_cache
    if _collection_cache is None:
        vectorstore_dir = settings.get("VECTORSTORE_DIR", "data/vectorstore")
        client = chromadb.PersistentClient(path=vectorstore_dir)
        _collection_cache = client.get_collection("trials")
    return _collection_cache


def _get_model(settings: dict) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(settings.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return _model_cache


def _vector_search(query: str, settings: dict, top_k: int = 20) -> list[dict]:
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


def _api_search(condition: str, age: int, sex: str, country: str, max_results: int = 10) -> list[dict]:
    """Fallback: query ClinicalTrials.gov API directly."""
    sex_map = {"Female": "FEMALE", "Male": "MALE"}
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "filter.geo": f"distance({country},500mi)",
        "fields": "NCTId|BriefTitle|OverallStatus|Phase|EligibilityCriteria|LocationCountry",
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
            results.append({
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "status": proto.get("statusModule", {}).get("overallStatus", ""),
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "eligibility_raw": proto.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
                "_source": "api",
            })
        return results
    except Exception as e:
        print(f"API search failed: {e}")
        return []


def retrieve_trials(
    diagnosis: str,
    age: int,
    sex: str,
    country: str,
    lab_data: dict,
    settings: dict,
) -> list[dict]:
    """
    Retrieve candidate trials via vector search (primary) + API (supplement).
    Returns up to TOP_K_VECTOR_RESULTS deduplicated trials.
    """
    query = f"Patient: {diagnosis}. Age: {age}. Sex: {sex}. Country: {country}."
    if lab_data:
        lab_str = ", ".join(f"{k}={v}" for k, v in list(lab_data.items())[:10])
        query += f" Lab values: {lab_str}."

    trials = []

    # Primary: vector store (fast, local)
    try:
        trials = _vector_search(query, settings, top_k=settings.get("TOP_K_VECTOR_RESULTS", 20))
    except Exception as e:
        print(f"Vector search failed (vectorstore may not be built yet): {e}")

    # Supplement with live API results
    api_results = _api_search(
        condition=diagnosis.split(".")[0][:100],
        age=age,
        sex=sex,
        country=country,
        max_results=settings.get("MAX_TRIALS_TO_RETRIEVE", 10),
    )

    # Deduplicate by nct_id
    seen = {t["nct_id"] for t in trials}
    for t in api_results:
        if t["nct_id"] not in seen:
            trials.append(t)
            seen.add(t["nct_id"])

    return trials[:settings.get("TOP_K_VECTOR_RESULTS", 20)]
