"""
trial_retriever.py
Cloud: ClinicalTrials.gov API only (no ChromaDB dependency)
Local: ChromaDB vector search + API fallback
"""
import json
import os
import requests


def _api_search(condition: str, age: int, sex: str, country: str, max_results: int = 10) -> list:
    """Live ClinicalTrials.gov API — no auth, no local deps needed."""
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
            elig = proto.get("eligibilityModule", {})
            phase_list = proto.get("designModule", {}).get("phases", [])
            phase = phase_list[0].replace("PHASE", "Phase ").title() if phase_list else "N/A"
            locs = proto.get("contactsLocationsModule", {}).get("locations", [])
            location = locs[0].get("country", "United States") if locs else "United States"
            results.append({
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "status": proto.get("statusModule", {}).get("overallStatus", ""),
                "eligibility_raw": elig.get("eligibilityCriteria", ""),
                "phase": phase,
                "locations": [{"country": location}],
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "_source": "api",
            })
        return results
    except Exception as e:
        print(f"API search failed: {e}")
        return []


def _vector_search(query: str, settings: dict, top_k: int = 20) -> list:
    """Local ChromaDB vector search — only runs if vectorstore exists."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        vectorstore_dir = settings.get("VECTORSTORE_DIR", "data/vectorstore")
        if not os.path.exists(vectorstore_dir):
            return []

        client = chromadb.PersistentClient(path=vectorstore_dir)
        collection = client.get_collection("trials")
        model = SentenceTransformer(settings.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=top_k)

        trials = []
        for i, nct_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            trials.append({
                "nct_id": nct_id,
                "title": meta.get("title", ""),
                "status": meta.get("status", ""),
                "url": meta.get("url", f"https://clinicaltrials.gov/study/{nct_id}"),
                "phase": meta.get("phase", "N/A"),
                "_score": results["distances"][0][i],
            })
        return trials

    except Exception as e:
        print(f"Vector search unavailable: {e}")
        return []


def retrieve_trials(diagnosis, age, sex, country, lab_data, settings):
    query = f"Patient: {diagnosis}. Age: {age}. Sex: {sex}. Country: {country}."

    trials = []

    # Try vector search first (local only)
    trials = _vector_search(query, settings, top_k=settings.get("TOP_K_VECTOR_RESULTS", 20))

    # Always supplement with live API
    api_results = _api_search(
        condition=diagnosis,
        age=age,
        sex=sex,
        country=country,
        max_results=settings.get("MAX_TRIALS_TO_RETRIEVE", 10),
    )

    # Deduplicate
    seen = {t["nct_id"] for t in trials}
    for t in api_results:
        if t["nct_id"] not in seen:
            trials.append(t)
            seen.add(t["nct_id"])

    print(f"Retrieved {len(trials)} trials ({len(trials) - len(api_results)} from vectorstore, {len(api_results)} from API)")
    return trials[:settings.get("TOP_K_VECTOR_RESULTS", 20)]