"""
build_vectorstore.py
Embeds clinical trial eligibility criteria into ChromaDB for fast RAG retrieval.
Usage:
  python scripts/build_vectorstore.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def build(processed_dir: Path, vectorstore_dir: Path, embedding_model: str) -> None:
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    client = chromadb.PersistentClient(path=str(vectorstore_dir))

    # Drop and recreate collection for clean build
    try:
        client.delete_collection("trials")
    except Exception:
        pass
    collection = client.create_collection("trials", metadata={"hnsw:space": "cosine"})

    input_files = list(processed_dir.glob("*.jsonl"))
    if not input_files:
        print(f"No processed files in {processed_dir}. Run preprocess_trials.py first.")
        return

    records = []
    for infile in input_files:
        with open(infile) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"Embedding {len(records)} trials...")

    BATCH = 256
    for i in tqdm(range(0, len(records), BATCH), unit="batch"):
        batch = records[i : i + BATCH]

        # Build text to embed: title + conditions + inclusion criteria (most signal-rich)
        texts = []
        for r in batch:
            inclusion = " ".join(r["eligibility_structured"]["inclusion"][:10])
            conditions = ", ".join(r["conditions"][:5])
            text = f"{r['title']}. Conditions: {conditions}. Eligibility: {inclusion}"
            texts.append(text[:2000])  # cap at 2000 chars

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[r["nct_id"] for r in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "nct_id": r["nct_id"],
                    "title": r["title"][:200],
                    "status": r["status"],
                    "conditions": ", ".join(r["conditions"][:5]),
                    "min_age": r["age_range"]["min_age"],
                    "max_age": r["age_range"]["max_age"],
                    "sex": r["age_range"]["sex"],
                    "url": r["url"],
                    "phase": ", ".join(r["phase"]) if isinstance(r["phase"], list) else r["phase"],
                }
                for r in batch
            ],
        )

    print(f"\nVector store built with {len(records)} trials.")
    print(f"Saved to: {vectorstore_dir}")


def main():
    settings_path = Path("SETTINGS.json")
    if settings_path.exists():
        with open(settings_path) as f:
            s = json.load(f)
        processed_dir = Path(s["PROCESSED_DATA_DIR"])
        vectorstore_dir = Path(s["VECTORSTORE_DIR"])
        embedding_model = s.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    else:
        processed_dir = Path("data/processed")
        vectorstore_dir = Path("data/vectorstore")
        embedding_model = "all-MiniLM-L6-v2"

    build(processed_dir, vectorstore_dir, embedding_model)


if __name__ == "__main__":
    main()
