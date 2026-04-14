"""
preprocess_trials.py
Cleans raw trial JSONL and extracts structured eligibility records for fine-tuning and RAG.
Usage:
  python scripts/preprocess_trials.py
"""

import json
import re
from pathlib import Path


def extract_eligibility_sections(raw_text: str) -> dict:
    """Split raw eligibility criteria into inclusion and exclusion sections."""
    if not raw_text:
        return {"inclusion": [], "exclusion": []}

    inclusion, exclusion = [], []
    current = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if "inclusion" in low:
            current = "inclusion"
            continue
        elif "exclusion" in low:
            current = "exclusion"
            continue

        # Strip bullet markers
        clean = re.sub(r"^[\-\*\•\d+\.\)]+\s*", "", line).strip()
        if len(clean) < 10:
            continue

        if current == "inclusion":
            inclusion.append(clean)
        elif current == "exclusion":
            exclusion.append(clean)

    return {"inclusion": inclusion, "exclusion": exclusion}


def extract_age_range(study: dict) -> dict:
    proto = study.get("protocolSection", {})
    elig = proto.get("eligibilityModule", {})
    return {
        "min_age": elig.get("minimumAge", "N/A"),
        "max_age": elig.get("maximumAge", "N/A"),
        "sex": elig.get("sex", "ALL"),
        "std_ages": elig.get("stdAges", []),
    }


def process_study(study: dict) -> dict | None:
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    desc_mod = proto.get("descriptionModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    nct_id = id_mod.get("nctId", "")
    if not nct_id:
        return None

    raw_eligibility = elig_mod.get("eligibilityCriteria", "")
    eligibility = extract_eligibility_sections(raw_eligibility)

    # Skip trials with no eligibility info
    if not eligibility["inclusion"] and not eligibility["exclusion"]:
        return None

    locations = []
    for loc in contacts_mod.get("locations", [])[:5]:
        locations.append({
            "country": loc.get("country", ""),
            "city": loc.get("city", ""),
            "facility": loc.get("facility", ""),
        })

    return {
        "nct_id": nct_id,
        "title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle", ""),
        "summary": desc_mod.get("briefSummary", "").strip(),
        "status": status_mod.get("overallStatus", ""),
        "phase": design_mod.get("phases", []),
        "study_type": design_mod.get("studyType", ""),
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "interventions": [
            {"type": i.get("type"), "name": i.get("name")}
            for i in proto.get("armsInterventionsModule", {}).get("interventions", [])
        ],
        "eligibility_raw": raw_eligibility,
        "eligibility_structured": eligibility,
        "age_range": extract_age_range(study),
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        "locations": locations,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def preprocess(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_files = list(raw_dir.glob("*.jsonl"))

    if not input_files:
        print(f"No .jsonl files found in {raw_dir}. Run download_trials.py first.")
        return

    total_in = total_out = 0

    for infile in input_files:
        out_file = out_dir / infile.name
        print(f"Processing {infile.name}...")

        with open(infile) as f_in, open(out_file, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                total_in += 1
                try:
                    study = json.loads(line)
                    record = process_study(study)
                    if record:
                        f_out.write(json.dumps(record) + "\n")
                        total_out += 1
                except json.JSONDecodeError:
                    continue

    print(f"\nDone. {total_in} trials in → {total_out} clean records out")
    print(f"Saved to: {out_dir}")


def main():
    settings_path = Path("SETTINGS.json")
    if settings_path.exists():
        with open(settings_path) as f:
            s = json.load(f)
        raw_dir = Path(s["RAW_DATA_DIR"])
        out_dir = Path(s["PROCESSED_DATA_DIR"])
    else:
        raw_dir = Path("data/raw")
        out_dir = Path("data/processed")

    preprocess(raw_dir, out_dir)


if __name__ == "__main__":
    main()
