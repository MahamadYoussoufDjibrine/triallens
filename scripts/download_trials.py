"""
download_trials.py
Fetches clinical trials from ClinicalTrials.gov API v2 (no API key required).
Usage:
  python scripts/download_trials.py --condition "breast cancer" --max 5000
  python scripts/download_trials.py --condition "lung cancer" --max 2000 --status RECRUITING
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "BriefSummary",
    "DetailedDescription",
    "Condition",
    "InterventionType",
    "InterventionName",
    "Phase",
    "OverallStatus",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "StdAge",
    "LocationCountry",
    "LocationCity",
    "LocationFacility",
    "StartDate",
    "PrimaryCompletionDate",
    "EnrollmentCount",
    "StudyType",
    "ContactName",
    "ContactPhone",
    "ContactEMail",
]


def fetch_page(condition: str, status: str, page_token: str | None, page_size: int = 100) -> dict:
    params = {
        "query.cond": condition,
        "filter.overallStatus": status,
        # "fields": "|".join(FIELDS),
        "pageSize": page_size,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_trials(condition: str, max_trials: int, status: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{condition.replace(' ', '_')}_{status}.jsonl"

    print(f"Downloading up to {max_trials} trials for: '{condition}' (status: {status})")
    print(f"Saving to: {out_file}")

    total_saved = 0
    page_token = None

    with open(out_file, "w") as f, tqdm(total=max_trials, unit="trials") as pbar:
        while total_saved < max_trials:
            page_size = min(100, max_trials - total_saved)
            try:
                data = fetch_page(condition, status, page_token, page_size)
            except requests.HTTPError as e:
                print(f"\nHTTP error: {e}. Retrying in 5s...")
                time.sleep(5)
                continue

            studies = data.get("studies", [])
            if not studies:
                print("\nNo more studies found.")
                break

            for study in studies:
                f.write(json.dumps(study) + "\n")
                total_saved += 1
                pbar.update(1)
                if total_saved >= max_trials:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.3)  # be polite to the API

    print(f"\nDone. Saved {total_saved} trials to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Download trials from ClinicalTrials.gov API v2")
    parser.add_argument("--condition", type=str, default="cancer", help="Medical condition to search")
    parser.add_argument("--max", type=int, default=1000, help="Maximum number of trials to download")
    parser.add_argument(
        "--status",
        type=str,
        default="RECRUITING",
        choices=["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "ALL"],
        help="Trial status filter",
    )
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    settings_path = Path("SETTINGS.json")
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
        out_dir = Path(settings.get("RAW_DATA_DIR", args.out_dir))
    else:
        out_dir = Path(args.out_dir)

    status = None if args.status == "ALL" else args.status
    download_trials(args.condition, args.max, status or "RECRUITING", out_dir)


if __name__ == "__main__":
    main()
