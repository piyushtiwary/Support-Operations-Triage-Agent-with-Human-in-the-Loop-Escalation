"""Freeze the benchmark inventory from checked-in splits, authored inputs and answer key."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from classification import normalize


DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_MANIFEST = DATA / "benchmark" / "manifest.json"
FILES = {
    "split": "../banking77/manifest.json", "train": "../banking77/train.csv",
    "dev": "../banking77/dev.csv", "test": "../banking77/test.csv",
    "cases": "cases.jsonl", "key": "key.jsonl", "rubrics": "rubrics.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    # Git may check text out as CRLF. Hash canonical LF UTF-8 on every platform.
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def split_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_manifest(path):
    manifest = read_json(path)
    if manifest["version"] != 1 or manifest["files"].keys() != FILES.keys():
        raise ValueError("Unexpected benchmark inventory")
    paths = {}
    for name, entry in manifest["files"].items():
        target = path.parent / entry["path"]
        if digest(target) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        paths[name] = target
    return paths


def inventory():
    paths = {name: DEFAULT_MANIFEST.parent / rel for name, rel in FILES.items()}
    split = read_json(paths["split"])
    cases, key = (json_lines(paths[name]) for name in ("cases", "key"))
    ids = {case["id"] for case in cases}
    public = {case["id"] for case in cases if "text" not in case}
    if (len(ids) != len(cases) or len(key) != len(ids) or {k["id"] for k in key} != ids
            or public != set(split["end_to_end_sample_ids"])):
        raise ValueError("Case IDs disagree with the reserved sample or the answer key")
    groups = Counter(k["group"] for k in key)
    if groups != {"public": 40, "routine": 4, "billing": 4, "security_data": 4,
                  "ambiguity": 4, "fault": 4, "mixed": 4}:
        raise ValueError("Unexpected cohort sizes")
    splits = {name: split_rows(paths[name]) for name in ("train", "dev", "test")}
    if Counter(r["intent"] for r in splits["test"] if r["id"] in public) != dict.fromkeys(split["mapping"], 5):
        raise ValueError("Expected five reserved rows per source intent")
    existing = {normalize(row["text"]) for rows in splits.values() for row in rows}
    for case in cases:
        if "text" in case:
            words = normalize(case["text"])
            if words in existing:
                raise ValueError(f"Duplicate authored text: {case['id']}")
            existing.add(words)
    rubrics = read_json(paths["rubrics"])
    if any(k["rubric"] not in rubrics or (k["id"] in public) != (k["group"] == "public") for k in key):
        raise ValueError("Invalid case rubric or cohort")
    return {"version": 1, "seed": split["seed"], "revision": split["revision"],
            "files": {name: {"path": FILES[name], "sha256": digest(path)} for name, path in paths.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the existing manifest without rewriting it")
    args = parser.parse_args()
    prepared = inventory()
    if args.check:
        if prepared != read_json(DEFAULT_MANIFEST):
            raise SystemExit("Frozen manifest does not match prepared inputs")
    else:
        DEFAULT_MANIFEST.write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")
    print(f"40 public + 24 authored cases; manifest SHA-256 {digest(DEFAULT_MANIFEST)}")


if __name__ == "__main__":
    main()
