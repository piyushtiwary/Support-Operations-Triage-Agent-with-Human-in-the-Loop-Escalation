"""Rebuild the frozen eight-intent slice from the original PolyAI CSV files."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

from classification import normalize

DATA = Path(__file__).resolve().parents[1] / "data" / "banking77"
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data"
CHECKSUMS = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
CATEGORIES = {
    "card_arrival": "delivery", "card_delivery_estimate": "delivery",
    "pending_transfer": "transfer", "transfer_timing": "transfer",
    "transaction_charged_twice": "billing_dispute", "extra_charge_on_statement": "billing_dispute",
    "compromised_card": "security", "lost_or_stolen_card": "security",
}
SEED = 42


def source_rows(split, source_dir):
    name = f"{split}.csv"
    if source_dir is None:
        with urlopen(f"{SOURCE}/{name}", timeout=30) as response:
            content = response.read()
    else:
        content = (source_dir / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != CHECKSUMS[name]:
        raise ValueError(f"Upstream checksum mismatch for {name}")
    for number, row in enumerate(csv.DictReader(StringIO(content.decode("utf-8"))), start=1):
        intent = row["category"]
        if intent in CATEGORIES:
            yield dict(id=f"{split}-{number}", intent=intent, category=CATEGORIES[intent],
                       text=row["text"])


def prepare(source_dir=None):
    test = list(source_rows("test", source_dir))
    held_out = {normalize(row["text"]) for row in test}
    groups = defaultdict(list)
    for row in source_rows("train", source_dir):
        groups[normalize(row["text"])].append(row)

    excluded = []
    by_intent = defaultdict(list)
    for key, rows in groups.items():
        intents = {row["intent"] for row in rows}
        if key in held_out or len(intents) > 1:
            excluded.append({"ids": [r["id"] for r in rows], "source_intents": sorted(intents),
                             "matches_test": key in held_out, "conflicting_intents": len(intents) > 1})
        else:
            by_intent[rows[0]["intent"]].append(rows)

    splits = {"train": [], "dev": [], "test": test}
    rng = random.Random(SEED)
    for intent in sorted(by_intent):
        groups_for_intent = by_intent[intent]
        rng.shuffle(groups_for_intent)
        n_dev = round(0.2 * len(groups_for_intent))
        for index, rows in enumerate(groups_for_intent):
            splits["dev" if index < n_dev else "train"].extend(rows)

    rng = random.Random(SEED)
    sample_ids = [row["id"] for intent in sorted(CATEGORIES)
                  for row in rng.sample([r for r in test if r["intent"] == intent], 5)]
    for name, rows in splits.items():
        with (DATA / f"{name}.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["id", "intent", "category", "text"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "upstream": SOURCE, "revision": REVISION, "sha256": CHECKSUMS, "mapping": CATEGORIES,
        "license": "CC BY 4.0", "seed": SEED, "excluded_training_groups": excluded,
        "counts_by_intent": {k: Counter(r["intent"] for r in v) for k, v in splits.items()},
        "end_to_end_sample_ids": sample_ids,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Use already-downloaded upstream CSVs")
    prepare(parser.parse_args().source_dir)
