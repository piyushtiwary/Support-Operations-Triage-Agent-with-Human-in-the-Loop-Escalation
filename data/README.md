# Provisional ticket classification

Run `uv run --locked uvicorn main:app` from the project root and open `/docs`.
Submit a ticket with `POST /tickets`, then call `POST /tickets/{id}/triage` with
its returned ID. `GET /tickets/{id}` shows the stored classification after a
restart. The service loads the included model once at startup and runs offline.

| Example ticket | Provisional result |
| --- | --- |
| How long does a bank transfer take? | `transfer` with a model score |
| Can I buy travel insurance? | `unknown` with the raw category and a below-threshold score |
| My app freezes when I open card tracking. | `technical_issue` with rule ID `card-tracking-freeze-v1` and a null score |

A ticket with zero matching features receives `unknown` and a null score.
If the artifact fails to load, model-based requests record `model_unavailable`.
Restore the model, restart the service and replay triage to retry those tickets.
Completed predictions survive a model change.

## Data and reproduction

The eight-intent slice derives from PolyAI's BANKING77 dataset by Iñigo
Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić.
The upstream data uses CC BY 4.0. This project selects rows, maps eight source
intents to four categories, and subdivides the original training split.
[Dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) · [Paper](https://aclanthology.org/2020.nlp4convai-1.5/) · [License](banking77/LICENSE)

The manifest records the upstream revision, original CSV checksums, label
mapping, exclusions, split counts and 40 reserved end-to-end IDs. IDs use the
official split and one-based CSV record number, excluding the header.
Preparation groups case-folded, Unicode-normalized words, excludes train/test
text matches and groups with conflicting source intents, and reserves 20% of
the remaining groups per source intent for development using seed 42.
This yields 839 training rows, 210 development rows and all 320 official test
rows. The vectorizer and logistic regression fit only the 839 training rows.

```powershell
uv run --locked python -m training.prepare
uv run --locked python -m training.fit
uv run --locked pytest -q tests/test_classification.py
```

Preparation verifies the downloaded files before use. To use local originals,
pass `--source-dir PATH` to the preparation command. Fitting works offline
from the included splits. It writes the pipeline and threshold to
`classifier.pkl` and development confusion counts to `development.json`.
Each confusion matrix uses gold categories as rows and predictions as columns.

Threshold selection equally weights the five category F1 scores using the
development split plus eight authored unsupported requests. The selected
threshold of **0.48** retained 206 correct predictions on the 210 supported
development tickets and identified five of eight unsupported examples.
The mixed smoke ticket “My card hasn't arrived and someone used it” currently
predicts `delivery`. Its security signal illustrates why the later routing
policy must inspect the original ticket text.
These are development results. Official test predictions are reserved for
the release benchmark.
