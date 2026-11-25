# Support & Operations Triage Agent

“Where is my card?” gets a draft backed by a mock dispatch record. Add “I see a payment I didn't make” and a security rule sends the ticket to a human reviewer, even when the classifier predicts delivery.

This local Python service uses FastAPI and a scikit-learn classifier to triage support tickets. It looks up account status or a runbook before drafting. Low confidence, risk signals and failed lookups trigger review. SQLite keeps the original decision, tool attempts and reviewer corrections together for inspection.

## Run the demo

With [uv](https://docs.astral.sh/uv/) installed, run these commands from the repository root.

```sh
uv sync --locked --python 3.12.13
uv run --locked --offline python -m benchmarks.demo
```

The demo walks through a routine draft, a human correction and a status-service outage. It uses a temporary database and requires no API keys. Run the tests with `uv run --locked --offline pytest -q`.

## Benchmark results

- The classifier reached **0.9763 macro-F1** on 320 held-out BANKING77 tickets across four categories.
- The service produced usable drafts for **7 of 15** eligible public tickets and queued **30 of 31** high-risk fixtures across both evaluation sets. The missed ticket combined card tracking with a report of missing receipts.
- Simulated operator time was **14.79% lower** than manual triage under the base assumptions and **7.50% higher** under slower review.

The [evaluation report](EVALUATION.md) includes the scoring criteria, per-case results and commands to reproduce the benchmark.
