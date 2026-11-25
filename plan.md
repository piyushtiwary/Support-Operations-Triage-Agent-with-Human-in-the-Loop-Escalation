# Support & Operations Triage Agent with Human-in-the-Loop Escalation

## Goal and definition of done

Build a local support desk for a fictional card and payments product. Given “Where is my card?”, the agent should find the right mock account record and prepare a draft that an operator can verify. Given “My card hasn't arrived and someone used it”, it should put the ticket in a human queue and explain which evidence triggered escalation. An interviewer should be able to replay both tickets, break a dependency, and trace the resulting decisions from intake through review.

The release should deliver evidence-backed routine drafts, deterministic escalation with human overrides, and a reproducible evaluation. Use the FastAPI docs as the operator interface. Keep all accounts and tools local, use read-only lookups, and end automation at draft preparation. A locally trained intent classifier and evidence-bound templates make the entire demo runnable on a CPU without credentials. The workflow chooses tools and actions in Python, so every routing decision can be inspected.

## Environment and library choices

The project environment uses CPython 3.12.13 in `.venv`. From the project root on Windows, the first command below recreates the locked environment on a fresh checkout. The second checks the interpreter without activating a shell. The feature code, tests, and benchmark described below are future deliverables. [uv environments](https://docs.astral.sh/uv/pip/environments/)

```powershell
uv sync --locked --python 3.12.13
.\.venv\Scripts\python.exe --version
```

| Choice | Where it earns its place |
| --- | --- |
| FastAPI, Uvicorn, pytest | One local API for intake and review. Exercise the same routes in pytest with an HTTPX async client and its in-process ASGI transport. [HTTPX ASGI transport](https://www.python-httpx.org/advanced/transports/#asgi-transport) |
| Pydantic v2 | Share strict schemas across API responses, mock tools and stored decisions. Reject unexpected fields and out-of-range scores at the boundary. [Strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/) |
| HTTPX | Use explicit timeouts for two read-only tools. Swap in a mock transport to inject a failure on an exact attempt. [Timeouts](https://www.python-httpx.org/advanced/timeouts/) · [Transports](https://www.python-httpx.org/advanced/transports/) |
| Python `sqlite3` | Store tickets, decisions and review events in one file. Transactions and unique keys keep request replays and reviewer updates consistent. [SQLite interface](https://docs.python.org/3.12/library/sqlite3.html) |
| scikit-learn | A TF-IDF and logistic regression pipeline provides a CPU classifier with class scores. Reuse the same library for per-class precision, recall, support and macro-F1. [Text classification example](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) · [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |

## Build sequence

Complete the steps in order. At each checkpoint, run the named focused tests, replay the stated scenario through the public API and run `uv run --locked pytest -q` for the cumulative suite. Use a temporary database and seeded tools in tests. Keep a routine card-delivery ticket and a mixed delivery/security ticket as recurring smoke cases so that each feature extends a workflow already exercised.

### Step 1 — Accept and recover a ticket

- **Build.** Add `POST /tickets` and `GET /tickets/{id}`. Accept an external ticket ID, nonblank text and an optional mock account reference. Store the ticket with state `received` in SQLite. Replaying an identical submission returns its existing ID. Reusing that external ID with different content returns HTTP 409.
- **Observable output.** An operator can submit the delivery ticket in the API docs, restart the process, and retrieve the same text and state by ID.
- **Test and exit.** In `tests/test_intake.py`, cover valid intake, blank or oversized text, unknown fields, unknown ID, concurrent duplicate submissions and conflicting payloads. Retrieve the ticket after reopening the database. Pass when a replay produces exactly one stored ticket.

### Step 2 — Classify a ticket and expose uncertainty

- **Build.** Prepare the BANKING77 slice specified below. Fit TF-IDF and logistic regression on training text. Record the top category and its score. Support `technical_issue` through curated known-bug phrases and `unknown` for unsupported or empty-feature input. Keep the classifier's raw prediction for evaluation even when policy later overrides routing. An exact bug-phrase rule records its rule ID and a null model score. Treat a model score as a routing signal whose threshold must be checked on development examples.
- **Observable output.** A stored ticket can acquire a persisted classification through `POST /tickets/{id}/triage`. At this checkpoint the result is a provisional classification. A transfer timing question and an unrecognized product request demonstrate the supported and uncertain paths.
- **Test and exit.** In `tests/test_classification.py`, test valid labels, score bounds, empty vocabulary matches, known bug phrases and model load failure. Fit only on training IDs and verify the saved pipeline reproduces predictions. Run the keyword baseline and classifier on the same development split. Save their per-class counts and choose a review threshold there. Replay an ingested ticket and retrieve its prediction after restart.

### Step 3 — Give the agent two reliable context tools

- **Build.** Seed a small mock status API keyed by account reference and a searchable runbook API containing delivery guidance, transfer guidance and a known mobile-app bug. Return typed evidence with a source ID and revision. Route specific status questions to the account tool and general “how long” or known-bug questions to the relevant runbook. Missing references produce a typed missing-context result.
- **Failure budget.** Give each lookup a 500 ms attempt cap inside a 1.5 s total lookup budget. Enforce the outer deadline across connection, reads, backoff and retries. Retry an idempotent GET once on a timeout or HTTP 503 if budget remains. A 404, invalid schema, empty match or account mismatch ends the lookup immediately. Preserve the failure reason for the review step.
- **Observable output.** The delivery ticket yields a record such as `card-A17@v1` with status `dispatched`. A known-bug ticket yields the matching runbook and its approved workaround. Disconnecting the status tool yields an explicit unavailable result.
- **Test and exit.** In `tests/test_tools.py`, inject success, 503 then success, repeated timeout, 404, malformed JSON, wrong-account data and a slow response that exhausts the total budget. Assert attempt counts and deadline handling with a fake clock. Add one loopback HTTP test with a delayed endpoint to catch timeout wiring errors hidden by mocks. Replay intake → classification → lookup and inspect the persisted evidence or failure.

### Step 4 — Draft from evidence or escalate by rule

- **Build.** Give the triage endpoint its final decision schema. Include `category`, `urgency`, nullable `confidence_score`, `recommended_action`, `needs_human_review`, `reason_codes`, `evidence_ids` and nullable `draft`. Allow only the actions `prepare_reply` and `request_review`. A ready draft requires evidence belonging to that ticket. A review decision requires a reason and an internal next step. Persist one final decision per ticket in the MVP.
- **Policy order.** First scan ticket text for billing disputes, suspected unauthorized use, security incidents and data loss, including mixed-intent reports. A risk match or risk-category prediction routes directly to review. Make security and data loss urgent. Next route unknown intent, classifier failure or a model score below the frozen threshold to review. For remaining cases, look up required context. Missing, conflicting or failed evidence forces review. Only a passing case can render a customer draft with whitelisted facts and a runbook-approved next step.
- **Observable output.** “Where is my card?” can produce “Your card has been dispatched” with its record ID and no invented delivery date. “Where is my card? I see a payment I didn't make” creates a review decision even if the model predicts delivery with a high score. A lookup outage yields a reason such as `status_unavailable` and an internal instruction to verify dispatch manually.
- **Test and exit.** In `tests/test_policy.py` and `tests/test_triage.py`, cover each rule, a score just below and exactly at the threshold, negated or historical risk mentions, a mixed-risk ticket, a technical workaround, and missing evidence. Assert that a risk match skips tools, a failed lookup never claims a status, and every final object passes the schema. Run all three example paths through the API. Count them as passed only when the stored state, reason and draft agree.

### Step 5 — Close the human review loop

- **Build.** Expose `GET /reviews?state=pending` and `POST /reviews/{id}/resolve`. Show the ticket, triggered reasons, available evidence and suggested next step. A reviewer can confirm the recommendation or correct the category, urgency or wording. Store reviewer ID, explanation, time and the final values alongside the original decision. Use a local demo reviewer identity.
- **Observable output.** The mixed-risk case appears once in the pending queue. After a reviewer corrects it, the ticket shows both decisions and leaves the pending list. Routine drafts remain retrievable from their tickets.
- **Test and exit.** In `tests/test_reviews.py`, cover confirmation, correction, a blank reason, two competing reviewers, request replay and persistence after restart. Save the triage decision and pending review atomically. Test a failure between writes and retry through the API. Pass when there is one final decision, at most one pending review per ticket, and one accepted resolution.

### Step 6 — Make one run explainable from its audit trail

- **Build.** Persist a run ID, ticket ID, classifier and ruleset versions, decision, monotonic elapsed time, tool names, attempts, outcomes and evidence IDs. Join later reviewer events by run ID. Record the usage source, input/output tokens and estimated provider cost. The local classifier and templated drafter make zero provider calls, so record zero tokens and zero provider cost. Track wall time separately. Log exceptions as typed outcomes and keep original ticket text out of diagnostic exports.
- **Observable output.** `GET /tickets/{id}/runs` explains why a ticket waited, which lookup failed or recovered, and who later changed the decision. A 503 followed by success shows two attempts in one run. Start a run before processing. Surface an interrupted run after a restart and permit a retry linked to it.
- **Test and exit.** In `tests/test_audit.py`, exercise a routine draft, an immediate risk escalation, a retried lookup, an interruption and an override. Verify nonnegative durations, exact attempt counts, zero provider usage and an intact original decision. Replay the three Step 4 paths and reconstruct each outcome using the audit view alone.

### Step 7 — Publish the benchmark and rehearse the interview demo

- **Build.** Add a dataset preparation command and `python -m benchmarks.run`. The runner takes a frozen manifest, exercises the production triage entry point against seeded dependencies, and writes one row per case plus a machine-readable summary. Produce `EVALUATION.md` with the data revision, split sizes, baselines, thresholds, results, timing assumptions, one failure analysis and the command to reproduce the tables. Run the offline smoke suite in GitLab CI on every change.
- **Test and exit.** In `tests/test_benchmark.py`, hand-score a tiny set containing a good draft, an unnecessary escalation, a missed risk and an incorrect draft. Assert the exact metric numerators, denominators and simulated times. Repeat on a fresh database and compare all non-timing outputs. Run the full benchmark and cumulative tests from a clean environment. Publish missed targets and representative errors alongside successes.
- **Interview output.** In five minutes, submit a routine ticket and inspect its evidence, submit a mixed-risk ticket and resolve it as a human, then force an outage and trace the fallback. End with the measured automation numerator/denominator, the risk misses, and how the time estimate changes under slower review. Pick one genuine development failure and show the regression test that caught it.

## Benchmark protocol

### Freeze the evaluation inputs before tuning

| Input | Construction | Question it answers |
| --- | --- | --- |
| Public intent slice | Use the original BANKING77 train/test files. Map `card_arrival` and `card_delivery_estimate` to **delivery**, `pending_transfer` and `transfer_timing` to **transfer**, `transaction_charged_twice` and `extra_charge_on_statement` to **billing_dispute**, and `compromised_card` and `lost_or_stolen_card` to **security**. Keep every official test row for these eight intents and publish the actual counts. | How well does the agent classify four supported categories in unseen public ticket text? |
| End-to-end subset of public tickets | Before tuning, sample five official test rows per source intent with seed 42 (40 tickets). Attach synthetic account and runbook fixtures. Hand-label expected route, allowed facts, required next step and acceptable evidence IDs from the text and fixtures. Resolve ambiguous labels before seeing predictions. Keep the answer key outside the application inputs. | On benchmark ticket text, how often does the system both classify correctly and produce a usable draft, or escalate appropriately? |
| Authored reliability suite | Freeze 24 additional held-out cases, allocating four each to clear routine/known-bug requests, billing disputes, security/data loss, ambiguity/unsupported requests, tool failures, and mixed-intent or misleading wording. Include low-confidence cases, risk paraphrases and benign near-matches. Write separate development examples for the milestone tests. | Can the entire workflow recover from faults and catch risks outside the public dataset's labels? |

BANKING77 contains 13,083 examples across 77 banking intents and has predefined training and test splits. This project reports results on its named subset. Preserve the upstream revision, file checksums, license (CC BY 4.0), attribution and category mapping. The attached account states, draft rubrics and reliability cases are project-authored. [Original dataset and license](https://github.com/PolyAI-LDN/task-specific-datasets) · [Dataset metadata and splits](https://huggingface.co/datasets/PolyAI/banking77/blob/96a234bd25c04939c4a79213eb764ae90e4d0d81/dataset_infos.json) · [Original paper](https://aclanthology.org/2020.nlp4convai-1.5/)

Reserve a stratified development split from the official training rows. Group normalized duplicate text in one split and remove training/development matches against held-out text. Record exclusions and conflicting labels. Fit the vectorizer on training text only. Freeze keywords, weights, threshold, templates, fixtures and rubric before evaluating. Keep all authored evaluation labels hidden from tool selection and drafting. A fix prompted by a final-set failure becomes a regression test. Label later scores on that set as reruns.

### Measure accuracy, draft quality and operator effort

1. **Classification baseline.** Freeze a small ordered keyword map with an `unknown` default on development data. Compare it with the local classifier on every public test row, including abstentions as misses. Report accuracy, macro-F1 over the four gold categories, per-class support and a confusion matrix with an `unknown` prediction column.
2. **End-to-end quality.** On the 40 public fixtures and 24 reliability fixtures, report each cohort separately. A usable draft needs the correct category, route, evidence, supported factual claims and an appropriate next step. Check IDs, statuses and forbidden promises automatically, then score wording and omissions against the frozen rubric by hand. Record the reviewer and per-case verdict. Report correct routine drafts / all gold routine tickets, usable drafts / all produced drafts, high-risk reviews / all gold high-risk tickets, and unnecessary reviews / all gold routine tickets. Show counts with percentages and use “n/a” for empty denominators.
3. **Human effort baseline.** Compare manual-only and assisted handling for the same 40 public fixtures. As an initial simulation assign 120 operator seconds to triage manually, 20 to verify a usable draft, 100 to handle an escalation, and 150 to detect and rework an incorrect draft or missed escalation. Charge the error bucket first, then escalation, then usable draft, so every case contributes once. A run that produces no decision gets a single 150-second recovery charge and counts as a quality failure. Publish `100 × (manual_total − assisted_total) / manual_total`, the assumptions and the case mix. Repeat with verification and escalation times 50% higher. Report machine latency separately. A negative saving is a valid result.

### Decide what is ready to show

| Measure | Predeclared target or reporting rule |
| --- | --- |
| Categorization | Aim for macro-F1 ≥ 0.80 on the four-category public slice and report the difference from the keyword baseline. |
| Routine drafting | Aim for usable drafts on ≥ 60% of gold routine tickets in the public end-to-end subset. Also report draft precision so broad automation cannot hide wrong replies. |
| Escalation and outages | Require zero missed gold high-risk cases and zero unsupported status claims in fault cases before demoing unattended draft preparation. Show counts for billing, security, data loss and mixed risk. Every eligible escalation must have one durable queue item. |
| Time and cost | Aim for ≥ 25% simulated operator-time reduction under the stated base assumptions. Publish the base and sensitivity results and model-provider cost per ticket. These measurements describe the mock workflow. |
| Latency and reproducibility | Record hardware, Python and package versions, manifest hash and seed. After one warm-up pass, run five serial repetitions with fresh databases. Report wall-clock p50/p95 for healthy and faulted requests separately, including retries. Repeated timing runs count once per ticket for quality metrics. |

After the release run, use the measured direction and counts to complete a CV bullet such as “Built a Python support-triage service that produced usable drafts for [n/N] routine benchmark tickets, routed [r/R] high-risk fixtures to human review, and reduced simulated operator time by [p]% under documented assumptions.” Link the repository and `EVALUATION.md` so an interviewer can inspect the cases behind each number.
