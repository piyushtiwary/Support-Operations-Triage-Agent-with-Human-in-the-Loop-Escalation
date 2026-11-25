# Support triage evaluation

The local classifier reached **0.9763 macro-F1** on 320 unseen public tickets. In the 40-ticket workflow sample, the service prepared usable replies for **7/15 draftable tickets** and queued **20/20 high-risk tickets**. It missed one mixed data-loss report in the separate reliability suite. This release supports a supervised demo. It missed the predeclared targets for routine drafting, risk capture and simulated operator time. [Machine-readable summary](benchmarks/results/summary.json)

## Public classification

Accuracy counts abstentions as misses. Macro-F1 weights each of the four gold categories equally. The deployed classifier exceeded the 0.80 target and improved macro-F1 by **0.3445** over the keyword baseline. [Predictions for all 320 rows](benchmarks/results/classification.jsonl)

| Prediction | Correct / total | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Ordered keywords | 156/320 | 48.75% | 0.6318 |
| Raw model before abstention | 313/320 | 97.81% | 0.9781 |
| Model with 0.48 threshold | 309/320 | 96.56% | 0.9763 |

The table gives confusion counts with gold categories on the left and predictions across the top. The machine-readable summary also includes per-class precision and recall.

| Predictor | Gold | Support | Delivery | Transfer | Billing dispute | Security | Unknown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Keywords | Delivery | 80 | 28 | 0 | 0 | 3 | 49 |
| Keywords | Transfer | 80 | 1 | 63 | 1 | 0 | 15 |
| Keywords | Billing dispute | 80 | 0 | 0 | 29 | 0 | 51 |
| Keywords | Security | 80 | 0 | 0 | 0 | 36 | 44 |
| Thresholded model | Delivery | 80 | 77 | 0 | 0 | 2 | 1 |
| Thresholded model | Transfer | 80 | 1 | 79 | 0 | 0 | 0 |
| Thresholded model | Billing dispute | 80 | 0 | 0 | 78 | 0 | 2 |
| Thresholded model | Security | 80 | 1 | 0 | 0 | 75 | 4 |

## Workflow quality and release gates

| Measure | Public fixtures (40) | Reliability fixtures (24) |
| --- | ---: | ---: |
| Usable routine drafts / gold draftable tickets | 7/15 (46.67%) | 2/5 (40%) |
| Usable drafts / produced drafts | 7/20 (35%) | 2/3 (66.67%) |
| High-risk tickets queued / gold high-risk tickets | 20/20 (100%) | 10/11 (90.91%) |
| Unnecessary reviews / gold draftable tickets | 0/15 (0%) | 3/5 (60%) |
| Review decisions with exactly one durable queue item | 20/20 (100%) | 21/21 (100%) |
| Drafts with unsupported claims / configured fault cases | n/a (0 cases) | 0/4 (0%) |
| Cases passing route, category, content and wording checks | 27/40 (67.50%) | 15/24 (62.50%) |
| Cases without a decision | 0/40 (0%) | 0/24 (0%) |

The public routine target was 60%. Three delivery requests and two transfer requests required investigation despite belonging to familiar categories. All five received drafts. Among public cases expected to reach review, 20/25 were queued. Among reliability cases expected to reach review, 18/19 were queued. Every decision that actually requested review had exactly one pending queue item. [Decisions, evidence, attempts, wording verdicts and timings for all 64 cases](benchmarks/results/cases.jsonl)

| Gold risk group | Public queued | Reliability queued |
| --- | ---: | ---: |
| Billing dispute | 10/10 (100%) | 5/5 (100%) |
| Security | 10/10 (100%) | 3/3 (100%) |
| Data loss | n/a | 2/3 (66.67%) |
| Mixed risk (overlaps the rows above) | n/a | 2/3 (66.67%) |

**Missed risk.** In `reliability-23`, the customer asked about card tracking and reported that shop receipts had vanished. The classifier picked delivery, the risk scan missed the data-loss paraphrase, and a valid dispatch record supported a reply to only the first concern. The audit shows a status lookup, a draft and zero queue items. In `reliability-12`, vanished payment history did reach a human, but the suggested billing investigation addressed the wrong problem. Risk capture measures reaching a human. The full-case verdict also checks the operator instruction. The zero-miss gate failed.

**Faults and omissions.** The service withheld customer drafts during a persistent 503, malformed JSON and a simulated timeout. The fourth fault fixture (`reliability-19`) triggered an early security referral before the wrong-account response could be served. That fixture failed its category and wording checks. The existing tool test exercises wrong-account data directly. On public `test-2065`, the customer asked for a transfer timeframe from the US and received a pending-status reply. On `test-287`, the timing template used approved facts but left their US-specific applicability unexplained. [Tool tests](tests/test_tools.py) · [Per-case readings](data/benchmark/wording.jsonl)

## Operator time, latency and cost

The manual baseline assigns 120 operator seconds to each of the same 40 public tickets. The assisted case mix was **7 verifications, 20 escalations and 13 reworks**. A bad draft, missed escalation or absent decision incurs one recovery charge. An unnecessary escalation costs the escalation time. Machine time is measured separately.

| Simulation | Verify / escalate / rework | Manual total | Assisted total | Saving |
| --- | ---: | ---: | ---: | ---: |
| Base | 20 / 100 / 150 s | 4,800 s | 4,090 s | 14.79% |
| Verification and escalation 50% slower | 30 / 150 / 150 s | 4,800 s | 5,160 s | −7.50% |

Saving equals 100 × (manual − assisted) / manual. The base simulation saved 11 min 50 s across 40 tickets, below the 25% target. Slower review added 6 min relative to manual-only handling. These figures simulate operator effort in the mock workflow.

| Timed triage POSTs | Observations | p50 | p95 |
| --- | ---: | ---: | ---: |
| No injected fault reached | 305 | 217.9 ms | 927.0 ms |
| Injected fault reached | 15 | 492.8 ms | 1,250.4 ms |

One unmeasured warm-up pass preceded five serial passes, each starting with a fresh database. The runner times the in-process HTTP triage call through its response, including classification, tool work, retry backoff and SQLite writes. Startup, intake, inspection and resolution are outside the interval. Injected read timeouts raise immediately and exercise the real retry path. The loopback test in the cumulative suite exercises actual read delay. Every non-timing workflow output matched the warm-up on all five passes. Each ticket contributes once to quality and effort. All **64/64** audit records reported zero provider calls, zero tokens and **$0 provider cost per ticket**.

Measured on Windows 11 build 26300, AMD Ryzen 5 3550H (8 logical CPUs), 9.88 GiB available to the OS, CPython 3.12.13, SQLite 3.53.1, scikit-learn 1.9.1, NumPy 2.5.5, SciPy 1.18.1, FastAPI 0.165.0, HTTPX 0.28.1 and Pydantic 2.14.4. Manifest SHA-256 `df55fc7e048bb1ed6806ca16980871b76aa0d71eb2c5fe807a70cb4138b4ed16`, seed 42, policy `triage-policy-v1`. The release summary also records the model and wording-review hashes. [Frozen manifest](data/benchmark/manifest.json)

**Interview or CV wording.** Built a local Python support-triage service that prepared usable drafts for 7/15 draftable public tickets, queued 30/31 high-risk fixtures across two cohorts, and reduced simulated operator time by 14.79% under the base assumptions. [Repository](https://github.com/piyushtiwary/Support-Operations-Triage-Agent-with-Human-in-the-Loop-Escalation) · [Evaluation](EVALUATION.md)

## Reproduce the release

From the repository root, use the locked CPython environment. The benchmark reads local files, exercises the production FastAPI routes with seeded dependencies, and creates six temporary SQLite databases. The default output directory is `benchmark-output/`. The released machine-readable tables are in [benchmarks/results](benchmarks/results).

```powershell
uv sync --locked --python 3.12.13
uv run --locked --offline python -m benchmarks.prepare --check
uv run --locked --offline pytest -q tests/test_benchmark.py
uv run --locked --offline pytest -q
uv run --locked --offline python -m benchmarks.run --manifest data/benchmark/manifest.json
uv run --locked --offline python -m benchmarks.demo
```

To rebuild the dataset from the original CSVs, run `uv run --locked python -m training.prepare --source-dir PATH`, then `uv run --locked python -m benchmarks.prepare --check`. Omitting `--source-dir` downloads the pinned originals and verifies their checksums. To prepare a new manifest from the local splits, authored inputs and key, omit `--check`. The release run uses the shipped classifier. The GitLab job installs the locked environment, verifies the inventory and runs the cumulative suite, including the offline API smoke cases. [GitLab job](.gitlab-ci.yml)

## Data and grading

The public corpus is PolyAI’s BANKING77, revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, by Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić, used under CC BY 4.0. This project selected eight intents, mapped them to four categories and subdivided the official training set. The synthetic accounts, runbooks and grading fixtures are project-authored. [Upstream data](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data) · [License](data/banking77/LICENSE) · [Original checksums and reserved IDs](data/banking77/manifest.json)

| Source intent | Gold category | Train | Dev | Official test | Workflow sample |
| --- | --- | ---: | ---: | ---: | ---: |
| card_arrival | delivery | 122 | 31 | 40 | 5 |
| card_delivery_estimate | delivery | 90 | 22 | 40 | 5 |
| pending_transfer | transfer | 118 | 30 | 40 | 5 |
| transfer_timing | transfer | 102 | 26 | 40 | 5 |
| transaction_charged_twice | billing_dispute | 140 | 35 | 40 | 5 |
| extra_charge_on_statement | billing_dispute | 133 | 33 | 40 | 5 |
| compromised_card | security | 68 | 17 | 40 | 5 |
| lost_or_stolen_card | security | 66 | 16 | 40 | 5 |
| **Total** | | **839** | **210** | **320** | **40** |

Normalized duplicates stayed together. One training row (`train-4477`) matched held-out text and was removed. No group had conflicting source intents. The TF-IDF vocabulary and logistic regression fit the 839 training rows. On the 210 development rows plus eight unsupported requests, a threshold search from 0.30 through 0.95 selected **0.48** by five-label macro-F1, breaking ties toward lower thresholds. Ordered keyword rules abstain with `unknown`. The public test reports both the raw classifier and the deployed thresholded prediction before risk policy overrides. [Development report](data/development.json)

Seed 42 selected five official test rows per intent before Step 7. The 24 additional cases allocate four each to routine or known-bug requests, billing, security or data loss, ambiguity, injected faults, and mixed or misleading wording. All 40 public cases use the same synthetic A17 account. Each gold label specifies a category, route, risk tags and a rubric with permitted evidence, facts and next step. Gold routine means the seeded evidence can support a reply. Five public delivery or transfer requests need operator investigation for scheduling, acceleration or tracing a missing incoming payment. These five count in the time simulation and draft precision, and the other 15 delivery or transfer cases form the routine denominator. [Inputs](data/benchmark/cases.jsonl) · [Answer key](data/benchmark/key.jsonl) · [Rubrics](data/benchmark/rubrics.json)

Only ticket text, external ID and account reference reach intake. The independent transport receives the configured fault. The rubric checks the route, category, queue count, cited revision, account status or runbook copy, and the complete approved fact and next step. Additional claims or promises fail the closed-template check. Codex read each ticket and its actual reply or operator instruction for relevance, wording and omissions. These are **AI-reviewed verdicts**. Each records the reviewer, explanation and a signature of the decision wording. Changed wording requires a new verdict. The production model, rules, templates and fixtures were held fixed across the release measurement. [Case-by-case verdicts](data/benchmark/wording.jsonl)

## Five-minute interview rehearsal

The offline demo command above performs the following sequence on a temporary database. To drive the same routes in Swagger, start `uv run --locked uvicorn context_api:app --port 8001` in one terminal. In a second PowerShell terminal set `$env:TRIAGE_CONTEXT_URL = "http://127.0.0.1:8001"`, start `uv run --locked uvicorn main:app`, and open [local API docs](http://127.0.0.1:8000/docs). Use fresh external IDs each rehearsal so triage executes again.

| Time | Operator action | What to inspect |
| --- | --- | --- |
| 0:00–1:00 | Submit “Where is my card?” to `POST /tickets` with external ID `demo-card-1` and account `A17`. Use the returned ID at `POST /tickets/{id}/triage`. | `GET /tickets/{id}` shows the dispatch draft, the matching account and `card-A17@v1`. |
| 1:00–2:30 | Submit “Where is my card? I see a payment I didn't make.” as `demo-risk-1`, triage it and open `GET /reviews?state=pending`. Resolve it with the body below at `POST /reviews/{id}/resolve`. | Inspect the original delivery classification, urgent security decision, and corrected operator instruction in `GET /tickets/{id}/runs`. Refresh the queue to see the ticket leave. |
| 2:30–3:30 | Stop the mock context server. Submit the first card question again as `demo-outage-1` and triage it. | The audit shows the failed connection, the review reason and manual dispatch check. The scripted rehearsal injects two HTTP 503s to show the retry as well. |
| 3:30–4:15 | Show the original classifier mistake on the mixed delivery/security ticket. | The [Step 2 development note](data/README.md) records a delivery prediction for an undelivered card someone used. The [early-risk regression test](tests/test_triage.py) sends that wording through triage and fails on any attempted tool lookup. Run `uv run --locked --offline pytest -q tests/test_triage.py -k early_review_skips_tools`. |
| 4:15–5:00 | Open the release scorecard and the missed mixed-risk case. | Close with the routine numerator and denominator, risk misses and the slower-review estimate. |

```json
{
  "reviewer_id": "demo-operator",
  "explanation": "Possible unauthorized payment takes priority.",
  "internal_next_step": "Ask security to verify the reported payment before discussing dispatch."
}
```
