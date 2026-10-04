# AI/ML Opportunity Radar

One-command radar: discover recent AI/ML papers via **OpenAlex** (the sole
scholarly discovery API; arXiv appears only as an OpenAlex location string),
screen the full pool with **CLEF** (preferred) or **Qwen** (chat fallback),
then analyze the shortlist with three specialists and a synthesizer using
the local **Strata** model through **PydanticAI** (typed outcomes,
Chat Completions path).

> Refactor note (issue #34, finished): the code now lives in the
> responsibility layout below. Old flat modules moved to their new homes
> with no compatibility shims.

## Responsibility architecture

```text
src/radar/
  cli.py               args, output printing, exit codes 0/2/3/4 (thin)
  pipeline.py          orchestration only (no HTTP, prompts, or rendering)
  config/
    interests.py       interest profile (AI/ML core + behavioral/economic lenses)
    runtime.py         all numeric bounds (service-free)
    searches.yaml      research questions and historical/frontier/challenge retrieval
    searches.py        profile/template expansion, bounded query plan
    qwen_screening.yaml  small-batch fallback concurrency and deadline policy
    triage.py          packaged, typed Qwen screening policy loader
    yaml.py            safe YAML parsing and typed package-resource loading
    triage_questions.py identical routing questions/criteria for both endpoints
  prompts/
    screening_questions.yaml   shared SystemOne questions and true/false criteria
    paper_triage.yaml          Qwen screening-agent instructions
    opportunity_analysis.yaml synthesis instructions, header and task template
    ml_methods.yaml           technical mechanisms and controlled evaluation
    behavioral_economics.yaml behavioral/economic transfer and identification
    evidence_review.yaml      falsification, evidence gaps and replication
    catalog.py                typed access to the prompt documents
  source/
    openalex.py        HTTPX discovery requests, retry/quota,
                       normalization, dedup, provenance
  provider/
    strata.py          LAN PydanticAI model construction + client lifecycle
    freetoken.py       retired-provider import compatibility only
    clef.py            LAN CLEF/SystemOne screening client (native HTTPX,
                       batch triage, deadlines, fail-fast)
  agent/
    research_team.py   common evidence cohort, specialists, shared deadline,
                       bounded contributions, synthesis and client cleanup
    paper_triage.py    same screening inputs/answer models over PydanticAI
                       Strata chat, complete-paper batches and deadlines
    opportunity_analysis.py  typed Agent[None, RadarDraft], complete-paper prompt
                       assembly + YAML instructions, deadline/usage,
                       output validation
  processing/
    triage_input.py    canonical model/state/questions input for both backends
    ranking.py         deterministic scoring/selection (pool → ranked topN)
    evidence.py        evidence index resolution/validation
    link_validation.py pure publisher-link / OpenAlex identity validation
    triage.py          CLEF shortlist policy (scored rank + unknown slot)
  storage/
    sqlite.py          authoritative pools, screening, reports and run history;
                       package-owned schema and directory writer lock
    graph_projection.py rebuild coordination and persisted graph readiness
    falkor.py          owned local engine lifecycle, graph mapping/read queries
    snapshots.py       strict legacy v1 import and programmatic compatibility
    triage.py          legacy programmatic sidecar compatibility (not CLI output)
  output/
    markdown.py        Markdown report rendering only
    json.py            JSON output shaping only (collect-only envelopes)
    triage.py          triage coverage summaries (considered/scored/unknown/failed)
  schema/
    papers.py          paper/plan contracts (CollectedWork and friends)
    opportunities.py   analysis/briefing contracts (RadarDraft and friends)
    triage.py          shared SystemOneResponse, Qwen batch correlation,
                       TriageBatch/TriageResult and strict probabilities
    configuration.py   immutable Pydantic search/question/prompt contracts
```

Dependency direction: `cli → pipeline → {config, source, provider, agent,
processing, storage, output, schema}`; adapters never depend on
cli/pipeline/output; `schema`, `config` and `prompts` are service-free. No plugin
framework.

## Editable searches, questions and prompts

YAML is the source of truth for research-driven search policy, shared screening
questions, and the agents' instructions. Python owns expansion, transport,
output schemas and hard safety limits; there is no second hardcoded search
algorithm or inline instruction fallback.

- `config/searches.yaml` v2: a learning agenda, with a stable question ID,
  explanatory research question, scope and three retrieval roles per question.
  **Foundation** retrieves historical context; **frontier** retrieves recent
  work; **counterevidence** seeks failure modes and competing explanations.
  A role records retrieval intent, not a verified characterization of a paper.
  Default questions cover generalization, uncertainty/causal identification,
  strategic objectives/incentives, and measurement/human-AI effects.
  Search defines the pool; System1 decides what merits investigation; Qwen
  investigates the selected evidence cohort and composes the synthesis.
- `semantic` sends natural-language questions through OpenAlex's documented
  `search.semantic`; `keyword` sends Boolean expressions through `search`;
  `recent` is newest-first keyword retrieval. Semantic searches are limited
  to 2000 characters and paced at one request/second. Keyword expressions
  retain the 300-character bound. Limits are checked, never silently truncated.
  Only frontier branches use `--lookback-days`; historical branches have no
  freshness filter. The live semantic API rejects day-level filters: use its
  supported publication-year range, then enforce the exact lower date locally.
  Semantic frontier rows with unknown/invalid dates cannot establish freshness
  and are skipped; historical branches can still discover those works. This
  may underfill a frontier page; no unseen pages are implied. Results retain `(question_id, role)` provenance through
  deduplication and SQLite persistence. No new paper source is introduced.
- `{keywords}` and `{domains}` are plain comma-separated phrases in semantic
  questions and escaped/quoted OR expressions in keyword queries. An empty
  domain profile skips only cross-domain questions. The planner interleaves
  questions by role, capped at 12 requests. Defaults request 15 rows per
  branch (at most 180 before dedup), below the global 200-work cap so later
  branches cannot be crowded out by earlier ones. Custom larger pages can
  reach that global cap early. CLI `--keywords` overrides profile terms.
- Legacy v1 flat `queries` remain readable with their original six-request
  cap and shared date window; their former `semantic` label is normalized to
  `keyword` because those templates never performed embedding search.
- `config/qwen_screening.yaml`: immutable typed fallback batch/concurrency
  and timeout settings. It changes transport scheduling, not System1 questions.
- `prompts/screening_questions.yaml`: the fixed two named questions
  (`research_importance`, `cross_domain_potential`), their `type: noul`,
  instruction templates and criteria. `{keywords}`/`{domains}` use the
  existing screening profile formatting. Quote the YAML keys `"true"`/`"false"`
  so they remain strings. Native CLEF and Qwen chat share the same rendered
  dictionaries and the same Pydantic `SystemOneAnswers` schema. Primary
  `research_importance` asks whether the abstract/metadata supports
  consequentially important, scientifically substantive evidence or advanced
  postgraduate learning payoff within broad AI/ML plus
  behavioral/economic/method transfer (foundational, negative/replication,
  mechanistic, boundary-condition, identification/incentive findings), not term
  relevance, popularity, hype, or citation counts; it distinguishes reported
  evidence from proposed transfer, treats missing detail as uncertainty rather
  than proof of low importance, and infers no global novelty or full-text validation.
  Secondary `cross_domain_potential` requires a grounded bridge with
  consequential payoff, not verbal analogy. Prompt guidance discourages score
  saturation; Qwen probabilities remain uncalibrated prompted estimates, not
  verified importance measurements. `rubric_version:
  clef-importance-v1` identifies the importance guidance; both backends record
  its canonical configuration SHA-256 as `rubric_hash` in stored screening
  records. `version: 1` is separately the configuration schema version.
  Historical `clef-triage-v1/v2` batches with legacy `ai_ml_relevance` remain
  readable in SQLite/JSON history with their old max-probability plus one
  reserved unknown-slot semantics; no migration or rewrite is performed.
- `prompts/paper_triage.yaml`: Qwen screening-agent instructions.
- `prompts/opportunity_analysis.yaml`: synthesis-agent instructions,
  `candidate_header`, and `task_template`. Task placeholders are
  `{max_opportunities}` and `{valid_range}`; Python supplies the actual run
  bounds. All chat agents pass instructions through PydanticAI's `instructions`
  interface, separate from untrusted user/paper data. Combined instructions
  and user messages retain the existing prompt budget and complete-block policy.

Documents require `version: 1` (search agendas use `version: 2`), are safely
parsed and validated into immutable Pydantic models, and are loaded from
package resources rather than the working directory. Both root packages
and the YAML resources are included in built distributions. Unknown/duplicate
keys, unsafe YAML tags, blank text, unsupported versions/placeholders and
oversized documents fail closed with a redacted configuration error (CLI
exit 4), before source/model calls. No environment interpolation or credentials
belong in these files. Defaults are cached per process: restart a long-running
process after edits; a new CLI refresh loads the edits automatically.

Metadata collection does not need agent prompts or inference. This slice
adds no autonomous search tool loop; bounded LLM exploration remains separate.

## Specialized research team

Normal analysis uses three actual PydanticAI specialists, followed by final
opportunity synthesis. Their YAML documents specify mission, method, evidence
restrictions and output/task standards. All share the configured LAN model;
they are not independent models or additional scientific sources.

- ML methods: learning/inference mechanisms, assumptions, evaluation blind
  spots, robustness and efficiency trade-offs.
- Behavioral/economics: grounded transfers through incentives, constructs,
  causal designs and human-system effects; no superficial analogies.
- Evidence/replication: independently examine the same abstracts for missing
  controls, alternatives and replication needs. This reviewer does not receive
  or approve the other specialists' proposals.
- Synthesis: reconcile contributions against original candidate data, retain
  uncertainty, deduplicate and propose at most two falsifiable opportunities.
  Agreement between agents is not scientific corroboration.

Each specialist returns a typed `RadarDraft`: at most one opportunity and
1500 serialized characters (prompt guidance targets about 1100 characters to
leave headroom under the hard cap). `ResearchResult` retains role-attributed reports,
the final draft/prompt and actual included papers for in-process callers.
The CLI renders the final report and names the roles in coverage notes;
specialist reports are not separately persisted. They remain untrusted
hypotheses, never source evidence or instructions.

At most two specialists run concurrently. Each stage permits two PydanticAI
requests including one validation retry, so analysis normally uses four
requests and at most eight; provider/SDK transport retries are disabled.
Specialist output caps are 1000 tokens (or the smaller CLI cap), and synthesis
retains the CLI cap. Model reasoning consumes the same token budget and can
exhaust it before producing a structured answer. The 1500-character validation
cap does not increase that token budget. The generic thinking-disable option
remains opt-in; the verified local Strata instance uses `STRATA_DISABLE_THINKING=1`
to reserve output tokens for answers. One
`--analysis-timeout` covers all specialist and
synthesis calls. Failure cancels outstanding work, closes the owned session
and produces no partial-success report.

Every stage sees the same complete candidate blocks and indices. Selection
reserves 5000 characters for intermediate context (three reports plus wrappers
fit), within the combined 12,000-character instruction/user-message budget. Richer prompts can reduce
the actual analyzed shortlist; coverage reports that cohort, not the requested
size or all discovered papers. Evidence indices, opportunity counts and
contribution sizes are validated in code with bounded retries. No agent has
research tools or PDF access; abstracts alone cannot establish quality,
causality, global novelty, replication or deployment safety. Specialization
adds inference cost/latency; metadata-only refreshes remain model-free.

### Evidence boundary

Model citation indices must be actual integers: booleans, numeric strings
and floats are rejected by the shared Pydantic draft schema, not silently
converted to a different paper index. The JSON wire schema still declares
integers. Specialist and synthesis calls retain their bounded validation
retry; persistent invalid citations fail without a successful report.

Evidence links are attached in code. Publisher URLs must be HTTP(S) with a
valid authority/port, no embedded credentials and no raw whitespace, controls,
backslashes, quotes or angle brackets. A malformed publisher URL falls back
to the validated OpenAlex work URL instead of aborting the report. If neither
link is safe, that citation is omitted; the opportunity and other valid links
remain. Source normalization requires an HTTP(S) `openalex.org/W...` identity
without ports, query strings or fragments, rather than a domain substring.
No snapshot migration, URL fetching, DNS lookup or reachability claim is
involved; legacy cached rows remain readable but unsafe links are not emitted.

## Offline transport acceptance

The full pipeline can be verified without an inference server or OpenAlex
access. The integration suite in `tests/radar/test_pipeline_chat_protocol.py`
substitutes only HTTP responses, preserving the real PydanticAI agents,
Chat Completions provider, native CLEF adapter, prompt documents, routing,
snapshot/sidecar storage, evidence attachment and Markdown rendering.

```sh
PYDANTIC_AI_NO_BANNER=1 uv run --locked python -m unittest discover \
  -s tests/radar -p test_pipeline_chat_protocol.py -v
```

It exercises fresh collection through the bounded search agenda, deduplication and
full-pool snapshot coverage; a cached 106-paper pool with 102 abstract-bearing
papers and the maximum 200-paper pool (192 abstracts, eight missing); native
CLEF preference and complete chat rerouting after native
failure; three specialists plus synthesis over shared indices; actual wire
token caps and opt-in thinking settings; validation retries, malformed or
unstructured answers, cancellation, client cleanup and redacted errors.
Source/analysis failures preserve previous snapshots and produce no false
successful report. Metadata-only cached runs perform no inference.

These tests use synthetic papers and predetermined HTTP model responses.
They neither read `.env` nor open network sockets, and do not establish model
quality, server compatibility or a successful live research run.
The operator retired FreeToken and explicitly selected Strata as its replacement.
The canonical adapter is now `provider/strata.py`; prior offline evidence does
not establish live compatibility or model quality for the replacement service.

`tests/radar/test_architecture.py` resolves the actual imported radar package
and scans nested application modules, refusing empty scans. It checks static
imports (including relative and root-package imports) using Python's AST,
so docstrings/comments mentioning another component are not dependency edges.
Its leaf-import subprocess uses that package's parent on `PYTHONPATH` and
blocks HTTP/LLM dependencies. These guards check the stated direct-import
rules, not arbitrary dynamic imports or transitive runtime reachability.

## Source vs provider vs agent

- **Source** (`source/openalex.py`) talks to the outside scholarly world:
  bounded OpenAlex requests over standard HTTPX, transient-429 backoff,
  confirmed-daily-quota fast-fail, and normalization into paper contracts.
  No LLM calls. Errors are redacted (no URLs, credentials, headers, or
  raw bodies).
- **Provider** (`provider/strata.py`) talks to your LAN machine: it
  validates the private-network endpoint, resolves the model id, and owns
  the inference HTTP client lifecycle (upstream internal `httpx2`,
  explicitly opened and deterministically closed per run). No prompts,
  no analysis.
- **Routing** (`provider/clef.py`, `agent/paper_triage.py`,
  `processing/triage.py`) decides
  *what* gets analyzed: the provider screens every collected paper over
  native HTTPX CLEF/SystemOne calls or PydanticAI Qwen chat calls (typed
  results, deadlines, bounded concurrency), and
  the pure shortlist policy for current `clef-importance-v1` ranks by primary
  `research_importance` DESC, then `cross_domain_potential` DESC, then stable
  work ID, requiring `research_importance >= 0.5` with no unknown reservation
  or backfill (unknowns stay recorded and unassessed for later investigation;
  no scores means no investigation calls and an honest note). Historical and
  legacy batches keep the old maximum-probability rank with one reserved
  unknown slot (only when unknowns exist and the shortlist holds two or more).
  This is decision routing; the synthesis agent below does the thinking.
  CLEF is preferred. If absent or unsuccessful, the Qwen endpoint screens
  the entire original pool through the existing Strata chat endpoint.
  Both use the same canonical `{model,state,questions}` inputs and the
  same Pydantic `SystemOneResponse` answer model. Qwen chat framing adds
  batch/work-ID correlation, not a different question/rubric or truncated
  paper text. Backend scores are never mixed. Qwen probabilities are
  prompted estimates, explicitly distinguished from native CLEF `noul` values.
- **Research team** (`agent/research_team.py`) runs the specialized workflow
  and final synthesis. `agent/opportunity_analysis.py` owns reusable complete-
  block prompt assembly and the standalone synthesis seam. Neither resolves
  providers or fetches sources; evidence URLs are attached in code.

## Usage

```sh
# Collect-only: bounded real OpenAlex candidates as JSON (no LLM).
uv run --env-file .env python -B -m radar --collect-only --max-candidates 8

# Full run: collect + SystemOne screening + local-model analysis as Markdown
# (requires working CLEF or the configured Strata Qwen chat service).
uv run --env-file .env python -B -m radar --max-candidates 8

# Unattended metadata refresh (full pool in SQLite and Falkor, no LLM).
uv run --env-file .env python -B -m radar --collect-only --storage-dir data/radar

# Cached synthesis: zero OpenAlex calls (SystemOne screening still mandatory,
# needs a working routing endpoint plus Strata for synthesis).
uv run --env-file .env python -B -m radar --from-db --max-candidates 2

# Import a legacy snapshot read-only (no discovery or inference).
uv run --env-file .env python -B -m radar --collect-only \
  --from-snapshot data/radar/snapshot.json --storage-dir data/radar

# Repair the graph from authoritative SQLite (no discovery or inference).
uv run --env-file .env python -B -m radar --rebuild-graph --storage-dir data/radar

# Options.
uv run --env-file .env python -B -m radar --help
uv run --env-file .env python -B -m radar --keywords "graph neural networks" --lookback-days 30
```

## Self-contained database storage

Every normal CLI run stores the entire bounded discovery pool, not just the
displayed shortlist, in `radar.sqlite3`. Screening records include the typed
results and backend/model/rubric provenance. Valid final reports and their
evidence links are stored there too; unsuccessful runs retain their available
pool/screening records and a failure category, not raw model error bodies.
Immutable pool rows preserve each run even when a paper's latest metadata changes.
Use Python's `-B` flag as shown above (or `PYTHONDONTWRITEBYTECODE=1`) to
prevent the interpreter from creating its own bytecode cache files.

SQLite is authoritative. Falkor is a rebuildable projection in `graph.rdb`:
`Paper` nodes, report `Run` nodes, hypothesis `Opportunity` nodes, and
`HAS_OPPORTUNITY`/`SUPPORTED_BY` links grounded in existing report evidence.
These are not paper-to-paper citation relationships or independently verified
discoveries. No additional scientific source or inference is used to build it.
Graph failure leaves SQLite intact and readiness explicitly pending; rerun
`--rebuild-graph`. Readiness is published only after graph persistence succeeds.

The storage directory defaults to `RADAR_STORAGE_DIR` or `data/radar` relative
to the current directory; pass an absolute `--storage-dir` when invoking Radar
elsewhere. It cannot be under `/tmp`. A directory lock permits one writer at a
time without a lockfile. SQLite may create database-managed journal files during
transactions. Existing legacy snapshots are never overwritten or removed.

Falkor's bundled Linux engine starts on an authenticated, ephemeral loopback
port for the storage operation and is stopped afterward. Startup arguments and
schema live in `src/radar`; Radar creates no disk config, PID, socket, log,
report, sidecar, extracted-text or diagnostic files. Database-managed storage
and any future downloaded PDFs are the only permitted application outputs.
Falkor's Python dependency is pinned; its native engine must be supported by
the host. It does not start or alter your Strata/CLEF inference services.

`--refresh-dir` and `--triage-output` are deprecated aliases for the database
directory, not JSON writers. Conflicting directory aliases are rejected.
The programmatic `PipelineRequest` legacy snapshot APIs remain compatible;
the CLI always opts into database storage. Collection JSON and analysis Markdown
are printed to stdout, not saved as extra files.

## Screening (mandatory routing; CLEF preferred, Qwen chat fallback)

Every analysis run screens the full pool before shortlist selection and
Strata synthesis. CLEF uses `CLEF_BASE_URL` / `--clef-base-url` and
`CLEF_MODEL` (default `clef-flash`). When CLEF is absent, unavailable,
malformed, incomplete, or deadline-expired, the full original pool is
screened through Strata Chat Completions. Explicit unsafe/invalid
CLEF configuration is refused, not bypassed.

Qwen uses the existing `STRATA_BASE_URL` / `--base-url` and
`STRATA_MODEL` / `--model` configuration and provider. CLEF receives
native SystemOne inputs; Qwen receives batches of those same full inputs
as chat user-message content. Both validate `model` and the same two
`answers` (`type: noul`, strict finite numeric `noul` in 0..1) via shared
Pydantic models. Batch work IDs must match exactly, with one bounded
validation retry. The questions, criteria and full paper data are shared;
there is no heuristic substitute, rewritten decision task, or need for
Strata to expose `/v1/systemone`. PydanticAI controls Qwen routing and
synthesis. Native screening and chat protocol paths are unchanged; synthesis follows
the specialist workflow described above.

`--clef-timeout` bounds CLEF requests (default 10 s). `--triage-timeout`
bounds only CLEF's whole-pool stage (default 60 s, maximum 300 s).
The Qwen fallback has its own `--qwen-triage-timeout` (maximum 3600 s),
defaulting to packaged policy: 1800 s overall, 90 s per batch, two complete
papers per batch and one active batch. These are provisional settings, not
live-validated throughput claims. The hard bounds remain 24 papers, two
active batches, 64 KiB input and 4096 output tokens. Each batch permits at
most two agent requests including validation retry, within its own deadline;
SDK network retries are disabled. A failed batch no longer cancels unrelated
batches: successful judgments survive in the final persisted batch. Safe
typed failure categories distinguish connection/read/request timeouts,
invalid responses, service errors and exhausted stage budgets; raw upstream
bodies and error messages are never stored. A missing fallback configuration is exit 4;
failed/malformed/deadline Qwen screening is exit 3 and blocks synthesis,
preserving the stored paper pool. Source/metadata refresh is independent.
Missing abstracts and oversized texts stay explicit unknowns. SQLite
screening records retain backend/model/rubric and fallback reason;
strict v1 source snapshots are never mutated. `--collect-only`
and an empty pool call no routing or synthesis model. No servers are launched.

Model discovery uses asynchronous HTTPX inside the Qwen stage deadline and
closes its lookup client even on cancellation. OS hostname DNS resolution
can still delay CLI loop shutdown after a timeout; use the configured LAN IP
when a strict short screening deadline matters.

Earlier probing of FreeToken `/v1/systemone` was based on an incorrect
transport interpretation. Strata also exposes chat; the same
screening input and Pydantic answer contract are carried over that endpoint.

Exit codes: 0 ok, 2 external-service (OpenAlex) failure, 3 analysis/report
or storage failure (including Strata/Falkor), 4 usage/config error.

## Configuration (env)

| Var | Meaning | Default |
| --- | ------- | ------- |
| `STRATA_BASE_URL` | User-owned loopback or private-LAN endpoint | required unless `--base-url` is passed |
| `STRATA_MODEL` | Model id (skips `/models` lookup) | first id from `GET {base}/models` |
| `STRATA_API_KEY` | Local API key placeholder | `strata-local` |
| `STRATA_DISABLE_THINKING` | Opt-in server-specific thinking-disable key | unset (omitted) |
| `OPENALEX_API_KEY` | Optional personal OpenAlex budget | unset (shared anonymous pool) |
| `CLEF_BASE_URL` | Preferred user-owned LAN CLEF endpoint | unset (Qwen fallback) |
| `CLEF_MODEL` | CLEF model id | `clef-flash` |
| `RADAR_STORAGE_DIR` | SQLite/Falkor database directory | `data/radar` |

Loopback, RFC 1918, IPv6 ULA, and RFC 6598 endpoints are allowed. Public,
link-local, multicast, reserved, and unspecified destinations are refused;
LAN hostnames must resolve exclusively to allowed addresses. The Strata
server is user-owned: this tool never starts, stops, or alters it, and
there are no cloud-model fallbacks. Configured private-LAN example (no
secrets involved):
`--base-url http://192.168.1.20:8080/v1 --model <served-model-id>`
(or the equivalent `STRATA_*` env vars).

Explicit `--base-url` / `--model` values win over `STRATA_*` settings;
`FREETOKEN_*` settings are compatibility fallbacks only. `STRATA_DISABLE_THINKING=0`
can override a legacy thinking-disable setting. The retired provider module
re-exports the same contracts; it has no separate transport. No machine-specific
endpoint is assumed when neither explicit nor environment configuration is set.
The CLI does not automatically load `.env`; export it before running:

```sh
set -a
. ./.env
set +a
python -m radar --from-snapshot data/radar/snapshot.json
```

## Discovery, quota, and paper counts

- OpenAlex requests ≤ 6/plan, ≤ 50/page, ≤ 200 total pool cap, timeout
  ≤ 30 s. An `OPENALEX_API_KEY` is optional, but a free personal key
  avoids the exhausted shared anonymous pool; without one, confirmed
  daily-budget 429s fail fast with an actionable error instead of
  blind retries. Transient 429s get bounded retries. Source errors never
  carry URLs, credentials, or bodies.
- Three distinct counts, never conflated: **collected** (full bounded
  pool persisted by refresh), **selected/analyzed** (the shortlist slice
  the LLM actually saw), and **opportunities** (what the model proposed).
  CLEF screening adds its own buckets: **considered** (entire pool accounted
  for),
  **scored**, **unknown** (missing abstracts, oversize, deadlines), and
  **failed**, with model/rubric provenance on the coverage line.
  Metadata-only runs honestly report `llm_selected/llm_analyzed` as zero.
- Discovery is a bounded OpenAlex sample (lookback window applies to
  every query): counts describe the snapshot only, never all of OpenAlex.
  Cross-domain discovery is broad AI/ML plus behavioral/economic lenses.
  Abstracts/metadata only; no PDFs are downloaded, no fulltext, no
  extra services.

## Snapshots and cached synthesis

`--refresh-dir PATH` writes one atomic `snapshot.json`: full normalized
pool plus UTC timestamp, coverage counts, and an OpenAlex-ID-keyed delta
(`new`/`changed`/`unchanged`, ignoring timestamp, ordering, and derived
scores). Only strict producer-v1 snapshots are accepted as previous
state; anything else is refused with the file left untouched. Overlapping
refreshes are refused via a nonblocking lock; any collection, validation,
or persistence failure preserves the last valid snapshot and exits
nonzero. Runtime `data/radar/` is git-ignored.

`--from-snapshot` analyzes cached metadata with zero OpenAlex calls and
discloses snapshot staleness; it can never be combined with
`--refresh-dir`. Analysis runs under a hard overall deadline
(`--analysis-timeout`, default 90 s); `--disable-thinking` is strictly
opt-in and backend-specific.

## Report

Markdown sections per opportunity: **The Wow**, **Investigate**,
**Reproduce**, plus **Evidence** links attached deterministically from the
model's candidate indices (the model never writes URLs), a global
**Ignore** list, and a **Next move**.

## Baseline vs refactor status

Historical pre-CLEF baseline only (not current verification): 125/125
tests green with a 106-work refresh in 3s, cached 3-paper briefing in
17s, and gated 8-paper briefing in 19s on heuristic ranking plus Qwen
synthesis. The Qwen fallback originally passed 190/190 tests, including 19 typed
Qwen chat regressions and 106-work acceptance (102 synthetic decisions,
4 missing-abstract unknowns, shortlist 8). Mock tests are not real model
quality evidence. CLEF is absent and never launched; Qwen uses Strata chat.

Live checks on 2026-10-03 are not yet successful: full cached106 routing
and a three-paper probe each reached the60s deadline without validated
answers. Health/model listing returned200, but a16-token chat probe did
not complete within15s. The source snapshot remained byte-identical and
synthesis was not reached. These observations do not establish a server
root cause; no extra SystemOne endpoint is needed for Qwen.

## Tests

```sh
uv run python -m unittest discover -s tests/radar -v
```

Boundaries, all offline: `httpx.MockTransport` at the source HTTP seam,
`FunctionModel`/`TestModel` at the agent seam, real temp filesystems for
storage, CLI exit-code/output-shape tests at the pipeline boundary, and
import/dependency-boundary checks enforcing the dependency direction above.
