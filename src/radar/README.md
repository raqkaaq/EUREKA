# AI/ML Opportunity Radar

One-command radar: discover recent AI/ML papers via **OpenAlex** (the sole
scholarly discovery API; arXiv appears only as an OpenAlex location string),
screen the full pool with **CLEF** (preferred) or **Qwen** (chat fallback),
then analyze the shortlist with three specialists and a synthesizer using
the local **FreeToken** model through **PydanticAI** (typed outcomes,
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
    searches.yaml      editable baseline search definitions
    searches.py        profile/template expansion, bounded query plan
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
    freetoken.py       LAN PydanticAI model construction + client lifecycle
    clef.py            LAN CLEF/SystemOne screening client (native HTTPX,
                       batch triage, deadlines, fail-fast)
  agent/
    research_team.py   common evidence cohort, specialists, shared deadline,
                       bounded contributions, synthesis and client cleanup
    paper_triage.py    same screening inputs/answer models over PydanticAI
                       FreeToken chat, complete-paper batches and deadlines
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
    snapshots.py       full-pool snapshots, deltas, coverage, strict v1
                       validation, atomic writes, locking
    triage.py          atomic triage sidecar (never mutates snapshots)
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

YAML is the source of truth for baseline search policy, shared screening
questions, and the agents' instructions. Python owns expansion, transport,
output schemas and hard safety limits; there is no second hardcoded search
algorithm or inline instruction fallback.

- `config/searches.yaml`: ordered definitions with `name`, `kind`
  (`semantic` = keyword relevance ranking, not embedding search; `recent` =
  newest first), `terms`, optional `scope`, `repeat`, `limit` and `per_page`.
  Defaults separately cover broad AI/ML, newest AI/ML, behavioral/economic
  intersections, generative-AI human/economic effects, methods/evaluation and
  causal/strategic mechanisms. Every request shares the active
  lookback filter. The planner lives in `config/searches.py`, replacing the
  old planner in the OpenAlex adapter.
- Search `{keywords}` and `{domains}` expand to OR-separated quoted profile
  phrases. `{keyword}` is available only with `repeat: keywords`; `limit`
  bounds that expansion. `scope: cross_domain` skips a definition when the
  profile has no domains. Profile terms are quoted/escaped as literal phrases.
  Identical requests are deduplicated (relevance/recent remain distinct), and
  expansion stops at the code-owned six-request cap. Overlong rendered queries
  are rejected, not silently truncated; split the configured query or shorten
  the profile instead. CLI `--keywords` and `--lookback-days` still apply.
- `prompts/screening_questions.yaml`: the fixed two named questions, their
  `type: noul`, instruction templates and criteria. `{keywords}`/`{domains}`
  use the existing screening profile formatting. Quote the YAML keys
  `"true"`/`"false"` so they remain strings. Native CLEF and Qwen chat share
  the same rendered dictionaries and unchanged Pydantic answer schema.
  `rubric_version: clef-triage-v2` identifies the expanded guidance; both
  backends record its canonical configuration SHA-256 as `rubric_hash` in
  triage sidecars. `version: 1` is separately the configuration schema version.
- `prompts/paper_triage.yaml`: Qwen screening-agent instructions.
- `prompts/opportunity_analysis.yaml`: synthesis-agent instructions,
  `candidate_header`, and `task_template`. Task placeholders are
  `{max_opportunities}` and `{valid_range}`; Python supplies the actual run
  bounds. All chat agents pass instructions through PydanticAI's `instructions`
  interface, separate from untrusted user/paper data. Combined instructions
  and user messages retain the existing prompt budget and complete-block policy.

Documents require `version: 1` (configuration schema version), are safely
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
1000 serialized characters. `ResearchResult` retains role-attributed reports,
the final draft/prompt and actual included papers for in-process callers.
The CLI renders the final report and names the roles in coverage notes;
specialist reports are not separately persisted. They remain untrusted
hypotheses, never source evidence or instructions.

At most two specialists run concurrently. Each stage permits two PydanticAI
requests including one validation retry, so analysis normally uses four
requests and at most eight; provider/SDK transport retries remain separate.
Specialist output caps are 1000 tokens (or the smaller CLI cap), and synthesis
retains the CLI cap. One `--analysis-timeout` covers all specialist and
synthesis calls. Failure cancels outstanding work, closes the owned session
and produces no partial-success report.

Every stage sees the same complete candidate blocks and indices. Selection
reserves 3500 characters for intermediate context, within the combined
12,000-character instruction/user-message budget. Richer prompts can reduce
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

It exercises fresh collection through all six searches, deduplication and
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
quality, server compatibility or a successful live research run. As of
2026-10-03 the operator is retiring FreeToken; live inference checks are stopped
pending replacement details. Existing provider configuration is retained,
not silently repointed or migrated to an unchosen service.

## Source vs provider vs agent

- **Source** (`source/openalex.py`) talks to the outside scholarly world:
  bounded OpenAlex requests over standard HTTPX, transient-429 backoff,
  confirmed-daily-quota fast-fail, and normalization into paper contracts.
  No LLM calls. Errors are redacted (no URLs, credentials, headers, or
  raw bodies).
- **Provider** (`provider/freetoken.py`) talks to your LAN machine: it
  validates the private-network endpoint, resolves the model id, and owns
  the inference HTTP client lifecycle (upstream internal `httpx2`,
  explicitly opened and deterministically closed per run). No prompts,
  no analysis.
- **Routing** (`provider/clef.py`, `agent/paper_triage.py`,
  `processing/triage.py`) decides
  *what* gets analyzed: the provider screens every collected paper over
  native HTTPX CLEF/SystemOne calls or PydanticAI Qwen chat calls (typed
  results, deadlines, bounded concurrency), and
  the pure shortlist policy ranks scored works with one reserved unknown
  slot (only when unknowns exist and the shortlist holds two or more).
  This is decision routing; the synthesis agent below does the thinking.
  CLEF is preferred. If absent or unsuccessful, the Qwen endpoint screens
  the entire original pool through the existing FreeToken chat endpoint.
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
uv run --env-file .env python -m radar --collect-only --max-candidates 8

# Full run: collect + SystemOne screening + local-model analysis as Markdown
# (requires working CLEF or the configured FreeToken Qwen chat service).
uv run --env-file .env python -m radar --max-candidates 8

# Unattended metadata refresh (full pool snapshot, no LLM).
uv run --env-file .env python -m radar --collect-only --refresh-dir data/radar

# Cached synthesis: zero OpenAlex calls (SystemOne screening still mandatory,
# needs a working routing endpoint plus FreeToken for synthesis).
uv run --env-file .env python -m radar --from-snapshot data/radar/snapshot.json --max-candidates 2

# Analysis with CLEF routing (CLEF server user-owned, not running yet).
# Set CLEF_BASE_URL to your LAN CLEF server; no endpoint is ever guessed.
uv run --env-file .env python -m radar --max-candidates 8 \
  --triage-output data/radar/triage

# Options.
uv run --env-file .env python -m radar --help
uv run --env-file .env python -m radar --keywords "graph neural networks" --lookback-days 30
```

## Screening (mandatory routing; CLEF preferred, Qwen chat fallback)

Every analysis run screens the full pool before shortlist selection and
FreeToken synthesis. CLEF uses `CLEF_BASE_URL` / `--clef-base-url` and
`CLEF_MODEL` (default `clef-flash`). When CLEF is absent, unavailable,
malformed, incomplete, or deadline-expired, the full original pool is
screened through FreeToken Chat Completions. Explicit unsafe/invalid
CLEF configuration is refused, not bypassed.

Qwen uses the existing `FREETOKEN_BASE_URL` / `--base-url` and
`FREETOKEN_MODEL` / `--model` configuration and provider. CLEF receives
native SystemOne inputs; Qwen receives batches of those same full inputs
as chat user-message content. Both validate `model` and the same two
`answers` (`type: noul`, strict finite numeric `noul` in 0..1) via shared
Pydantic models. Batch work IDs must match exactly, with one bounded
validation retry. The questions, criteria and full paper data are shared;
there is no heuristic substitute, rewritten decision task, or need for
FreeToken to expose `/v1/systemone`. PydanticAI controls Qwen routing and
synthesis. Provider endpoints are unchanged; normal synthesis now follows
the specialist workflow described above.

`--clef-timeout` bounds CLEF requests (default 10 s). Qwen chat requests
are bounded by 60 s (or the smaller remaining overall budget), with at
most two concurrent batches, 24 complete inputs per batch, 64 KiB input,
4096 output tokens, and two agent requests per batch including validation retry.
`--triage-timeout` bounds each backend's whole-pool stage (default 60 s,
maximum 300 s). A failed CLEF stage can therefore be followed by one
separately bounded Qwen stage. A missing fallback configuration is exit 4;
failed/malformed/deadline Qwen screening is exit 3 and blocks synthesis,
preserving the last valid snapshot. Source/metadata refresh is independent.
Missing abstracts and oversized texts stay explicit unknowns. Atomic
`--triage-output` JSON sidecars record backend/model/rubric and fallback
reason; strict v1 source snapshots are never mutated. `--collect-only`
and an empty pool call no routing or synthesis model. No servers are launched.

Earlier probing of FreeToken `/v1/systemone` was based on an incorrect
transport interpretation. FreeToken correctly exposes chat; the same
screening input and Pydantic answer contract are carried over that endpoint.

Exit codes: 0 ok, 2 external-service (OpenAlex) failure, 3 analysis/report
failure (including FreeToken), 4 usage/config error.

## Configuration (env)

| Var | Meaning | Default |
| --- | ------- | ------- |
| `FREETOKEN_BASE_URL` | User-owned loopback or private-LAN endpoint | required unless `--base-url` is passed |
| `FREETOKEN_MODEL` | Model id (skips `/models` lookup) | first id from `GET {base}/models` |
| `FREETOKEN_API_KEY` | Local API key placeholder | `freetoken-local` |
| `FREETOKEN_DISABLE_THINKING` | Opt-in server-specific thinking-disable key | unset (omitted) |
| `OPENALEX_API_KEY` | Optional personal OpenAlex budget | unset (shared anonymous pool) |
| `CLEF_BASE_URL` | Preferred user-owned LAN CLEF endpoint | unset (Qwen fallback) |
| `CLEF_MODEL` | CLEF model id | `clef-flash` |

Loopback, RFC 1918, IPv6 ULA, and RFC 6598 endpoints are allowed. Public,
link-local, multicast, reserved, and unspecified destinations are refused;
LAN hostnames must resolve exclusively to allowed addresses. The FreeToken
server is user-owned: this tool never starts, stops, or alters it, and
there are no cloud-model fallbacks. Configured private-LAN example (no
secrets involved):
`--base-url http://192.168.0.166:1919/v1 --model Qwen3.6-35B-A3B-NVFP4`
(or the equivalent `FREETOKEN_*` env vars).

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
quality evidence. CLEF is absent and never launched; Qwen uses FreeToken chat.

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
