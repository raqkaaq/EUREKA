# AI/ML Opportunity Radar

One-command radar: discover recent AI/ML papers via **OpenAlex** (the sole
scholarly discovery API; arXiv appears only as an OpenAlex location string),
screen the full pool with **CLEF** (preferred) or **Qwen** (chat fallback),
then analyze the shortlist with a local **FreeToken** model through
**PydanticAI** (typed outcomes, Chat Completions path).

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
    triage_questions.py identical routing questions/criteria for both endpoints
  source/
    openalex.py        HTTPX discovery, query plan/requests, retry/quota,
                       normalization, dedup, provenance
  provider/
    freetoken.py       LAN PydanticAI model construction + client lifecycle
    clef.py            LAN CLEF/SystemOne screening client (native HTTPX,
                       batch triage, deadlines, fail-fast)
  agent/
    paper_triage.py    same screening inputs/answer models over PydanticAI
                       FreeToken chat, complete-paper batches and deadlines
    opportunity_analysis.py  typed Agent[None, RadarDraft], prompt +
                       instructions (no separate prompt file), deadline/usage,
                       output validation
  processing/
    triage_input.py    canonical model/state/questions input for both backends
    ranking.py         deterministic scoring/selection (pool → ranked topN)
    evidence.py        evidence index resolution/validation
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
```

Dependency direction: `cli → pipeline → {config, source, provider, agent,
processing, storage, output, schema}`; adapters never depend on
cli/pipeline/output; `schema` and `config` are service-free. No plugin
framework.

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
- **Agent** (`agent/opportunity_analysis.py`) does the thinking: it builds
  the prompt and instructions, runs the typed PydanticAI agent under a
  hard overall deadline with per-run token/request/usage caps, and
  validates the structured outcome. It never touches the network
  directly; the model never writes URLs (evidence indices only).

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
synthesis; the existing synthesis provider/agent wiring is unchanged.

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
synthesis. Current offline status: 190/190 tests green, including 19 typed
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
