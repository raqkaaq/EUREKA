# AI/ML Opportunity Radar

One-command radar: discover recent AI/ML papers via **OpenAlex** (the sole
scholarly discovery API; arXiv appears only as an OpenAlex location string),
then analyze them with a local **FreeToken** model through **PydanticAI**
(typed outcomes, Chat Completions path).

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
  source/
    openalex.py        HTTPX discovery, query plan/requests, retry/quota,
                       normalization, dedup, provenance
  provider/
    freetoken.py       LAN PydanticAI model construction + client lifecycle
  agent/
    opportunity_analysis.py  typed Agent[None, RadarDraft], prompt +
                       instructions (no separate prompt file), deadline/usage,
                       output validation
  processing/
    ranking.py         deterministic scoring/selection (pool → ranked topN)
    evidence.py        evidence index resolution/validation
  storage/
    snapshots.py       full-pool snapshots, deltas, coverage, strict v1
                       validation, atomic writes, locking
  output/
    markdown.py        Markdown report rendering only
    json.py            JSON output shaping only (collect-only envelopes)
  schema/
    papers.py          paper/plan contracts (CollectedWork and friends)
    opportunities.py   analysis/briefing contracts (RadarDraft and friends)
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
- **Agent** (`agent/opportunity_analysis.py`) does the thinking: it builds
  the prompt and instructions, runs the typed PydanticAI agent under a
  hard overall deadline with per-run token/request/usage caps, and
  validates the structured outcome. It never touches the network
  directly; the model never writes URLs (evidence indices only).

## Usage

```sh
# Collect-only: bounded real OpenAlex candidates as JSON (no LLM).
uv run --env-file .env python -m radar --collect-only --max-candidates 8

# Full run: collect + local-model analysis as Markdown.
uv run --env-file .env python -m radar --max-candidates 8

# Unattended metadata refresh (full pool snapshot, no LLM).
uv run --env-file .env python -m radar --collect-only --refresh-dir data/radar

# Cached synthesis: zero OpenAlex calls (needs FreeToken for analysis).
uv run --env-file .env python -m radar --from-snapshot data/radar/snapshot.json --max-candidates 2

# Analysis with CLEF routing (CLEF server user-owned, not running yet).
# Endpoint below is a placeholder: replace with your LAN CLEF server.
uv run --env-file .env python -m radar --max-candidates 8 \
  --clef-base-url http://192.168.1.20:1921/v1 --triage-output data/radar/triage

# Options.
uv run --env-file .env python -m radar --help
uv run --env-file .env python -m radar --keywords "graph neural networks" --lookback-days 30
```

## CLEF screening (mandatory routing, server absent)

Every analysis run CLEF-scores the full pool before shortlist selection
and FreeToken synthesis; there is no opt-out and no heuristic fallback
on service failure. Configure `CLEF_BASE_URL` (or `--clef-base-url`;
no endpoint is ever guessed), optionally `CLEF_MODEL` (default
`clef-flash`), `--clef-timeout` (per request, default 10 s), and
`--triage-timeout` (overall, default 60 s). Missing configuration is
exit 4 with zero model calls; unreachable/malformed/deadline/failed
screening is exit 3 and stops before synthesis with the last valid
snapshot preserved. Missing abstracts and oversize texts stay typed
unknowns with one unknown slot reserved; `--collect-only` never calls
CLEF or FreeToken. An optional atomic `--triage-output` JSON sidecar
records model/rubric provenance; strict v1 snapshots are never mutated.
The CLEF server is not launched by radar and live CLEF checks are
skipped while it is absent.

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
| `CLEF_BASE_URL` | User-owned LAN CLEF endpoint (**required** for analysis) | unset (no guess) |
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
  pool persisted by refresh), **selected/analyzed** (the ranked topN slice
  the LLM actually saw), and **opportunities** (what the model proposed).
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

Verified on the finished layout (`b925eb2`, 125/125 tests green):
metadata refresh of 106 real works in 3s with an idempotent rerun
(0 new / 0 changed / 106 unchanged), cached 3-paper briefing exit 0 in
17s, and gated 8-paper briefing exit 0 in 19s (selected=8 analyzed=8
opportunities=2 pool=106), all with evidence URLs inside the pool and
the snapshot preserved. The five-hour automation stays aligned to
validated paths only.

## Tests

```sh
uv run python -m unittest discover -s tests/radar -v
```

Boundaries, all offline: `httpx.MockTransport` at the source HTTP seam,
`FunctionModel`/`TestModel` at the agent seam, real temp filesystems for
storage, CLI exit-code/output-shape tests at the pipeline boundary, and
import/AST checks enforcing the dependency direction above.
