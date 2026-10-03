# AI/ML Opportunity Radar (prototype)

One-command radar: discover recent AI/ML papers via **OpenAlex** (the sole
scholarly discovery API; arXiv appears only as an OpenAlex location string),
then analyze them with a local **FreeToken** model through **PydanticAI**
(structured output, Chat Completions path).

## Official API basis (verified 2026-09-30)

- Package: `pydantic-ai-slim[openai]` (per
  https://ai.pydantic.dev/models/openai/ and
  https://ai.pydantic.dev/models/compatible-apis/).
- Custom OpenAI-compatible endpoint: `OpenAIChatModel(model_name,
  provider=OpenAIProvider(base_url=..., api_key=...))`
  (see "Other endpoints" in the compatible-APIs guide).
- Structured output: `Agent(model, output_type=RadarDraft)`; `run_sync(...).output`
  is a validated `RadarDraft`.
- Tests inject `TestModel`/`FunctionModel` (see
  https://ai.pydantic.dev/guides/testing/) — no network, no LLM.

## Networking (httpx)

- All direct HTTP (OpenAlex discovery, FreeToken `/models`) uses standard
  `httpx` (direct dependency), injected via `httpx.MockTransport` in tests.
  OpenAlex clients ignore proxy env and never follow redirects; transport
  errors are redacted (no URLs, credentials, headers, or bodies).
- Inference stays PydanticAI-only (`Agent[None, RadarDraft]` over
  `OpenAIChatModel`/`OpenAIProvider`; no raw SDK calls). The provider
  receives an explicitly owned `httpx2.AsyncClient` (the PydanticAI 2.52
  typed seam; the legacy `httpx.AsyncClient` path warns), closed
  deterministically after each run. Per-run `max_tokens`/timeout/usage
  limits apply; the model never writes URLs (evidence indices only, with
  the valid `0..N-1` range stated in the prompt).

## Usage

```sh
# Collect-only: bounded real OpenAlex candidates as JSON (no LLM).
uv run python -m radar --collect-only --max-candidates 8

# Full run: collect + local-model analysis as Markdown.
uv run python -m radar --max-candidates 8

# Options.
uv run python -m radar --help
uv run python -m radar --keywords "graph neural networks" --lookback-days 30
```

## Configuration (env)

| Var | Meaning | Default |
| --- | ------- | ------- |
| `FREETOKEN_BASE_URL` | User-owned loopback or private-LAN endpoint | required unless `--base-url` is passed |
| `FREETOKEN_MODEL` | Model id (skips `/models` lookup) | first id from `GET {base}/models` |
| `FREETOKEN_API_KEY` | Local API key placeholder | `freetoken-local` |
| `FREETOKEN_DISABLE_THINKING` | Opt-in server-specific thinking-disable key | unset (omitted) |
| `OPENALEX_API_KEY` | Optional polite OpenAlex pool | unset |

Loopback, RFC 1918, IPv6 ULA, and RFC 6598 endpoints are allowed. Public,
link-local, multicast, reserved, and unspecified destinations are refused;
LAN hostnames must resolve exclusively to allowed addresses. The FreeToken
server is user-owned: this tool never starts, stops, or alters it, and there
are no cloud-model fallbacks. Configure it, for example, with
`FREETOKEN_BASE_URL=http://192.168.1.20:1919/v1`.

## Report

Markdown sections per opportunity: **The Wow**, **Investigate**,
**Reproduce**, plus **Evidence** links attached deterministically from the
model's candidate indices (the model never writes URLs), a global **Ignore**
list, and a **Next move**.

## Bounds

OpenAlex requests ≤ 6/plan, ≤ 50/page, ≤ 200 total (hard candidate-pool
cap), timeout ≤ 30 s; candidates ≤ 25 (`--max-candidates` bounds the final
ranked output slice, never plan execution); prompt ≤ 25 candidates /
12 000 chars, at most 2 opportunities with short fields; per-run model
output ≤ 2000 tokens (`--max-tokens` 128..8000), request limit 2,
validation retries 0; overall analysis deadline 90 s
(`--analysis-timeout` (0, 300]); per-request model timeout fits inside it.
Transient OpenAlex HTTP 429 responses are retried at most twice; valid
`Retry-After` values are capped at 45 seconds, while missing or malformed
values use a two-second backoff.

## Cached snapshot synthesis (no OpenAlex calls)

```sh
# Inspect a cached snapshot (no network, no LLM).
uv run python -m radar --collect-only --from-snapshot data/radar/snapshot.json

# Analyze cached metadata (zero OpenAlex calls; needs FreeToken).
uv run python -m radar --from-snapshot data/radar/snapshot.json --max-candidates 2

# Known-working LAN synthesis (model must be served; SYNTHETIC fixture label
# applies until a free OpenAlex key restores real discovery):
uv run python -m radar --from-snapshot data/radar/live/fixture_synthetic.json \
  --max-candidates 2 --base-url http://192.168.0.166:1919/v1 \
  --model Qwen3.6-35B-A3B-NVFP4 --disable-thinking
```

`--from-snapshot` loads a valid producer-v1 snapshot, reproducibly selects
the ranked topN (score desc, OpenAlex ID asc), and discloses the collection
timestamp/staleness on stderr. Invalid sources are rejected before any
model call; `--from-snapshot` + `--refresh-dir` is refused (cached
synthesis never rewrites works). Coverage updates to a refreshed snapshot
are generation-aware: a stale run never stomps fresher coverage.

## Opt-in thinking disable

`--disable-thinking` (or `FREETOKEN_DISABLE_THINKING=1`) sends the
server-specific `extra_body.chat_template_kwargs.enable_thinking=false` key
(probed once against the LAN FreeToken: accepted, honoring not confirmed).
The key is omitted by default and assumed supported by no backend; a server
rejection produces a clear rerun-without-it error. Never used to force
empty or fake output: few or zero opportunities remain valid.

## Lookback

`--lookback-days` (default 90, 1..3650) sets `from_date` on **every**
discovery query: each request carries
`filter=from_publication_date:<today-lookback>`. Semantic queries omit
`sort` so OpenAlex relevance ranking applies; only the recent query adds
`sort=publication_date:desc` for newest-first ordering.

## Metadata-only refresh (no LLM)

```sh
# Unattended metadata refresh: persists the FULL bounded pool as one
# atomic snapshot (data/radar/snapshot.json). No FreeToken needed.
uv run python -m radar --collect-only --refresh-dir data/radar

# Refresh, then run the optional FreeToken synthesis separately.
uv run python -m radar --max-candidates 8
```

`--refresh-dir PATH` writes a single `snapshot.json` containing the full
normalized pool (metadata, abstracts, locations, provenance) plus a UTC
timestamp, explicit coverage counts (`collected`, `with_abstracts`,
`missing_abstracts`, `llm_selected`, `llm_analyzed` -- zeros for
metadata-only runs), and an OpenAlex-ID-keyed delta (`new`/`changed`/
`unchanged` counts). Reruns with an unchanged pool report zero changes:
delta identity ignores snapshot timestamp, work ordering, and the derived
ranking score. Only producer schema v1 snapshots are accepted as the
previous state; anything else (unknown schema, malformed entries, missing
or duplicate OpenAlex IDs) is refused with a clear error and the previous
file is left untouched, never silently healed. Works with missing abstracts are retained and counted, never dropped.
Overlapping refreshes are refused via a nonblocking lock; collection,
validation, or persistence failures preserve the last valid snapshot and
exit nonzero. Runtime `data/radar/` is git-ignored.

## Coverage limitations

Discovery is a bounded OpenAlex sample (<= 6 requests, <= 50/page,
<= 200 pooled works, `--lookback-days` window): coverage counts describe
the snapshot only, never all of OpenAlex. Cross-domain discovery is
broad AI/ML plus behavioral/economic lenses. No PDFs are downloaded.

## Next work

Batched full-pool triage, evals, and a personal interest profile. Real
OpenAlex discovery stays blocked until a free API key is configured
(currently quota-exhausted); synthetic-fixture runs are not end-to-end
passes.

## Tests

```sh
uv run python -m unittest discover -s tests/radar -v
```
