# PRD — sci-eureka — LOCKED v1.0

> Evidence, Understanding, Replication, Extraction, Knowledge & Attribution.
> Written by the orchestrator (manual). All coding and test writing below is
> delegated to subagents. Orchestrator writes no implementation code.

## 0. Laws (apply to every phase)

- KISS + SOLID + YAGNI. Simplest code passing gates. Flat functions, early
  returns. Shape in `models/` only, LLM in `services/` only, I/O in
  `clients/` + `store/` only. Shared pure helpers in `common/` only
  (`text, ids, io, time`), no sideways imports. DRY on third occurrence.
- Python-only. `uv` + Python 3.14 + `uv.lock`. `uv run` only, **never pip**.
  No shell scripts, no system binaries (`s3cmd`, `latexpand`, TeXLive CLI).
  Cypher as `str` consts in `store/`.
- Setup-only (no serving changes; FreeToken/Qwen3.6-35B at
  `127.0.0.1:1919/v1` is user-owned). Local-only now; OpenRouter future via
  `OPTIMIZER_*` env swap.
- **Nobody makes an ontology here.** Reuse pinned standards only. SHAPE
  governs, extraction conforms, never defines. LLM never defines schema.
- Autonomous-safe: quarantine-don't-block, read-only snapshot tries,
  budgets enforced, audit everything, paper routing mandatory (Section 9).

## 1. Goal

Autonomous (user-absent) local researcher for cs/math/physics (bio/chem
out of scope; math is a subset of science, extracted deepest) that:

1. Searches/fetches arXiv via SDK (Project Euclid + S3 bulk later),
2. Builds an ontology-grounded knowledge graph in Kuzu (browsable),
3. Chats with citations (search/fetch tools shared with chat),
4. Improves via SkillOpt (both skills),
5. Later: draft-math agent + paper/math replication + Lean validation
   (AVO outer loop then; stubs only now).

Accuracy + coverage are both required. Entity linking quality is crucial.

## 2. Packages (`sci-eureka`)

- Distribution: `sci-eureka`. Import: `sci_eureka`. CLI: `eureka`
  (`uv run eureka ...`).
- Layout (zero floaters; everything inside one package):

```text
sci-eureka/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/                  # gitignored runtime state (raw, kuzu, quarantine)
└── src/
    └── sci_eureka/
        ├── __init__.py
        ├── clients/       # arxiv.py, search.py (external I/O only)
        ├── pipelines/     # ingest.py (fixed S0→S7 orchestration, no planner)
        ├── models/        # ontology.py (FROZEN consts), chunks.py, extractions.py
        ├── services/      # extraction.py, linking.py, qa.py (LLM/business logic)
        ├── store/         # kuzu.py, queries.py (Kuzu backend only)
        ├── skills/        # registry.py, arxiv_extract.py, kuzu_qa.py (emit SKILL.md)
        ├── evals/         # datasets.py, scorers.py (emit JSONL; shared f)
        ├── observability/ # logging.py (Logfire local-only)
        ├── security/      # delegation.py (R1–R8 v0-lite)
        └── common/        # text.py, ids.py, io.py, time.py (pure fns only)
```

- External-tool artifacts (SKILL.md / YAML / JSONL) are *generated* from
  `skills/` + `evals/` Python (single source of truth), never hand-edited
  floaters. Future `draft/` + `replication/` subpackages later (stubs only).

## 3. Runtime / ingest (PDF-first, LaTeX-opportunistic)

- `arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)` reused +
  tenacity backoff + `fetch_ledger.jsonl`. `search_arxiv(query, max<=20,
  sort=SubmittedDate)` + `fetch_paper(id)`. Shared `arxiv` toolset used by
  ingest AND chat. SlowAPI rejected (incoming-throttle, wrong layer).
- Per paper: try `source_url()` → in-Python `gzip+tarfile` unpack, find
  `main.tex` (`\documentclass` + `\begin{document}`); always fetch `pdf_url`.
  Record `branch{tex+pdf|pdf-only}` in manifest. `data/raw/` immutable.
  SDK now; S3 bulk (`boto3`, requester-pays `us-east-1`, manifest) deferred.
- Project Euclid is a deferred mathematics source. Use only content available
  without a paid subscription; retain publisher, journal, DOI, license/access,
  and source attribution metadata.
- S1: Docling LaTeX + TexSoup/arXiTeX (macros, `\input` cap 10) /
  PDF fallback Docling PDF `do_formula_enrichment` + PyMuPDF bbox.
- S2 chunks: section-aware (`title,abstract,intro,related,background,method,
  model,theory,experiments,results,discussion,limitations,conclusion,appendix`;
  exclude `references,acks`), `max_chars=2000 / overlap=200 / hard_max≈512
  tok-eq` (chars are truth), hierarchy `paper>section>subsection>block>chunk`,
  atomic theorem/lemma/proof/equation/caption/table (whole + `is_overflow`
  + `NEXT_CHUNK`), dual spans `pdf{page,bbox} + tex{path,span}`.
  Gate: orphan chars <2%, order monotonic.
- S3 anchors: tex `\label/\ref/\cite` + envs; pdf `Theorem 3.2 / Eq.(4)` +
  proximity linking. Gate: unresolved `\ref` >20% → re-expand (tex).
- S4 narrow LLM: 1 call/chunk, `output_type=Pydantic`, `temp=0`,
  `reasoning=none`, `retries≤2` else quarantine. JSON-schema + domain/range
  allowlist in prompt. LaTeX strings verbatim (`edit_distance>0` reject).
- S5 math-deep per equation: tex
  `{label,env,enclosing_theorem,is_boxed,verbatim,file:line,mathml?,
  py_parse_ok}` (Python `pylatexenc` check only, flag otherwise — no TeXLive);
  pdf `{page,bbox,mathml/text,numbered?,enclosure,confidence}` (best-effort,
  flag low-conf, never silent drop). Authority: proposition = stated result,
  proof = argument.
- S6 check (deterministic): typing + domain/range vs frozen shape, no
  self-loops/dangling/unknown. 0 invalid writes; rejects logged.
- S7 dedup + upsert (deterministic + Sci-ZSEL linking, Section 5):
  normalize + `RapidFuzz>92` + embedding top-10 same-type → synonym judge →
  `MERGE` canonical + `aliases[]`. Batch entities then edges
  (`2000 nodes / 5000 rels per txn`), single `kuzu.Database`, serial txn,
  `CHECKPOINT` per ~20 papers. Gate: re-run delta=0 + sample `MATCH`.

## 4. Ontology SHAPE v0.1.0 (frozen, rigor now, reuse-only)

- Direction: single-shape primary **Schema.org**
  (`ScholarlyArticle/Dataset/Person`, `author/citation/about/hasPart`).
- Linking `E` (closed union, pinned SKOS, versioned):
  **CSO v3.5** (cs — ACM CCS 2012 ruled stale/pre-LLM) ∪ **MSC2020** (math) ∪
  **PhySH** (physics, CC0). Math kinds use OMDoc names, citations use CiTO
  names, provenance uses PROV-O names — as string labels, not imports.
  License heterogeneity noted (CC-BY / CC-BY-NC-SA / CC0).
- Wikidata: **inform-only for `search_arxiv`/websearch query expansion**
  (local snapshot, tagged `wikidata-inform`, capped). Never a link target,
  never schema, never a gate. Off flag available.
- Tables (PK `id: STRING`, all carry `provenance{source_branch,
  extractor_version}`): `Paper, Author, Venue, Chunk, Claim(result|assumption|
  limitation), Method, Dataset, MathArtifact(Theorem|Lemma|Proposition|
  Corollary|Definition|Axiom|Proof|ProofStep|Equation|Example|Theory),
  Concept, Mention`.
- Rels (all require `evidence{chunk_id,span,page/bbox or tex_span}`):
  `AUTHORED, PUBLISHED_IN, HAS_CHUNK, NEXT_CHUNK, CITES(cito_type),
  STATES, PROVES, DEFINES, USES_METHOD, USES_DATASET, ABOUT, SUPPORTED_BY,
  DEPENDS_ON, GROUNDS, REFERS_TO`. Closed world; `schema_viol 0/1k`;
  `link_precision>0.85 (n=50)`; `orphan<0.15`. Expansion only via versioned
  migration + human gate; unknown → `Concept{qid=null}` quarantine.

## 5. Linking S7 (Sci-ZSEL verbatim intent — see paper links §9)

`E_EM ∪ E_BT` entity-side (Eq1–2); prompt on definition `d(e)` (`{domain}` +
3 in-`E` examples); `P1 + P2 + PSyn` (Eq3/8–10); neighbor filter Eq4–7
(`s_anchor` vs `s_drift` over parents/children/siblings, `τ=0.9`, local
cosine); retriever `PC-POS`; reranker top-K=64 + augmentation; HO/LO/NO
tracked. Filter is load-bearing (unfiltered regresses). LLM proposes,
ontology filter disposes.

## 6. Chat / browsing

Tools only `vector_search(k 8–12) + graph_lookup(hops 2)` (we build Cypher),
RRF fuse. `Planning(Sqlite) + SubAgents(budgets: req~10/tok~20k/timeout
300s/max_calls 50/contain_errors) + ToolOutputLimits(spill 10k) +
TieredCompaction + Guardrails(citations required, read-only snapshot, hidden
destructive tools) + Skills deferred + StepPersistence minimal`.
Must-cite `[arxiv_id:chunk]` or abstain. `search_arxiv`/`fetch_paper` shared
with ingest (chat can fetch-then-extract on demand under same budgets).
Explorer `:8000` read-only snapshot (`uv run eureka explore`).

## 7. SkillOpt (all, sequential)

`arxiv-extract` first (on frozen snapshots), then `kuzu-qa`.
`openai_compatible` → FreeToken local Qwen student+optimizer (OpenRouter
teacher later via `OPTIMIZER_*` env swap). Adapters `~200 lines` each
(`load_split_items`, `run_batch→{hard,soft}+conversation.json`, 4-method
`EnvAdapter`). Paper/entity-disjoint splits `~80–150 train / 30–50 val /
50+ test`. `lr 2–4 edits`, rejected-buffer, slow/meta update, strict `>`
gate. Shared `scorers{correctness,acc,cov}`,
`S=2·acc·cov/(acc+cov)−λ·cost`; correctness fail = 0. Artifact
`best_skill.md` (300–2000 tok) → `SKILL.md`. Zero deploy cost.

## 8. Observability + security

Logfire **local-only** (console/file exporter, redact latex/pdf payloads,
off by default via env): agent/model/tool/Kuzu spans, tokens/cost/latency,
`DEBUG arxiv.arxiv` logger piped. Delegation R1–R8 **v0-lite now**
(budgets attenuation, hidden tools, read-only snapshot, local audit ledger
with `unknown_after_crash`); full capability-token broker future with
OpenRouter/MCP.

## 9. Papers (mandatory per task — cite sections in commits)

- S7 linking: `2609.00228 Sci-ZSEL` — https://arxiv.org/abs/2609.00228 ,
  https://arxiv.org/html/2609.00228v1 , https://arxiv.org/pdf/2609.00228 ,
  code https://github.com/rasel-isu/Sci-ZSEL .
  Read §5.1 Eq1–2, §5.2 Eq3/8–10 + Eq4–7 filter, §5.3/App A.3,
  §5.4/App A.4, §3/§6 HO/LO/NO. Any linking change cites Eq number.
- Security: `2609.00267 Delegation Without Trust` —
  https://arxiv.org/abs/2609.00267 ,
  https://arxiv.org/html/2609.00267v1 , https://arxiv.org/pdf/2609.00267 .
  Read §3 T1–T4 + untrusted-model, §4 R1–R8, §7 broker. Any
  delegation/tool change maps to R#.
- Skills: `SkillOpt arXiv:2605.23904` https://arxiv.org/abs/2605.23904 ,
  repo https://github.com/microsoft/SkillOpt ,
  guideline https://microsoft.github.io/SkillOpt/docs/guideline.html .
  Gate proof required; never live-edit skills in ingest.
- Future evolution (deferred, no work): compare `AVO arXiv:2603.24517`
  https://arxiv.org/abs/2603.24517 (+ §3 `Vary(P)=Agent(P,K,f)`) with
  `AlphaEvolve arXiv:2506.13131`
  https://arxiv.org/abs/2506.13131 and
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
  before selecting the Auto Research evolution design.
- Future verification (deferred, no work): evaluate
  `LLM-as-a-Verifier arXiv:2607.05391`
  https://arxiv.org/abs/2607.05391 and code
  https://github.com/llm-as-a-verifier/llm-as-a-verifier for candidate
  selection, progress scoring, and feedback. Benchmark it for EUREKA rather
  than treating verifier scores as ground truth.
- Math precedent: `TheoremGraph 2606.25363`
  https://arxiv.org/abs/2606.25363 + ArXiTeX
  https://github.com/uw-math-ai/arXiTeX (§3 env/label/body/proof).
- Replication future (deferred): `PaperBench 2504.01848`
  https://arxiv.org/abs/2504.01848 .
- Ontologies (pinned, reuse): CSO v3.5
  https://cso.kmi.open.ac.uk/downloads , MSC2020
  https://mathscinet.ams.org/mathscinet/msc/msc2020.html , PhySH
  https://physh.org/ + https://github.com/physh-org/PhySH , Schema.org
  https://schema.org/docs/schemas.html , SPAR
  http://www.sparontologies.net/ , OMDoc https://kwarc.info/systems/omdoc .
- Future paper source: Project Euclid https://projecteuclid.org/ (open-access
  content only; no subscription dependency).

## 10. Autonomy (user-absent execution contract)

Decide alone within shape+budgets. On uncertainty: quarantine to
`data/quarantine/<date>/` + `needs_review=true` + continue next paper.
Never delete `data/raw/`, widen tool scope, cloud-send, evolve ontology,
or force-commit. Resume via ledger + snapshots. `run_report.md` per batch
(papers in/out, branch, counts, link proxy, orphan, HO/LO/NO, quarantined
IDs, cost). Done = acceptance below.

## 11. Acceptance (offline CI)

`uv sync --offline` clean; `uv run --offline` green; `Pydantic pass>95%`;
`schema_viol 0/1k`; `link_precision>0.85`; `orphan<0.15`; `cited-answer>90%`
val; re-run delta=0; Explorer loads snapshot; no non-Python scripts;
`uv run` only; paper citations present in touching commits; extensive edge
tests per phase (empty PDF, single-`.tex.gz`, tar-flat, withdrawn/PDF-only,
macro-heavy tex, 429/retries, corrupt PDF, math-only paper, no-abstract,
huge section overflow, dup papers, ambiguous aliases, missing QIDs).

## 12. Build order P0–P6 (subagents do ALL coding + tests; orchestrator only)

- P0 scaffold (`eureka` layout, `uv.lock`, vendor manifest, PRD copy).
- P1 `clients/` (search/fetch + ledger + budgets) + edge tests.
- P2 `pipelines/` + `models/` + `services/extraction` + `store/` S1–S7 +
  gates + edge tests.
- P3 `services/qa` + `chat` + `explore` + shared tools wiring + edge tests.
- P4 EL baseline gold (150–300 mentions) + frozen `P1+PSyn+BLINK` before
  any training + edge tests.
- P5 SkillOpt adapters + sequential runs + gate proof + edge tests.
- P6 Logfire-local + audit + `run_report` + edge tests.
- Deferred stubs only: Project Euclid, S3 bulk, full broker, AlphaEvolve vs
  AVO evaluation, LLM-as-a-Verifier evaluation, Lean replication, draft-math.
