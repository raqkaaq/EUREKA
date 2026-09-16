---
name: code-review
description: Review a change or pull request against both its originating requirements and repository engineering standards. Use for completed or in-progress diffs that need evidence-backed findings.
---

# Code Review

Pin the target and fixed point. For branch work, use the merge base and record the commits reviewed. Stop on an invalid reference or empty diff. Use `git` for pull-request context.

Find requirements in linked issues or commits, then user-provided or repository specifications. Ask only if none can be found; never invent them.

Read [references/standards.md](references/standards.md) for the baseline engineering heuristics.

Run two independent passes:

1. **Requirements:** missing or contradictory behavior, unrequested scope, silent decisions, and absent acceptance evidence.
2. **Engineering:** correctness, security, boundaries, tests, complexity, interfaces, documentation, and observability.

Report only actionable, evidenced findings. Rank within each pass; do not globally rerank them. Include residual risks and checks not run. Do not change code or publish a review unless requested; publication belongs to `git`.
