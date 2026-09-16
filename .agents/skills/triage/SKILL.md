---
name: triage
description: Evaluate incoming work and recommend its category, readiness, and next action. Use for new, ambiguous, blocked, or externally submitted issues and pull requests.
---

# Triage

Triage raw incoming issues and external pull requests. Work created by `specify` or `slice-work` should already be ready. `git` alone reads or changes GitHub state.

Use the repository's configured category and state mapping. When none exists, reason in these semantic roles:

- category: `bug` or `enhancement`;
- state: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, or `wontfix`.

Choose one category and one state. Conflicts require maintainer direction.

## Evaluate

1. Establish the claimed problem or outcome.
2. Search code and existing work for duplicates, prior implementation, or conflicting decisions. If prior rejection may apply, read [references/out-of-scope.md](references/out-of-scope.md).
3. Verify bugs against behavior and pull-request claims against the diff and checks.
4. Recommend category, state, and next route before asking questions.
5. For ready work, use [references/agent-brief.md](references/agent-brief.md).

Apply the shared [ready-work criteria](../workflow/references/readiness.md). Do not force incoming work through further questioning when the maintainer gives a direct state override.

On resume, preserve established facts and ask only unresolved questions. A maintainer override wins. Preview consequential mutations, then hand exact changes to `git`.
