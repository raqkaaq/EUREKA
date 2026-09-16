---
name: handoff
description: Preserve enough state for another agent or session to continue active work. Use at a genuine context, agent, repository, or phase boundary.
---

# Handoff

Summarize the active objective, authoritative issue or pull request, completed work, settled decisions, evidence, commands and checks run, working-tree state, blockers, remaining acceptance criteria, and exact next action.

Reference existing issues, pull requests, specifications, commits, and documents instead of duplicating their contents. Redact secrets and unnecessary personal information. Clearly distinguish verified facts from assumptions and recommendations.

Prefer the active GitHub issue or pull request as the durable handoff location. Invoke `git` to publish or update it only when authorized. Use a separate handoff file only when crossing to a context that cannot access the project artifacts or when the user specifically requests one.
