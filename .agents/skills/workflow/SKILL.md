---
name: workflow
description: Orient the user or agent within the project's GitHub-first idea-to-merge lifecycle. Use for workflow reminders, project status, blockers, next-step decisions, or recommending the skill for the next phase.
---

# Workflow

This is a two-faced router. Choose the entry mode, recommend a route, then wait for the user's authorization before starting that next phase. Supporting skills may gather read-only evidence needed for the recommendation; do not perform mutations or begin the recommended workflow implicitly.

## User-Facing Mode

**idea/design → evidence → specification → implementation issues → implementation → pull request → review → merge**

Use when the user asks for a reminder, status, blockers, or what to do next. Ask `git` for issues, pull requests, dependencies, and discrepancies. Explain the current phase and evidence; what is ready, active, blocked, under review, or awaiting them; and the next gate.

## Agent-Facing Mode

Use when the agent encounters a phase boundary or cannot determine the proper workflow. Inspect only enough project state to identify the next route. Present the recommendation and reasoning to the user instead of silently continuing.

## Routes

- Unformed, artifact-free choices → `brainstorm`; choices that must be reconciled with current code or durable project decisions → `design-session`.
- Missing facts answerable from authoritative sources → `research`; uncertainty whose answer requires running an experiment → `prototype`.
- Large, foggy route → `map-work`, returning through `specify` when clear.
- Settled design → `specify`; approved work needing multiple issues → `slice-work`.
- Raw incoming issue or pull request → `triage`; an issue satisfying the shared [ready-work criteria](references/readiness.md) → `implement`.
- Hard defect → `debug`; completed change → `code-review`.
- Structural maintenance → `architecture-review`; context transfer → `handoff`.
- Any Git or GitHub operation → `git`.

Name one recommended skill, the evidence that its entry conditions are met, and its expected output. Ask the user to authorize that route and stop. A request to inspect status or choose a route authorizes the read-only inspection needed to answer, but not the recommended next phase.

If new evidence or code changes invalidate an existing specification, return to `design-session` when a decision must be reopened or to `specify` when only the recorded contract is stale.

Keep design through slicing in one conversation while its reasoning remains useful. Start each implementation issue fresh from its GitHub brief. At other boundaries, read [references/phase-boundaries.md](references/phase-boundaries.md).

GitHub issues and pull requests are the work-state source of truth; repository documents preserve design knowledge. Git and GitHub operations belong to `git`.
