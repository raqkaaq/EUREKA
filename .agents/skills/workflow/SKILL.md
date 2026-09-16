---
name: workflow
description: Orient the user or agent within the project's GitHub-first idea-to-merge lifecycle. Use for workflow reminders, project status, blockers, next-step decisions, or recommending the skill for the next phase.
---

# Workflow

This is a two-faced router. Choose the entry mode, recommend a route, then wait for the user's authorization. Never invoke the routed skill automatically.

## User-Facing Mode

**idea/design → evidence → specification → implementation issues → implementation → pull request → review → merge**

Use when the user asks for a reminder, status, blockers, or what to do next. Ask `git` for issues, pull requests, dependencies, and discrepancies. Explain the current phase and evidence; what is ready, active, blocked, under review, or awaiting them; and the next gate.

## Agent-Facing Mode

Use when the agent encounters a phase boundary or cannot determine the proper workflow. Inspect only enough project state to identify the next route. Present the recommendation and reasoning to the user instead of silently continuing.

## Routes

- Unformed choices → `brainstorm`; repository-grounded choices → `design-session`.
- Missing facts → `research`; executable uncertainty → `prototype`.
- Large, foggy route → `map-work`, returning through `specify` when clear.
- Settled design → `specify`; approved work needing multiple issues → `slice-work`.
- Raw incoming issue or pull request → `triage`; ready issue → `implement`.
- Hard defect → `debug`; completed change → `code-review`.
- Structural maintenance → `architecture-review`; context transfer → `handoff`.
- Any Git or GitHub operation → `git`.

Name one recommended skill, the evidence that its entry conditions are met, and its expected output. Ask the user to authorize that route and stop. A prior request to inspect status or choose a route is not authorization to perform the routed work.

Keep design through slicing in one conversation while its reasoning remains useful. Start each implementation issue fresh from its GitHub brief. At other boundaries, read [references/phase-boundaries.md](references/phase-boundaries.md).

GitHub issues and pull requests are the work-state source of truth; repository documents preserve design knowledge. Git and GitHub operations belong to `git`.
