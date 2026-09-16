---
name: architecture-review
description: Survey a codebase for structural friction and high-value redesign candidates. Use for deliberate architecture maintenance, not as an automatic gate on ordinary changes.
---

# Architecture Review

Invoke `module-design` for the design vocabulary and read relevant domain language and decisions. Scope the survey to a user-named area or to recently changing hotspots; do not generate theoretical refactors across stable code.

Look for behavior scattered across callers, interfaces that expose too much internal knowledge, duplicated policy, sideways dependencies, test seams that miss real failures, and modules that add indirection without concentrating complexity.

For each credible candidate, report:

- the observed friction and evidence;
- the affected behavior and callers;
- a possible improved seam and what it would hide;
- expected gains in testability and locality;
- conflicts with existing decisions;
- recommendation strength and uncertainty.

When several candidates exist, present them in a self-contained visual HTML report in a dedicated temporary directory, including before/after structure and a top recommendation. Give the user its path, state that it is disposable, and open it only with authorization. Remove the directory after the user is finished unless a handoff depends on it; report the cleanup.

Do not refactor or fully design every candidate. Let the user select one, then use `design-session` for the focused interview. Use `git` to record approved follow-up work in GitHub; never create issues directly.
