---
name: map-work
description: Clarify a large, uncertain effort through a dependency map of decisions and investigations. Use when the route to a buildable specification cannot fit in one design session.
---

# Map Work

Map uncertainty; do not start implementation.

1. Name the destination in observable terms.
2. Separate known decisions, open decisions, missing facts, and explicitly excluded territory. Keep foreseeable but not yet precise questions as **fog**, not tickets.
3. Invoke `interviewing` to build the dependency graph and expose only questions whose prerequisites are settled.
4. Handle user decisions through that interview; route missing facts to `research`, executable uncertainty to `prototype`, and other evidence gathering to a bounded analysis task. Do not re-enter `brainstorm` for a decision already represented in the map.
5. Fold each resolved question back into the map without duplicating its detailed evidence.

Create a ticket only when its question and completion condition are precise. Each decision task should fit a fresh context and end in a decision, not a production deliverable. Resolve at most one decision ticket per session, except independent research. Refer to tasks by meaningful titles rather than bare identifiers.

Use `git` to create or update the GitHub map, child issues, links, and state only when authorized. The map is complete when the remaining route can be collapsed through `specify` and, if needed, `slice-work`.
