---
name: git
description: Manage Git and GitHub project state, including issues, dependencies, labels, branches, commits, pull requests, reviews, merges, and conflicts. Use for any Git operation or any read or write involving GitHub Issues or pull requests.
---

# Git and GitHub

This skill alone performs Git and GitHub operations. Other skills supply issue drafts, dependency graphs, reviews, or transition recommendations.

GitHub Issues and pull requests are the work-state source of truth. Repository documents hold durable design knowledge.

## Shared Rules

- Read state when relevant; mutate only within the user's authorization.
- Resolve the exact repo, branch, worktree, issue, or PR first.
- Preserve unrelated work; never bypass protections or checks silently.
- Verify resulting state after every mutation.

## Route

- Branches, staging, commits, tags, pushes: [references/commits.md](references/commits.md)
- Issues, labels, dependencies, milestones: [references/issues.md](references/issues.md)
- PR drafts, reviews, checks, merges: [references/pull-requests.md](references/pull-requests.md)
- Project orientation and ready-work frontier: [references/status.md](references/status.md)
- Merge or rebase conflicts: [references/conflicts.md](references/conflicts.md)

Load only the procedure required. Report identifiers, links, checks, and unresolved state.
