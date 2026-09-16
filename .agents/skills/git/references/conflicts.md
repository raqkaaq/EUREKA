# Conflicts

Resolve an active merge or rebase by intent:

1. Inspect operation state, history, and all conflicted paths.
2. Trace both sides to commits, issues, PRs, tests, and decisions.
3. Preserve both intents when compatible; otherwise follow the operation's goal and note the trade-off. Add no unrelated behavior.
4. Stage only resolved paths; confirm no markers or unmerged entries remain.
5. Run focused then broad integration checks.
6. Continue through remaining commits and verify final history and worktree.

Finish the operation. Abort, discard, force-push, or rewrite additional history only on explicit request.
