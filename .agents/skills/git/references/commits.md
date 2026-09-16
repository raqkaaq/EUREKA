# Commits

Inspect instructions, branch, status, diff, recent subjects, and pre-existing changes.

For each commit:

1. Define one coherent concern.
2. Stage only its files or hunks; exclude unrelated edits, generated files that are not intended outputs, credentials, secrets, and runtime data.
3. Run relevant checks and review the staged diff.
4. Commit without bypassing hooks.
5. Verify the commit and remaining worktree.

Use Conventional Commits:

```text
<type>[optional scope][!]: <imperative description>

[optional body]

[optional footer(s)]
```

Choose type from the purpose, not the files touched:

- `build`: build system, packaging, or dependencies
- `chore`: maintenance with no more specific type
- `ci`: continuous integration
- `docs`: documentation only
- `feat`: user-visible or externally usable capability
- `fix`: faulty behavior
- `perf`: performance without intended behavior changes
- `refactor`: restructuring without behavior changes
- `revert`: reversal of an earlier commit
- `style`: formatting only, not product UI changes
- `test`: tests without production behavior changes

Add a scope only when useful, then a short imperative description without a trailing period. Add a body when rationale, behavior, migration, or trade-offs are not clear from the subject. Add footers after a blank line as `Token: value` or `Token #value`; replace spaces in ordinary tokens with hyphens. Mark breaking changes with `!`, a `BREAKING CHANGE:` or `BREAKING-CHANGE:` footer, or both. Include required issue or research citations.

For Semantic Versioning, `fix` means PATCH, `feat` means MINOR, and a breaking change means MAJOR. Other types have no release meaning unless project tooling assigns one.

If a hook fails, fix the problem when it is in scope and retry; do not assume the commit exists.

Fetch, pull, push, amend, rebase, tag, set upstreams, or rewrite history only when authorized. Never force-push implicitly. If asked only for a message, change nothing.
