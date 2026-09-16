---
name: git
description: Compose, review, and create focused Git commits and prs that follow Conventional Commits 1.0.0. Use when choosing commit boundaries, staging changes, writing or validating commit messages, or committing completed work.
---

# Git Commits

Follow repository instructions and existing conventions when they are stricter than this skill.

## Workflow

When creating a commit:

1. Inspect instructions, status, diffs, and recent subjects.
2. Keep one coherent concern per commit; split independent concerns only when authorized.
3. Choose the type and a concise title from the change's intent.
4. Stage only requested changes; exclude unrelated edits, generated files, credentials, and secrets.
5. Run appropriate checks, review the staged diff, and commit. Do not bypass hooks, push, amend, rebase, or rewrite history unless explicitly requested.
6. Inspect the new commit and status, then report its ID, title, and failed or skipped checks. If a hook fails, fix the issue when in scope and retry; do not assume a commit was created.

If asked only to suggest or review a message, do not modify the repository.

## Choose the Type From Intent

Choose the type from the purpose of the change, not the files touched:

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

## Message

Use the Conventional Commits 1.0.0 format for the title and body:

```text
<type>[optional scope][!]: <description>

[optional body]

[optional footer(s)]
```

- Add a scope only when useful, then a short imperative description without a trailing period.
- Add a body when the reason, behavior, migration, or tradeoff is not clear from the subject.
- Add footers after a blank line as `Token: value` or `Token #value`; replace spaces in ordinary tokens with hyphens.
- Mark breaking changes with `!` before the colon, a `BREAKING CHANGE: description` footer, or both. `BREAKING-CHANGE` is also valid.

For Semantic Versioning, `fix` corresponds to PATCH, `feat` to MINOR, and any breaking change to MAJOR. Do not assign release meaning to other types unless project tooling defines it.

## Pull Requests

Use a concise title describing the overall change. In the body, summarize the rationale, notable behavior, testing, and breaking changes; do not merely repeat the commit log.
