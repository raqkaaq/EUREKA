---
name: agent-writing
description: Write or revise instructions and durable artifacts intended for coding agents. Use for skills, AGENTS.md, workflow references, issue templates, and other documents agents must reliably act on.
---

# Agent Writing

Write for repeatable decisions. Define the reader, trigger, changed behavior, and completion condition. A completion condition must be both checkable and demanding enough to prevent premature completion.

For skill invocation and router design, read [references/skill-mechanics.md](references/skill-mechanics.md).

Place information by loading cost:

- essential sequence and invariants in the main instructions;
- conditional or substantial detail behind clearly worded references;
- generated-output material in assets;
- deterministic repeated operations in scripts.

Keep one authority for each rule and say when a reference is needed. Use compact, established leading words to anchor repeated behavior. Remove generic advice, duplication, speculative cases, and permissions the document cannot grant. Prefer positive instructions. Validate names, frontmatter, links, and costly examples.
