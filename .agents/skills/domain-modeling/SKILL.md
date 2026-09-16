---
name: domain-modeling
description: Sharpen the project's shared language and durable domain decisions. Use when terms are vague, overloaded, contradictory, or changing, or when glossary and ADR updates are being considered.
---

# Domain Modeling

This skill governs the language used to describe the product and codebase. It does not authorize expanding product scope or externally governed schemas.

Read existing terms and decisions first. Test ambiguous terms against normal, boundary, and failure cases; surface contradictions with code.

When durable domain documents are needed, read [references/domain-docs.md](references/domain-docs.md). Create them lazily only after a term or decision has actually settled.

Separate:

- domain terms: what the project means;
- implementation names: how the current code realizes it;
- external standards and ontologies: definitions the project reuses but does not own;
- requirements: behavior to build;
- architectural decisions: costly choices and their rationale.

Propose one canonical term per concept and aliases to retire. Update a glossary only after meaning settles; use an ADR only for consequential, non-obvious, hard-to-reverse choices.

Do not silently edit terminology, requirements, schema boundaries, or ADRs. Apply changes only within the enclosing task's authorization. Use `git` for related GitHub discussion or repository history.
