---
name: design-session
description: Explore a design against the repository, its terminology, and its architecture. Use when a brainstorm should be grounded in current code and may lead to proposed requirements, glossary, or ADR changes.
---

# Design Session

Read the relevant code, requirements, glossary or context documents, and architectural decisions before interviewing. Treat draft design material as revisable unless the repository or user establishes otherwise.

Invoke `interviewing` for the decision process. Invoke `domain-modeling` when terminology or domain relationships change, and `module-design` when module shape or test seams are at stake.

Continuously compare proposed behavior with the repository. Surface contradictions instead of quietly choosing one source. Distinguish:

- settled decisions;
- hypotheses requiring `research`;
- questions requiring a `prototype`;
- glossary changes;
- durable architectural decisions;
- requirements changes;
- implementation details that should wait.

Propose document edits as decisions settle, but apply them only when requested. Use `git` for every GitHub or Git operation. End with a concise decision record and the next appropriate workflow.
