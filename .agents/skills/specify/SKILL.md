---
name: specify
description: Synthesize an already-developed conversation or design into an implementation-ready specification. Use when decisions are settled enough to record without conducting another interview.
---

# Specify

Do not reopen the design by default. Synthesize the conversation, repository evidence, research, prototypes, and existing decisions. If a missing fact makes the specification unsafe or ambiguous, identify the gap and route back to the appropriate workflow rather than guessing.

Produce:

- problem and desired outcome;
- present behavior and relevant constraints;
- user-visible or externally observable behavior;
- accepted design and important rejected alternatives;
- affected modules and intended interfaces, without brittle implementation recipes;
- failure behavior and operational constraints;
- testing seams and acceptance criteria;
- dependencies, migration concerns, research citations, and non-goals;
- unresolved questions that prevent readiness.

Prefer stable domain language and behavior over file-level instructions. Prefer existing, high public testing seams and as few new seams as practical. Confirm the proposed seams with the user, then evaluate the result with the shared [ready-work criteria](../workflow/references/readiness.md).

Return a proposed specification. When the user wants it recorded in GitHub, invoke `git` to create or update the corresponding issue; do not perform GitHub operations directly.
