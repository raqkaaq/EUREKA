# Durable Agent Brief

The ready brief is the contract a fresh agent implements. Earlier issue discussion remains context.

Include:

- category and a one-line outcome;
- current behavior or current state of an attached pull-request diff;
- desired observable behavior, including edge and error cases;
- stable interfaces, domain constraints, and relevant decisions;
- independently verifiable acceptance criteria;
- explicit exclusions;
- blockers and required evidence.

Write behavior, not a sequence of edits. Prefer stable type and interface names over file paths or line numbers. Each criterion must admit a concrete pass/fail check. For a pull request, describe what remains to make the existing diff complete instead of pretending implementation starts from zero.

If essential behavior, scope, or verification is still unknown, the item is not ready.
