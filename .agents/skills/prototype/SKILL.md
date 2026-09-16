---
name: prototype
description: Build a disposable experiment that answers one named design question. Use when executable evidence can resolve uncertainty faster or more reliably than discussion.
---

# Prototype

State the question, possible outcomes, and the observation that will distinguish them before writing code. The prototype exists to answer that question, not to become an early production implementation.

Choose the smallest runnable form that exposes the relevant behavior. Keep setup trivial, state visible, persistence disposable, and dependencies minimal. Skip production hardening, generalized abstractions, and unrelated polish unless they are part of the question being tested.

Choose the prototype shape from the question:

- For business logic, state transitions, or data shape, read [references/logic.md](references/logic.md).
- For visual hierarchy or interaction alternatives, read [references/ui.md](references/ui.md).

Mark prototype code unmistakably. Do not merge it into production by gradual cleanup. After running the experiment, record:

- the question and setup;
- observations and limitations;
- the resulting decision or remaining uncertainty;
- which parts, if any, are evidence rather than reusable code.

After the question is answered, keep production branches limited to the validated decision. Use `git`, when authorized, to preserve the prototype as a primary source on a throwaway branch and link it from the governing issue. Do not silently delete it or promote prototype code directly into production.
