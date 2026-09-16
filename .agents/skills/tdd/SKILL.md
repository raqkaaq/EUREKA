---
name: tdd
description: Implement behavior test-first in small vertical slices. Use when building a feature or regression fix through an observable seam with a red-green-refactor loop.
---

# Test-Driven Development

Agree on observable behavior and its stable public seam. Derive expected results independently of the implementation.

Read [references/testing.md](references/testing.md) when choosing test level, substitutes, or mocks.

For each vertical slice:

1. Add one focused test that fails for the intended reason and run it.
2. Add the smallest production change that satisfies that behavior.
3. Run the focused and relevant neighboring checks.
4. Repeat for the next behavior.

Do not assert private calls, copy the implementation algorithm, or mock through the behavior. Use the repository's configured environment and task runner. Include applicable requirements and edge cases, run the broader suite before completion, and report skipped checks. Keep this loop red-green; reserve broad restructuring for review.
