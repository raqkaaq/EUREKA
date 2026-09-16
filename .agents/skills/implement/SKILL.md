---
name: implement
description: Deliver bounded work from an approved specification or ready issue. Use when requirements and dependencies are settled and the user wants the change built through review and pull-request preparation.
---

# Implement

1. Invoke `git` to load the authoritative issue or specification, confirm readiness, inspect the working tree, and establish the intended branch state.
2. Identify the observable behavior and testing seams. Use `module-design` when the interface is unsettled.
3. Use `tdd` for each behavior that can be driven through a reliable seam. Follow repository instructions when orchestration and coding must be performed by different agents.
4. Run focused checks throughout and the appropriate broader checks before review.
5. Invoke `code-review` against both the originating work item and repository standards. Address in-scope findings or report why they remain.
6. Invoke `git` for commits, issue updates, and pull-request creation or updates when authorized.

Do not expand the issue silently. Record discovered follow-up work separately through `git` when authorized. Completion requires traceability from acceptance criteria to evidence, a clean account of skipped checks, and accurate GitHub state; producing code alone is not completion.
