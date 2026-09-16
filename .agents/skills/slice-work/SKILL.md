---
name: slice-work
description: Decompose an approved specification into small, dependency-aware implementation slices. Use when work is understood but too large for one implementation cycle.
---

# Slice Work

Read the complete specification and current repository state. Split by observable capability, not by technical layer.

First identify any small preparatory refactor that makes the change easier without changing behavior. Make it an earlier slice only when independently safe and verifiable.

Each slice must:

- deliver a narrow end-to-end behavior or independently verifiable foundation;
- fit a fresh implementation context;
- state its acceptance criteria and testing seam;
- name prerequisites and consumers;
- include required research or safety gates;
- exclude work owned by another slice;
- leave the repository in a coherent state.

Prefer the smallest dependency graph that preserves correctness. Do not create artificial chains when slices can proceed independently. For unavoidable wide migrations, use an expand, migrate, contract sequence that keeps intermediate states valid.

Present the proposed slices and dependency edges. Ask the user to approve granularity, genuine blockers, and any merges or splits. Use `git` to search for duplicates and publish the approved child issues without rewriting or closing the parent.
