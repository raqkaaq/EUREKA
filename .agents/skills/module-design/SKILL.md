---
name: module-design
description: Design or improve module interfaces, seams, and dependency placement. Use when deciding where behavior belongs, how callers should access it, or how to make behavior testable without leaking internals.
---

# Module Design

A **module** hides meaningful complexity behind a small **interface**. A **seam** is where callers exercise or substitute it. **Depth** is behavior gained per interface learned; **leverage** is that benefit across callers; **locality** concentrates change and knowledge behind the interface. An **adapter** satisfies an interface at a seam.

Use these terms consistently: module, interface, implementation, seam, adapter, depth, leverage, and locality. An interface includes invariants, errors, ordering, configuration, and performance expectations—not only a type signature.

Evaluate what belongs together, what callers must know, whether complexity is hidden or scattered, whether variation is real, and whether tests use the caller-facing seam.

When consolidating existing modules or designing dependency seams, read [references/deepening.md](references/deepening.md).

Follow the repository's existing dependency direction and package-placement rules unless the design explicitly changes them.

For consequential interfaces, compare at least three radically different shapes: minimal surface, extensibility, and dominant-caller convenience. Report the chosen interface, hidden responsibilities, dependencies, errors, test seam, and rejected alternatives. Do not implement unless requested.
