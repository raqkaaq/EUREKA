---
name: debug
description: Diagnose hard bugs, flaky failures, and performance regressions through reproducible evidence. Use when the user asks for diagnosis or when the cause is not already demonstrated.
---

# Debug

Diagnosis does not imply authorization to implement a fix.

## Establish the Signal

Build and run the tightest harness that can fail for the reported symptom. Do not form a favored diagnosis before confirming this red signal. For intermittent failures, measure reproduction rate.

If no red-capable loop can be built, stop diagnosis, list what was tried, and request the missing access, artifact, or permission. Do not theorize past this gate.

## Narrow the Cause

1. Minimize the input and environment while preserving the failure.
2. Generate 3–5 falsifiable hypotheses.
3. Rank them by evidence and cost of discrimination.
4. Change or instrument one variable at a time.
5. Record observations that eliminate alternatives, not just evidence for the leading theory.

Show the ranked hypotheses to the user before testing them; continue without waiting if the user is unavailable.

For performance problems, establish a repeatable baseline and profile or bisect before optimizing. Redact secrets and sensitive payloads from captured artifacts.

Report reproduction, minimal case, causal evidence, scope, and remedy. If authorized to fix, add a regression test, make the smallest adequate change, rerun the original reproduction and broader checks, and remove tagged instrumentation.
