# Ready Work

An issue is ready for implementation only when a fresh agent can act without inventing product or design decisions. It must identify:

- one observable outcome and the current behavior or state;
- desired normal, edge, and failure behavior;
- scope boundaries, exclusions, and relevant durable decisions;
- concrete acceptance criteria and a verification approach;
- dependencies, blockers, and required evidence.

Unknown implementation details do not block readiness when the required behavior and interface constraints are settled. Unknown behavior, scope, dependencies, or verification keeps the issue out of the ready state.

## Examples

Ready: "Retry failed object-store uploads with exponential backoff (3 attempts, 1s/2s/4s) on 5xx only; 4xx surfaces immediately. Acceptance: unit test covers 503-then-success and 503x3-then-error; existing `uploader` seam unchanged." Outcome, edge behavior, scope, criteria, and seam are all settled — the implementer chooses only retry-loop internals.

Not ready: "Make uploads more reliable." No observable outcome, no scope boundary, no acceptance check — the implementer must invent what reliability means.
