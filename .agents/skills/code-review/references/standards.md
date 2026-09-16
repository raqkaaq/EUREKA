# Engineering Review Baseline

Repository rules override this heuristic baseline. Tool-enforced formatting and lint failures belong to tool output, not duplicated review commentary.

Look for:

- names that obscure responsibility;
- duplicated policy or algorithms;
- behavior placed closer to another module's data than its own;
- groups of primitives that repeatedly travel together and represent a concept;
- repeated branching on the same kind that belongs in one dispatch point;
- one change scattered across many unrelated locations;
- one module changing for unrelated reasons;
- abstractions, extension points, or parameters without a current requirement;
- callers navigating deep object chains that should be hidden;
- pass-through modules that add vocabulary without hiding complexity;
- inheritance whose implementations reject the parent contract.

These are signals, not automatic violations. Cite the hunk and consequence.
