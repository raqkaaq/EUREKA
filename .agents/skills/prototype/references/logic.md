# Logic Prototype

Build a directly runnable demonstration that lets a person drive the model through normal, edge, and invalid scenarios while observing the complete relevant state after every action.

Keep the decision-bearing logic separate from the demonstration shell. Prefer a pure reducer, explicit state machine, small pure function set, or narrow stateful module according to the question. The shell should speak domain language and provide both free exploration and repeatable guided scenarios.

Persistence is in-memory unless persistence is the question. Avoid frameworks and build systems when a self-contained artifact can answer the question. The validated model may inform production design, but production code still requires normal implementation and tests.
