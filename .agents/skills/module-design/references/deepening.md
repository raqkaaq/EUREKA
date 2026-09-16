# Deepening Modules

Choose seams by dependency:

- **In-process:** consolidate and test directly through the new external interface.
- **Locally substitutable:** use a realistic local implementation inside tests; keep the seam internal.
- **Remote but owned:** define a port at the deployment seam with production and in-memory adapters.
- **Truly external:** inject a narrow port and test with a controlled adapter.

A seam must serve real variation. Replace implementation-coupled tests with behavior tests at the new interface. For consequential design, compare alternatives that optimize surface area, extensibility, and the dominant caller; judge depth, locality, dependencies, errors, and caller knowledge.
