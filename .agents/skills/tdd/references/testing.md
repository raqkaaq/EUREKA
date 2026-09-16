# Test and Substitute Guidance

## Durable Tests

Test one observable claim through the caller-facing interface. Derive expected values independently. Avoid private-call assertions, side-channel observations, duplicated algorithms, and horizontal batches written before the first slice teaches anything.

## Boundaries

Use real in-process collaborators. Substitute true boundaries—remote APIs, time, randomness, filesystem, or database—only when a lightweight real instance is impractical. Inject varying dependencies through narrow, typed seams; avoid generic request mocks.
