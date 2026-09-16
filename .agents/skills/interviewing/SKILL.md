---
name: interviewing
description: Drive a decision-focused interview that exposes assumptions and resolves dependent choices. Use beneath brainstorming, design, triage, and planning workflows when user judgment is required.
---

# Interviewing

Model the discussion as a dependency graph of decisions. Ask only questions whose prerequisites are already settled; defer downstream questions until their inputs are known.

Work in short rounds. Ask the entire current frontier in each round, but never include a question that depends on another answer in that round. For each question:

- explain the decision and why it matters;
- offer concrete, mutually exclusive options when possible;
- state a recommended answer and its trade-off;
- distinguish a user choice from a fact the agent can investigate.

Look up available facts rather than delegating research to the user. If evidence is missing, mark the dependent branch as blocked and continue with other answerable questions.

After each round, restate what changed and recompute the frontier. Do not implement, publish, or edit artifacts as part of the interview unless the enclosing workflow separately authorizes it.

Finish when every material branch is settled, deliberately deferred, or assigned an evidence-gathering next step, and the user confirms the shared understanding.
