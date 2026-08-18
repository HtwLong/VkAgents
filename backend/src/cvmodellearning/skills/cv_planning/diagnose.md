# Diagnose the CV planning problem

- Separate explicit hard constraints from qualitative preferences.
- Use only facts present in the pipeline state or retrieved evidence. Never invent dataset statistics.
- Distinguish measured facts, estimates, user-reported concerns, derived conclusions, and unknowns.
- For classification, inspect class coverage, effective diversity, label semantics, domain alignment, grouping/leakage, and deployment constraints.
- For detection, additionally inspect instance coverage, object sizes, annotation compatibility, crowding, negatives, and scene/video grouping when those facts exist.
- Identify the dominant risks that matter to the current decision. If relevant evidence is missing, state the uncertainty instead of assuming a value.
- Keep the diagnosis concise and use it to compare alternatives; do not mechanically map one condition to one fixed answer.
