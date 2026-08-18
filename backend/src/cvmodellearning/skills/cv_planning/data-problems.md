# Reason about common CV data problems

- Small dataset: judge effective independent diversity and domain similarity, not image count alone. Consider conservative transfer learning, augmentation, regularization, and validation uncertainty.
- Imbalance: distinguish image coverage from detection instance imbalance. Consider sampling, loss weighting, targeted data collection, and per-class metrics; oversampling does not create diversity.
- Small objects: use object-size evidence when available. Consider resolution, tiling, multi-scale features, and object-preserving augmentation. Never infer AP-small from overall mAP.
- Domain shift: distinguish visual shift from label-definition or class-prior shift. Stronger augmentation cannot fix incompatible semantics.
- Multiple sources: check annotation compatibility, source/class confounding, lineage, duplicates, and group-aware splitting before mixing.
- Limited compute: trade batch size, gradient accumulation, precision, resolution, and model capacity together. Preserve task-critical resolution where possible.
- When a problem is only user-reported, treat it as a requirement or hypothesis rather than a measured dataset property.
