# Select datasets and plan data splits

- Select only sources listed for the exact class in `allowed_sources_by_class`; GraphRAG characteristics are evidence and never grant eligibility.
- Give each selected source a purpose: primary target-domain training, coverage supplement, or generalization supplement. Do not mix sources merely to increase dataset count.
- Prefer real target-domain data as the base distribution for general deployment, but allow derived-only or synthetic-heavy selections when the user domain, availability, or evidence justifies them. State the transfer risk; do not treat a preferred domain share as an executable constraint.
- Compare domain alignment, available counts, native task, annotation semantics, resolution, lineage, and source compatibility when supplied. Report missing facts.
- Classification requires valid image-level labels. Detection requires compatible bounding-box annotations.
- Treat derived, translated, synthetic, duplicate, video, scene, subject, site, and capture-session relationships as leakage risks when metadata supports them.
- Choose counts and a defensible split strategy based on coverage and independent groups. Local deterministic code owns exact assignments, count conservation, and leakage validation.
- Report insufficient class coverage rather than inventing images or exceeding availability.
