# Select a CV model from retrieved candidates

1. Consider only executable models exposed by the local registry and, when GraphRAG is enabled, compare the retrieved candidate models.
2. Eliminate candidates only for explicit hard constraints. Treat qualitative accuracy and latency requirements as preferences.
3. Compare at least two feasible candidates when possible. Consider task fit, pretrained weights, capacity versus available data, deployment memory, model size, and comparable benchmark evidence.
4. Do not compare metrics as equivalent when dataset, input size, protocol, or hardware differs. Overall detection mAP is not evidence of small-object performance.
5. Distinguish measured latency from estimates and unknown target-device performance.
6. Select the best contextual trade-off, not the graph ranking automatically. Mention a fallback and conditions that would change the choice in the rationale.
