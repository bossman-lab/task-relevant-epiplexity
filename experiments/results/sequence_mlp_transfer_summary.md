# Sequence MLP Transfer Experiment

A one-hidden-layer MLP is trained for next-token prediction on synthetic
binary sequences. The frozen hidden representation is then used for few-shot
classification of the target recurrence family.

Setup:

- sequence length: `26`
- context length: `6`
- hidden units: `10`
- source train sequences: `2200`
- target few-shot labels: `8`
- repeats: `18`

| source | initial val loss | final val loss | learning progress proxy | loss reduction | transfer accuracy |
|---|---:|---:|---:|---:|---:|
| random | 0.6984 +/- 0.0023 | 0.6933 +/- 0.0001 | 0.0115 +/- 0.0060 | 0.0051 +/- 0.0024 | 0.5277 +/- 0.0876 |
| irrelevant_rule | 0.6982 +/- 0.0048 | 0.5855 +/- 0.1161 | 1.5835 +/- 1.5587 | 0.1126 +/- 0.1161 | 0.5380 +/- 0.1105 |
| relevant_rule | 0.4871 +/- 0.0615 | 0.2599 +/- 0.0109 | 0.4760 +/- 0.1213 | 0.2272 +/- 0.0605 | 0.5685 +/- 0.0865 |
| mixed_rule | 0.6990 +/- 0.0031 | 0.6399 +/- 0.0646 | 0.7811 +/- 0.8659 | 0.0591 +/- 0.0644 | 0.5859 +/- 0.1044 |

Reading:

- `irrelevant_rule` is learnable for next-token prediction but does not match
  the target recurrence label.
- `relevant_rule` is aligned with the downstream target and should transfer
  better when the learned hidden representation captures the recurrence.
- `mixed_rule` contains both structures and tests whether a small model can
  preserve task-relevant features while also fitting irrelevant structure.
