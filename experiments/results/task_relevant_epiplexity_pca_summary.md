# Synthetic PCA Transfer Experiment

This is a controlled toy experiment for the paper draft. It uses a PCA observer,
so it should be read as a proof-of-concept for the definitions rather than as
evidence about frontier language models.

Setup:

- input dimension: `32`
- source samples per run: `6000`
- target few-shot labels: `24`
- target test samples: `4000`
- repeats: `64`
- target task depends only on the last four coordinates
- `irrelevant_high` has strong source structure in the first eight coordinates
- `relevant_mid` has weaker source structure aligned with the target coordinates
- `mixed_budget_limited` contains both, but a small representation budget spends
  its top components on the irrelevant high-variance structure

| k | source | total structure proxy | target alignment | TRE proxy | transfer accuracy |
|---:|---|---:|---:|---:|---:|
| 4 | random | 0.4755 +/- 0.0218 | 0.1258 +/- 0.0842 | 0.0602 +/- 0.0408 | 0.5444 +/- 0.0484 |
| 4 | irrelevant_high | 22.0220 +/- 0.2044 | 0.0001 +/- 0.0001 | 0.0024 +/- 0.0016 | 0.5006 +/- 0.0079 |
| 4 | relevant_mid | 7.0369 +/- 0.1126 | 0.9967 +/- 0.0009 | 7.0137 +/- 0.1127 | 0.8100 +/- 0.0401 |
| 4 | mixed_budget_limited | 21.0801 +/- 0.2218 | 0.0006 +/- 0.0004 | 0.0120 +/- 0.0084 | 0.5022 +/- 0.0085 |
| 4 | relevant_high | 24.5222 +/- 0.2280 | 0.9993 +/- 0.0002 | 24.5048 +/- 0.2283 | 0.8120 +/- 0.0390 |
| 12 | random | 0.9523 +/- 0.0306 | 0.3719 +/- 0.0968 | 0.3544 +/- 0.0942 | 0.5792 +/- 0.0472 |
| 12 | irrelevant_high | 42.0432 +/- 0.3305 | 0.1444 +/- 0.0863 | 6.0739 +/- 3.6316 | 0.5220 +/- 0.0338 |
| 12 | relevant_mid | 7.0194 +/- 0.1096 | 0.9977 +/- 0.0006 | 7.0034 +/- 0.1092 | 0.7200 +/- 0.0591 |
| 12 | mixed_budget_limited | 40.1883 +/- 0.2833 | 0.9975 +/- 0.0010 | 40.0895 +/- 0.2840 | 0.7212 +/- 0.0514 |
| 12 | relevant_high | 24.4804 +/- 0.2119 | 0.9995 +/- 0.0002 | 24.4675 +/- 0.2111 | 0.7173 +/- 0.0581 |

Reading:

- `irrelevant_high` has larger total structure than `relevant_mid`, but near-zero
  target alignment and chance-level transfer.
- `relevant_mid` has less total structure, but high task alignment and strong
  few-shot transfer.
- `mixed_budget_limited` shows observer dependence: with `k=4`, the observer
  misses the target-relevant structure; with `k=12`, the same source becomes
  highly transferable because the representation budget can include both
  irrelevant and relevant structure.
