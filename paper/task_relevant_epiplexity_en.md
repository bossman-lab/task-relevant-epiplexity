# Task-Relevant Epiplexity: Separating Learnable Structure from Useful Structure

Author: bossman-lab <bosis826@tutamail.com>

## Abstract

Epiplexity formalizes the structural information that a computationally bounded observer can extract from data. It offers a useful lens for understanding why likelihood training, synthetic data, data ordering, and emergent systems can produce reusable internal programs rather than mere memorization. However, task-agnostic structural information is not sufficient to explain transfer: a dataset may induce large amounts of structure while being irrelevant to a target task. We extend epiplexity in three ways. First, we define task-relevant epiplexity (TRE), the reduction in target task description length attributable to source-trained structure beyond matched controls. Second, we make the observer explicit through an observer specification covering architecture, optimizer, representation, ordering, context interface, inference protocol, and compute budget. Third, we introduce contextual epiplexity, the temporary structural information supplied by prompts, demonstrations, retrieval, and tool outputs at inference time. We provide two controlled experiments. A transparent PCA observer shows that total structure can be high while transfer remains at chance when the learned structure is target-irrelevant. A small next-token MLP experiment shows that learning-progress proxies can also decouple from transfer. These results do not claim to settle large-model behavior, but they motivate a stricter distinction between learnable structure and useful, transferable structure.

## 1. Introduction

Large models are often described as systems that learn from scale: more data, more parameters, and more compute. This description is true but incomplete. In practice, not all data of equal size has equal value. Data order, formatting, curriculum, synthetic generation, retrieval, and even a few in-context examples can change what a model learns or exhibits.

A natural explanation is that models do not merely store data. They build internal programs that compress and predict the data under finite compute. These programs may include syntax, algorithms, latent variables, world-model fragments, causal regularities, or reusable circuits. The recent notion of epiplexity captures this intuition by measuring structural information available to a computationally bounded observer.

Yet a gap remains. Structural information is not the same as useful structural information. A music corpus, a cellular automaton, or a formal grammar can contain substantial learnable structure, but that structure may not help with legal reasoning, chess evaluation, or code execution. This creates a failure mode for strong proxy-based explanations of out-of-distribution performance: high structural complexity alone cannot guarantee transfer.

This paper proposes a more task-aware view. We argue that a theory of data value should answer three questions:

1. What structure can an observer extract from the data?
2. Which parts of that structure are relevant to a target task?
3. Which parts can be made available at inference time through context, tools, or additional computation?

Our contributions are:

- We define an observer specification so epiplexity-like quantities are explicitly tied to model class, representation, ordering, inference protocol, and compute.
- We define task-relevant epiplexity as a target-conditioned reduction in description length beyond matched controls.
- We define contextual epiplexity as the inference-time analogue of structural information supplied by context.
- We provide two controlled experiments showing that total structure and learning progress can decouple from transfer.

## 2. Background

### 2.1 Entropy, Compression, and Structure

Shannon entropy measures average code length under a distribution. It captures unpredictability, but not whether that unpredictability contains learnable structure. A random string can have high entropy and little reusable structure. A short algorithmic text can have lower entropy yet induce reusable mechanisms in a learner.

Kolmogorov complexity measures the length of the shortest program that generates an object. It connects compression and programs, but its classical form assumes unbounded computation. Under unbounded compute, many complex phenomena have short descriptions: chess rules, cellular automata, and physical dynamics may be compactly specified even though their useful predictive abstractions are difficult to obtain.

The minimum description length (MDL) principle combines model cost and data cost. A useful model is worth encoding when it reduces the residual cost of the data. Epiplexity can be understood as a time-bounded MDL decomposition: the model part captures structural information, and the residual code length captures time-bounded randomness.

### 2.2 Epiplexity

Let an observer be restricted to probabilistic programs that run within a time budget `T`. For a random variable `X`, the time-bounded MDL objective selects a program `P*` minimizing:

```text
|P| + E[-log P(X)].
```

The epiplexity `S_T(X)` is the description length `|P*|`; the time-bounded entropy `H_T(X)` is the residual expected code length `E[-log P*(X)]`. Intuitively, epiplexity measures the structural information that must be stored in the model to explain the data under the compute budget.

This perspective helps explain why deterministic computation can create information for bounded observers, why data order matters, and why likelihood training may induce predictive structures that were not explicit in the generating process.

### 2.3 The Missing Distinction

Epiplexity is a measure of extractable structure. Transfer requires more: the extracted structure must be relevant to the target. A strong form of the claim "higher epiplexity implies better OOD performance" is therefore false in general. The same issue applies to many proxies: held-out loss, compression ratio, learning progress, or representation complexity can fail when the learned structure is unrelated to the downstream task.

## 3. Observer Specification

We define an observer specification:

```text
Omega = (A, M, tau, O, C, I, T)
```

where:

- `A` is the training algorithm and optimizer.
- `M` is the model family.
- `tau` is the representation or tokenization.
- `O` is the data ordering and conditioning protocol.
- `C` is the context interface, including retrieval, demonstrations, and tools.
- `I` is the inference protocol, such as chain-of-thought, recurrence, or tool use.
- `T` is the training and inference compute budget.

We then write:

```text
S_Omega(X)
```

for the structural information extractable from data `X` by observer `Omega`. This notation is deliberately less elegant than `S_T(X)`, but it is more faithful to engineering reality. The same data can expose different structure to different architectures, tokenizers, orderings, and context protocols.

## 4. Task-Relevant Epiplexity

Let `X` be source data and `Z = (U, V)` be a target task with inputs `U` and outputs `V`. Let `theta_X` be the state, weights, or encoded program produced by training observer `Omega` on `X`.

Define the target transfer gain:

```text
Delta_Omega(X -> Z) = L_Omega(Z) - L_Omega(Z | theta_X),
```

where `L_Omega(Z)` is the description length of the target task without the source-trained state, and `L_Omega(Z | theta_X)` is the description length when the source-trained state is available. This conditional code must include any adapter, probe, prompt, or fine-tuning cost required to use `theta_X`.

To avoid confusing leakage, surface statistics, or generic complexity with target relevance, we introduce matched controls. Let `C(X)` be a set of control sources that preserve non-target properties of `X` such as length distribution, token frequencies, local statistics, or total structure, while disrupting the target-relevant relationship.

We define task-relevant epiplexity:

```text
TRE_Omega(X; Z)
  = Delta_Omega(X -> Z) - E_{X' in C(X)} Delta_Omega(X' -> Z).
```

This quantity separates two claims:

```text
S_Omega(X) is high: the observer learned much structure.
TRE_Omega(X; Z) is high: the observer learned structure useful for Z.
```

### Proposition 1: High Structure Does Not Imply High TRE

There exist source data distributions `X1`, `X2` and a target task `Z` such that:

```text
S_Omega(X1) > S_Omega(X2)
TRE_Omega(X1; Z) < TRE_Omega(X2; Z).
```

Sketch. Let the observer learn a high-dimensional structured subspace unrelated to `Z` from `X1`, and a lower-dimensional subspace aligned with `Z` from `X2`. The first source has higher extractable structure, but only the second reduces the target code length. The PCA experiment in Section 7 instantiates this construction.

## 5. Contextual Epiplexity

Inference-time context can change what a model is able to do without changing its weights. Few-shot examples, retrieved evidence, intermediate reasoning, and tool outputs can expose latent variables, specify a data format, or reduce a search space.

For query `Q`, answer `Y`, and context `K`, define:

```text
CE_Omega(K; Y | Q)
  = L_Omega(Y | Q) - L_Omega(Y | Q, K).
```

This contextual epiplexity measures how much the context reduces the description length of the answer. A controlled version subtracts matched contexts:

```text
CRE_Omega(K; Y | Q)
  = CE_Omega(K; Y | Q) - E_{K' in C(K)} CE_Omega(K'; Y | Q).
```

Examples of control contexts include label-shuffled demonstrations, same-topic but irrelevant retrieval, format-preserving explanations with broken causal steps, or tool outputs with corrupted values.

## 6. Predictions

The framework makes several falsifiable predictions:

- Total structure can be high while transfer is low.
- Task-relevant structure should predict transfer better than total structure when controls are matched.
- Data ordering affects transfer through the structures it makes accessible to the observer.
- In-context examples improve performance when they supply task-relevant temporary structure, not merely when they add format or length.
- Learning-progress proxies can fail when progress is driven by target-irrelevant regularities.

If task-relevant epiplexity does not outperform loss, loss slope, total structure, or representation probes under controlled interventions, then the framework should be weakened.

## 7. Experiment 1: PCA Observer

### 7.1 Setup

We construct a transparent observer: PCA followed by a few-shot ridge classifier. Inputs have dimension 32. The downstream target depends only on the last four coordinates. The source data varies in whether its dominant structure is target-relevant:

- `random`: isotropic Gaussian data.
- `irrelevant_high`: strong structure in the first eight coordinates.
- `relevant_mid`: weaker structure in the target coordinates.
- `mixed_budget_limited`: both irrelevant and relevant structure.
- `relevant_high`: strong target-relevant structure.

The PCA observer learns `k` components from source data. We then freeze the PCA representation and train a ridge classifier on 24 labeled target examples.

We report:

- `total_structure_proxy`: variance above isotropic background captured by PCA.
- `target_alignment`: overlap between the PCA subspace and the target direction.
- `TRE proxy`: product of total structure and target alignment.
- `transfer_accuracy`: few-shot test accuracy.

### 7.2 Results

At `k = 4`, `irrelevant_high` has total structure `22.0220 +/- 0.2044`, far higher than `relevant_mid` at `7.0369 +/- 0.1126`. Yet `irrelevant_high` transfer is `0.5006 +/- 0.0079`, while `relevant_mid` transfer is `0.8100 +/- 0.0401`.

Across all runs, the Pearson correlation between total structure and transfer is `-0.0252`. The correlation between target alignment and transfer is `0.9086`. The correlation between the TRE proxy and transfer is `0.6004`, with Spearman correlation `0.7230`.

The mixed source illustrates observer dependence. With `k = 4`, the observer spends capacity on irrelevant high-variance structure and transfers at chance. With `k = 12`, the target-relevant subspace becomes visible, target alignment rises to `0.9975 +/- 0.0010`, and transfer accuracy rises to `0.7212 +/- 0.0514`.

Figures:

- `experiments/figures/pca_metric_scatter.svg`
- `experiments/figures/pca_budget_effect.svg`
- `experiments/figures/framework_diagram.svg`

## 8. Experiment 2: Next-Token MLP

### 8.1 Setup

The PCA experiment is transparent but not language-model-like. We therefore implement a second controlled experiment using a one-hidden-layer MLP trained for next-token prediction on synthetic binary sequences. After training, the hidden representation is frozen and used by a few-shot ridge classifier for a target recurrence task.

Source distributions:

- `random`: random binary sequences.
- `irrelevant_rule`: a learnable recurrence unrelated to the target label.
- `relevant_rule`: a recurrence aligned with the target label.
- `mixed_rule`: a mixture of target-relevant and irrelevant recurrences.

The learning-progress proxy is the area of the validation loss curve above the final loss.

### 8.2 Results

The `irrelevant_rule` source has learning progress `1.5835 +/- 1.5587`, higher than `relevant_rule` at `0.4760 +/- 0.1213`, but its transfer accuracy is lower: `0.5380 +/- 0.1105` versus `0.5685 +/- 0.0865`. Across all runs, learning progress has Pearson correlation `-0.1084` with transfer, and loss reduction has Pearson correlation `-0.0162`.

This result is weaker and noisier than the PCA experiment, but it adds a next-token training dynamic: simply being learnable by an autoregressive model does not imply usefulness for the target task.

Figure:

- `experiments/figures/sequence_mlp_bars.svg`

## 9. Related Work

This work builds on epiplexity and computationally bounded information measures, MDL and prequential coding, compression views of language modeling, and work on in-context learning and induction heads. It is also related to representation learning, transfer learning, data selection, curriculum learning, information bottleneck methods, and recent counterexamples to strong proxy-based explanations of OOD performance.

The key distinction is that we do not propose a new task-agnostic universal data value. Instead, we separate task-agnostic structure from target-relevant structure and make the observer explicit.

## 10. Limitations

The experiments are controlled and small. They do not establish large-model scaling behavior. The PCA observer is intentionally simple; the MLP observer is closer to language modeling but still far from modern transformers.

TRE is target-dependent. This is a limitation if one wants a single universal data value, but it is necessary for explaining transfer to a specified target family.

The definition depends on controls. Poorly chosen controls can hide or exaggerate target relevance. A practical theory of TRE requires careful matched-control design.

The conditional code `L_Omega(Z | theta_X)` must account for adapters, probes, prompts, or fine-tuning. If these costs are ignored, TRE may collapse into downstream performance.

Finally, contextual epiplexity remains conceptual in this draft. It requires dedicated experiments with label-shuffled demonstrations, corrupted retrieval, and tool-output controls.

## 11. Conclusion

Epiplexity reframes learning as the extraction of structural information by a computationally bounded observer. We argue that a theory of transfer must go one step further: it must separate structure from useful structure.

Task-relevant epiplexity measures whether source-trained structure reduces the description length of a target task beyond matched controls. Observer specification makes the quantity reproducible and architecture-aware. Contextual epiplexity extends the same idea to inference-time information supplied by prompts, retrieval, demonstrations, and tools.

The resulting picture is:

```text
Observed intelligence
  ~= stored structure in parameters
   + task-relevant reusable structure
   + temporary structure in context
   + inference-time compute and tools.
```

This is not a complete theory of intelligence. It is a proposal for making one important distinction measurable: the distinction between data that a model can learn from and data that helps it solve the tasks we care about.

## References

Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., and Wilson, A. G. (2026). From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence. arXiv:2601.03220.

Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165.

Olsson, C., et al. (2022). In-context Learning and Induction Heads. arXiv:2209.11895.

Delétang, G., et al. (2023). Language Modeling Is Compression. arXiv:2309.10668.

Rissanen, J. (2004). MDL Denoising. IEEE Transactions on Information Theory.

Grunwald, P. (2007). The Minimum Description Length Principle. MIT Press.

Zhao, Z., Wallace, E., Feng, S., Klein, D., and Singh, S. (2021). Calibrate Before Use: Improving Few-Shot Performance of Language Models. ICML.

Li, H. (2026). A Controlled Counterexample to Strong Proxy-Based Explanations of OOD Performance: in a Fixed Pretraining-and-Probing Setup. arXiv:2605.11554.
