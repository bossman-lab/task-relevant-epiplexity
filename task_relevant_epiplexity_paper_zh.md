# 从结构信息到可迁移结构：任务相关 Epiplexity 与计算受限智能的可检验框架

作者：bossman-lab <bosis826@tutamail.com>

## 摘要

大模型的能力通常被解释为大规模数据、大规模参数与最大似然训练共同作用的结果。然而，仅用训练损失、困惑度或传统信息量来描述数据价值，会混淆两类不同成分：一类是样本中不可预测的随机性，另一类是模型为了压缩和预测数据而必须内化的可复用结构。Finzi 等人提出的 epiplexity 将后者形式化为计算受限观察者可从数据中提取的结构信息，为理解合成数据、数据顺序、涌现结构和 OOD 泛化提供了有力框架。本文在此基础上提出三点扩展。第一，将 epiplexity 从任务无关结构信息扩展为任务相关可迁移结构信息，定义 task-relevant epiplexity，用以区分“学到很多结构”与“学到对目标任务有用的结构”。第二，将“计算受限观察者”工程化为由架构、优化器、表示方式、上下文协议和计算预算共同决定的 observer specification，使该理论可复现、可比较、可审计。第三，引入 contextual epiplexity，用于描述提示、检索、示例和工具输出如何在推理时临时改变模型可用的结构信息。本文同时给出一组可证伪实验，并实现一个最小受控 PCA 实验，展示总结构量与任务迁移能力可以分离。我们的核心主张是：大模型智能不应被理解为对数据的记忆，也不应只被理解为对分布的匹配；它更接近于计算受限条件下，为压缩和预测数据而构造可复用程序的过程。但可复用程序是否转化为泛化能力，取决于其与目标任务、表示通道和推理上下文之间的对齐关系。

## 1. 引言

现代大模型的经验事实迫使我们重新审视“数据价值”这一概念。同样规模的数据可能导致完全不同的迁移能力；同一批数据换一种顺序或格式，可能改变训练动态和下游表现；合成数据虽然由已有模型和确定性程序产生，却能提升能力；上下文中的少量示例不改变权重，却能显著改变模型表现。这些现象都提示：数据价值不能只用样本数量、Shannon entropy、训练损失或静态困惑度来刻画。

一种更贴近机器学习实践的观点是：模型学到的不是数据本身，而是为了高效压缩和预测数据，不得不形成的内部程序。这些程序可以是语法、算法、世界模型片段、因果结构、棋理、代码执行规则、归纳策略或工具使用模式。Finzi 等人提出的 epiplexity 正是在这个方向上的关键进展：它将可学习结构定义为计算受限观察者在最小描述长度意义下必须写入模型的那部分信息。

但是，原始 epiplexity 框架仍有三个开放问题。

第一，结构信息不等于有用结构信息。一个数据源可以诱导大量复杂结构，却与目标任务无关。例如音乐数据可能具有很高结构复杂度，但对法律问答未必有直接帮助。若我们直接用任务无关 epiplexity 解释 OOD 泛化，就容易得到过强结论。

第二，计算受限观察者的定义还不够工程化。真实模型不是抽象 Turing machine，而是由架构、tokenizer、优化器、训练课程、上下文长度、推理协议和工具接口共同决定的系统。同一数据对不同观察者呈现出的结构可能不同。

第三，训练时结构和推理时结构需要统一描述。In-context learning 显示，模型不只在权重中长期存储结构，也会在上下文中临时构造或调用结构。若 epiplexity 只描述训练数据写入权重的结构，就无法覆盖提示工程、检索增强、few-shot examples 和工具结果对智能表现的影响。

本文的目标是补齐这些缺口，使 epiplexity 从一种解释性理论进一步变成可检验的数据选择与能力分析工具。

## 2. 背景：从熵到结构信息

Shannon entropy 衡量随机变量的平均编码长度。它适合描述不可预测性，却不区分“随机复杂”与“结构复杂”。一段加密随机串可能 entropy 很高，但模型从中学不到可复用结构；一段算法代码 entropy 未必最高，却可能迫使模型形成可迁移的内部机制。

Kolmogorov complexity 衡量生成某个对象的最短程序长度。它把压缩与程序联系起来，但通常默认观察者拥有无限计算能力。在无限计算下，许多现实中复杂的现象可以被极短程序描述，例如元胞自动机、多体动力学或棋类规则。可是在有限计算下，直接运行底层规则可能不可行，模型反而必须学习更高层的涌现结构。

Epiplexity 的关键移动是：将信息量相对于计算受限观察者来定义。给定时间预算 `T`，一个模型既要短，又要能在预算内给数据分配高概率。最优二部编码中的模型程序长度对应结构信息，剩余负对数似然对应时间受限随机性。

直觉上：

```text
总时间受限信息 = 可学习结构 + 剩余不可预测性
MDL_T(X) = S_T(X) + H_T(X)
```

其中 `S_T(X)` 是 epiplexity，代表观察者为了压缩数据 `X` 必须内化的结构；`H_T(X)` 是 time-bounded entropy，代表在该结构下仍然需要逐样本编码的随机部分。

## 3. 原框架的边界

Epiplexity 的优势在于解释了一些传统信息论显得别扭的现象。

确定性计算可以对计算受限观察者“创造信息”。伪随机数、元胞自动机、自博弈和合成数据都可能把简单起点转化为观察者难以直接还原、但可部分学习的结构。

数据顺序会改变可学习信息。传统熵满足链式分解的对称性，但有限模型面对不同顺序时可能需要完全不同的计算。棋谱到最终局面容易，最终局面到可能棋谱困难；自然语言左到右通常比反向更符合生成与预测结构。

最大似然不只是分布匹配。为了预测一个 token，模型可能必须反推出隐藏变量或归纳潜在机制。侦探小说就是典型例子：作者可以先决定凶手再写线索，读者和模型却必须从线索反推凶手。

这些解释非常有力，但若要成为可操作理论，必须承认两个限制。

第一，高 epiplexity 不保证特定 OOD 任务表现好。它只说明模型学到结构，不说明结构与目标任务相关。

第二，epiplexity 不是绝对量。它依赖模型族、训练过程、数据表示和计算预算。不同观察者会从同一数据中看到不同结构。

因此，我们需要从“结构信息”走向“目标相关的可迁移结构信息”。

## 4. Observer Specification：让观察者可复现

我们定义一个工程化观察者：

```text
Omega = (A, M, tau, O, C, I, T)
```

其中：

- `A` 是训练算法与优化器，例如 AdamW、SGD、RL self-play、蒸馏或数据选择策略。
- `M` 是模型族，例如 decoder-only Transformer、CNN、state-space model 或 recurrent model。
- `tau` 是表示与 tokenization，包括文本 tokenizer、图像 VQ tokenization、棋局格式或代码序列化方式。
- `O` 是数据顺序与条件化协议，例如自回归顺序、输入输出拼接方式、反向建模方式。
- `C` 是上下文接口，包括上下文长度、few-shot 格式、检索片段、工具输出和可调用外部程序。
- `I` 是推理协议，包括是否允许 chain-of-thought、self-consistency、test-time compute、tool use 或 recurrence。
- `T` 是训练与推理计算预算。

于是 epiplexity 不再写作单一的 `S_T(X)`，而应写作：

```text
S_Omega(X)
```

它表示在观察者 `Omega` 下，数据 `X` 对该系统呈现出的可提取结构量。

这种写法牺牲了形式上的简洁，但换来了工程意义。一个理论若要指导大模型训练，必须能回答：“对哪类模型、哪种 tokenizer、哪种训练协议、多少计算预算成立？”

## 5. Task-Relevant Epiplexity：从结构量到有用结构量

设源数据为 `X`，目标任务为 `Z = (U, V)`，其中 `U` 是目标输入，`V` 是目标输出。令 `theta_X` 表示观察者 `Omega` 在源数据 `X` 上训练得到的模型状态或可编码程序。

我们关心的不是 `theta_X` 包含多少结构，而是它能在多大程度上减少目标任务的描述长度。定义源数据对目标任务的迁移收益：

```text
Delta_Omega(X -> Z)
  = L_Omega(Z) - L_Omega(Z | theta_X)
```

其中 `L_Omega(Z)` 是在没有源预训练状态时编码目标任务所需的最短描述长度，`L_Omega(Z | theta_X)` 是在允许使用源模型状态 `theta_X` 后编码目标任务所需的描述长度，且应计入必要 adapter、prompt、fine-tuning 或 probe 的编码成本。

但这个值可能被表面统计、标签泄漏或数据污染放大。因此我们引入 matched controls。令 `C(X)` 是一组控制数据，尽量保留源数据的非目标相关性质，例如 token 频率、长度分布、局部 n-gram、训练损失或总 epiplexity，同时破坏目标相关结构。

定义任务相关 epiplexity：

```text
TRE_Omega(X; Z)
  = Delta_Omega(X -> Z) - E_{X' in C(X)} Delta_Omega(X' -> Z)
```

直觉上，`TRE` 衡量的是：源数据中超出表面统计和一般结构复杂度之外，真正能被目标任务复用的结构。

这给出一个关键区分：

```text
S_Omega(X) 高：模型学到了很多结构。
TRE_Omega(X; Z) 高：模型学到了对目标任务 Z 有用的结构。
```

前者解释“模型内部程序有多复杂”，后者解释“这些程序是否迁移到我们关心的问题”。

## 6. Contextual Epiplexity：上下文中的临时结构

大模型的智能不仅来自权重，也来自上下文。Few-shot examples、检索材料、工具调用结果、任务说明和中间推理都能临时改变模型的有效计算问题。

设查询为 `Q`，答案为 `Y`，上下文为 `K`。定义上下文带来的描述长度减少：

```text
CE_Omega(K; Y | Q)
  = L_Omega(Y | Q) - L_Omega(Y | Q, K)
```

这可以理解为 contextual epiplexity：上下文 `K` 在当前查询 `Q` 下向模型暴露或构造了多少有用结构。

为了避免把答案泄漏误认为结构，应使用控制上下文：

```text
CRE_Omega(K; Y | Q)
  = CE_Omega(K; Y | Q) - E_{K' in C(K)} CE_Omega(K'; Y | Q)
```

其中 `C(K)` 可以是标签打乱的 examples、语义破坏但长度匹配的检索片段、同主题但无关的证据，或保留格式但破坏因果链的解释。

这个定义把 prompt engineering 放入同一框架：好的 prompt 不只是“命令模型”，而是在当前计算预算内为模型提供可压缩、可调用、可迁移的临时结构。

## 7. 可证伪预测

本文框架给出以下预测。

预测一：高任务无关 epiplexity 不必然带来高 OOD 泛化。若数据结构与目标任务无关，则 `S_Omega(X)` 可以高而 `TRE_Omega(X; Z)` 低。

预测二：当控制总 epiplexity 和训练损失后，`TRE_Omega(X; Z)` 应比 `S_Omega(X)` 更好预测目标任务迁移。

预测三：数据顺序对迁移的影响由任务相关结构中介。某些顺序会提高总 epiplexity，但只有当它诱导目标相关表示时才提高下游表现。

预测四：上下文的作用可以被描述为临时结构注入。标签打乱或语义破坏的 few-shot examples 会保留格式复杂度，却显著降低 contextual task-relevant epiplexity。

预测五：若一个数据选择方法有效，它未必降低训练损失；它可能选择 loss 下降更快、结构吸收更多的数据，从而提高 `TRE` 或 `CRE`。

这些预测都可能失败。若在严格控制下，`TRE` 不能比 perplexity、loss slope、representation probing 或 gradient diversity 更好解释迁移，则本文扩展框架应被削弱或修正。

## 8. 实验设计

### 8.1 高结构但低相关性的负例

构造三个源数据集：

1. `X_irrelevant_high`：高结构但与目标无关，例如复杂音乐语法、程序化迷宫或无关形式语言。
2. `X_relevant_mid`：中等结构但包含目标任务需要的因果或算法模式。
3. `X_random`：高 entropy 但低结构，例如加密随机串或打乱标签数据。

目标任务 `Z` 只依赖 `X_relevant_mid` 中的结构。理论预期：

```text
S(X_irrelevant_high) >= S(X_relevant_mid)
TRE(X_relevant_mid; Z) > TRE(X_irrelevant_high; Z)
Transfer(X_relevant_mid -> Z) > Transfer(X_irrelevant_high -> Z)
```

该实验用于证明总结构量不是迁移充分条件。

### 8.2 顺序干预

使用棋类、数学证明或程序执行轨迹。对同一数据构造多种序列化：

- 正向：步骤到结果。
- 反向：结果到步骤。
- 局部打乱：保留局部统计，破坏全局推理。
- 因果重排：突出目标任务所需隐变量。

测量训练损失、prequential epiplexity、relevant transfer gain 和目标任务表现。若某一顺序提高 epiplexity 但不提高目标任务，说明结构未对齐；若提高 `TRE`，则应带来更强迁移。

### 8.3 语义破坏控制

对文本数据构造 matched controls：

- 保留 unigram / bigram 统计但打乱句子。
- 保留文档长度和主题词但破坏事实关系。
- 保留推理链格式但替换关键因果步骤。
- 保留代码 token 分布但破坏变量绑定和执行语义。

若 epiplexity proxy 只反映表面复杂度，则这些控制数据可能仍显得高；若它捕捉可复用结构，则语义破坏应降低 `TRE` 和 contextual relevance。

### 8.4 Contextual epiplexity 实验

在 few-shot 任务上比较：

- 正确示例。
- 标签打乱示例。
- 输入输出格式正确但规则错误的示例。
- 检索到真实证据的上下文。
- 检索到同主题无关文本的上下文。

测量：

```text
CE = loss_without_context - loss_with_context
CRE = CE_correct - CE_control
```

若 CRE 与任务准确率、校准度、推理稳定性相关，则说明上下文确实以“临时结构”的形式影响模型智能。

### 8.5 与其他 proxy 的比较

必须将本文指标与以下 baseline 正面比较：

- held-out loss / perplexity
- compression ratio
- loss slope / learning progress
- representation probing
- intrinsic dimension
- gradient diversity
- data influence
- scaling-law parameters
- downstream validation mixture

如果 `TRE` 只是 loss slope 的重命名，它的理论价值有限。它必须在至少一类受控任务中解释 loss 或 perplexity 解释不了的迁移差异。

## 9. 初步受控实验：结构量与任务相关结构量分离

为了验证本文扩展的必要性，我们实现了一个最小的 synthetic proof-of-concept。该实验不声称模拟大模型训练，而是用一个透明的线性 PCA 观察者展示：总结构量可以与任务迁移能力分离，任务相关结构才更接近下游表现。

实验脚本见 `experiments/task_relevant_epiplexity_pca.py`，完整结果见 `experiments/results/task_relevant_epiplexity_pca_summary.md`。实验设置如下：输入维度为 32，目标任务只依赖最后 4 个坐标；源数据用于无监督 PCA 预训练，随后固定 PCA 子空间，只用 24 个目标标签训练一个 ridge 分类器。我们比较五类源数据：

- `random`：各向同性随机数据。
- `irrelevant_high`：前 8 个坐标有强结构，但与目标无关。
- `relevant_mid`：最后 4 个目标坐标有中等结构。
- `mixed_budget_limited`：同时包含强无关结构和中等相关结构。
- `relevant_high`：目标坐标有强结构，作为正控制。

我们报告四个量：

- `total structure proxy`：PCA 捕获的非各向同性结构量。
- `target alignment`：PCA 子空间与目标方向的重合度。
- `TRE proxy`：结构量与目标对齐度的乘积。
- `transfer accuracy`：固定表示后的 few-shot 目标测试准确率。

小表示预算 `k=4` 下，结果如下：

| 源数据 | 总结构 proxy | 目标对齐 | TRE proxy | 迁移准确率 |
|---|---:|---:|---:|---:|
| random | 0.4755 +/- 0.0218 | 0.1258 +/- 0.0842 | 0.0602 +/- 0.0408 | 0.5444 +/- 0.0484 |
| irrelevant_high | 22.0220 +/- 0.2044 | 0.0001 +/- 0.0001 | 0.0024 +/- 0.0016 | 0.5006 +/- 0.0079 |
| relevant_mid | 7.0369 +/- 0.1126 | 0.9967 +/- 0.0009 | 7.0137 +/- 0.1127 | 0.8100 +/- 0.0401 |
| mixed_budget_limited | 21.0801 +/- 0.2218 | 0.0006 +/- 0.0004 | 0.0120 +/- 0.0084 | 0.5022 +/- 0.0085 |
| relevant_high | 24.5222 +/- 0.2280 | 0.9993 +/- 0.0002 | 24.5048 +/- 0.2283 | 0.8120 +/- 0.0390 |

该结果给出一个清晰负例：`irrelevant_high` 的总结构量显著高于 `relevant_mid`，但因为结构与目标方向几乎无关，迁移准确率接近随机；`relevant_mid` 的总结构量更低，却因目标对齐度高而获得显著更好的迁移。这说明“高 epiplexity 数据更利于 OOD”这一强说法不够精确；更合理的判断对象是任务相关结构。

进一步的相关性分析见 `experiments/results/task_relevant_epiplexity_pca_analysis.md`。在所有 runs 上，`total_structure_proxy` 与迁移准确率的 Pearson 相关为 `-0.0252`，几乎没有解释力；`target_alignment` 的 Pearson 相关为 `0.9086`；`tre_proxy` 的 Pearson 相关为 `0.6004`、Spearman 相关为 `0.7230`。在 `k=4` 的小预算设置下，`total_structure_proxy` 与迁移准确率的 Pearson 相关为 `-0.0201`，而 `tre_proxy` 的 Pearson 相关为 `0.7904`、Spearman 相关为 `0.8378`。框架图见 `experiments/figures/framework_diagram.svg`；实验图表见 `experiments/figures/pca_metric_scatter.svg` 和 `experiments/figures/pca_budget_effect.svg`。

该实验也验证了 observer specification 的必要性。对 `mixed_budget_limited`，当 PCA 表示预算为 `k=4` 时，观察者把容量花在强无关结构上，迁移准确率为 `0.5022 +/- 0.0085`；当预算增大到 `k=12` 时，同一源数据的目标对齐度升至 `0.9975 +/- 0.0010`，迁移准确率升至 `0.7212 +/- 0.0514`。换言之，同一数据是否“有用”，取决于观察者能否在给定表示和计算预算下看见目标相关结构。

这个 toy 实验支持本文的两个主张：第一，总结构信息和可迁移结构信息必须区分；第二，结构是否可用必须相对于具体观察者定义。它不能替代大模型实验，但为后续在语言、代码、棋类和合成数据上开展更昂贵实验提供了最小可复现 sanity check。

## 10. 第二个受控实验：next-token MLP 与学习进度 proxy

为了补充一个更接近语言模型训练动态的证据，我们实现了一个小型 autoregressive next-token 实验，脚本见 `experiments/sequence_mlp_transfer.py`。该实验使用一个 NumPy 实现的一隐藏层 MLP 预测二进制序列的下一个 token，随后冻结隐藏表示，用 8 个目标标签训练 ridge probe 来分类目标 recurrence family。

该实验比较四种源数据：

- `random`：随机二进制序列。
- `irrelevant_rule`：具有可学习 next-token 结构，但规则与目标标签无关。
- `relevant_rule`：next-token 规则与目标 recurrence 标签对齐。
- `mixed_rule`：同时混合相关与无关结构。

结果见 `experiments/results/sequence_mlp_transfer_summary.md` 和 `experiments/results/sequence_mlp_transfer_analysis.md`。核心表格如下：

| 源数据 | 初始 val loss | 最终 val loss | learning progress proxy | loss reduction | 迁移准确率 |
|---|---:|---:|---:|---:|---:|
| random | 0.6984 +/- 0.0023 | 0.6933 +/- 0.0001 | 0.0115 +/- 0.0060 | 0.0051 +/- 0.0024 | 0.5277 +/- 0.0876 |
| irrelevant_rule | 0.6982 +/- 0.0048 | 0.5855 +/- 0.1161 | 1.5835 +/- 1.5587 | 0.1126 +/- 0.1161 | 0.5380 +/- 0.1105 |
| relevant_rule | 0.4871 +/- 0.0615 | 0.2599 +/- 0.0109 | 0.4760 +/- 0.1213 | 0.2272 +/- 0.0605 | 0.5685 +/- 0.0865 |
| mixed_rule | 0.6990 +/- 0.0031 | 0.6399 +/- 0.0646 | 0.7811 +/- 0.8659 | 0.0591 +/- 0.0644 | 0.5859 +/- 0.1044 |

这个实验比 PCA sanity check 更弱，方差也更大，但它提供了一个语言模型式补充：`irrelevant_rule` 的 learning progress proxy 比 `relevant_rule` 高 `1.1075`，但迁移准确率低 `0.0305`；在所有 runs 上，`learning_progress_proxy` 与迁移准确率的 Pearson 相关为 `-0.1084`，`loss_reduction` 与迁移准确率的 Pearson 相关为 `-0.0162`。换言之，在这个受控设置下，仅仅“能被 next-token 模型学到”并不能推出“对目标任务有迁移价值”。图表见 `experiments/figures/sequence_mlp_bars.svg`。

该结果不应被解读为对大模型的强实证结论。它的作用是补充 PCA 实验所缺少的训练动态元素：即便用 autoregressive 训练和学习曲线 proxy，任务无关的可学习结构仍可能与迁移表现脱钩。因此，正式的大模型实验应进一步比较 prequential epiplexity、loss slope、target relevance 和 OOD transfer。

## 11. 理论讨论

### 11.1 不违反数据处理不等式

本文所说的“计算创造信息”不是指违反 Shannon 意义下的数据处理不等式。更精确地说，确定性计算可以把对有限观察者不可用的隐含结构，转化为有限计算内可学习、可压缩、可迁移的结构。信息的增加是相对于观察者与计算预算而言的。

### 11.2 为什么生成过程简单，预测程序可能复杂

采样与似然评估并不等价。一个生成者可以先采样隐藏变量再生成观测；预测者面对观测时却必须反推隐藏变量。作者写侦探小说时可以先决定凶手，读者预测凶手时必须归纳证据。模型通过最大似然学习时，经常站在读者而不是作者的位置。

因此，数据生成程序短，不意味着最优预测程序短。在有限计算下，模型可能需要学到比生成过程更丰富的中间结构。

### 11.3 为什么上下文能改变智能表现

上下文改变的不是权重，而是当前问题的有效编码。一个好的上下文可以减少搜索空间、暴露隐变量、提供局部规则、指定表示方式、调用相关子程序。于是同一模型在不同上下文中表现出不同能力，不是因为模型本体变了，而是因为当前可用结构变了。

## 12. 局限

第一，`TRE` 依赖目标任务，因此不再是完全任务无关的数据价值指标。这是有意为之：迁移能力本来就需要目标参照。

第二，`Omega` 的定义增加了实验复杂度。不同模型族和 tokenizer 之间的 epiplexity 不应被轻易横向比较，除非观察者规范明确。

第三，估计方法仍然是 proxy。Prequential coding 方便但启发式强，requential coding 更严格但昂贵。实际使用时需要同时报告估计误差、训练动态和控制组。

第四，任务相关结构可能以非线性方式组合。两个源数据集单独 `TRE` 不高，混合后可能产生新的迁移结构。本文定义可扩展到数据组合，但实验上需要额外设计。

第五，高 `TRE` 仍不保证真实世界可靠性。数据污染、评估泄漏、社会偏见、工具错误和分布漂移都可能让结构看似可迁移但实际不可用。

## 13. 结论

Epiplexity 把大模型学习从“记忆数据”或“拟合分布”重新解释为计算受限条件下的结构提取：模型为了压缩和预测数据，必须构造可复用内部程序。本文进一步提出，理解智能和泛化不仅要问“数据中有多少结构”，还要问“这些结构对哪个观察者可见、对哪个目标任务有用、能否在当前上下文中被调用”。

因此，本文主张将大模型智能写成一个组合量：

```text
表现出的智能
≈ 参数中已积累的结构
 + 当前上下文暴露的结构
 + 目标任务可复用的结构
 + 推理时可用的计算与工具
```

在这个视角下，训练数据、数据顺序、合成数据、prompt、检索和工具调用不再是分散技巧，而是同一个问题的不同侧面：如何让计算受限模型获得、保存、暴露并复用结构信息。

本文没有声称给出智能的终极定律。相反，它给出一套更可证伪的研究纲领：用 observer specification 固定比较对象，用 task-relevant epiplexity 区分结构与有用结构，用 contextual epiplexity 描述推理时结构，用负例和因果干预检验理论边界。若这些指标在受控数据选择和迁移实验中稳定超过传统 proxy，它们将为大模型训练提供一种比困惑度更接近能力本质的数据价值度量。

## 参考文献

Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence*. arXiv:2601.03220.

Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165.

Olsson, C., et al. (2022). *In-context Learning and Induction Heads*. arXiv:2209.11895.

Delétang, G., et al. (2023). *Language Modeling Is Compression*. arXiv:2309.10668.

Rissanen, J. (2004). *MDL Denoising*. IEEE Transactions on Information Theory.

Grünwald, P. (2007). *The Minimum Description Length Principle*. MIT Press.

Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. ICML.

Li, H. (2026). *A Controlled Counterexample to Strong Proxy-Based Explanations of OOD Performance: in a Fixed Pretraining-and-Probing Setup*. arXiv:2605.11554.
