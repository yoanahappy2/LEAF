---
title: "LEAF: 面向低资源语言翻译的分层评估与保证框架"
subtitle: "发现并缓解 ReAct 检索系统中的查询漂移问题"
author: "黄咏郁（2026110008）"
date: "2026年6月"
course: "大模型驱动的软件开发 · 清华大学"
---

# 摘要

全球 7,000 种语言中超过 40% 面临濒危，主流大语言模型（GPT-4、GLM-4）对低资源语言的零样本翻译 BLEU 得分为 0——完全无法工作。本文提出 **LEAF（Layered Evaluation and Assurance Framework）**，一个面向低资源语言翻译品质保证的 Multi-Agent 框架，以排湾语（Paiwan, ISO 639-3: pag）为验证语言。

我们在 ReAct 架构的检索增强翻译系统中发现了一种稳定的失败模式——**查询漂移（Query Drift）**：LLM 在迭代推理过程中擅自改写检索查询，导致精确匹配失败，漂移率高达 35%，其中 71% 的漂移直接导致翻译错误。LEAF 通过四层品质保证架构（CoreVocab / RAG / Grammar / Verify）和 Multi-Agent 编排机制，将查询漂移率从 35% 降至 0%，翻译准确率从 65% 提升至 **95%**。消融实验证明，30 个百分点的提升全部来自架构设计本身（Coordinator 控制查询透传），而非单一组件或 prompt engineering。

实验结果在排湾语（5,923 条语料）和阿美语（6,413 条语料）上完成双语言交叉验证，证实框架的可迁移性。

**关键词**：低资源语言翻译、Multi-Agent 系统、ReAct、查询漂移、消融实验

\newpage

# 1 引言

## 1.1 问题背景

低资源语言（Low-Resource Language）面临双重困境：数字语料匮乏导致大语言模型无法直接处理，而语言濒危的紧迫性又要求技术方案必须"少数据、快部署"。以排湾语为例，主流 LLM（GLM-4-Flash、GLM-4-Plus）的零样本翻译 BLEU 得分为 0.0，完全匹配率为 0-2.5%（表 1）。

**表 1：零样本基线实验**

| 模型 | 方向 | 测试题数 | BLEU | 完全匹配率 |
|------|------|---------|------|-----------|
| GLM-4-Flash | 中文→排湾语 | 40 | 0.000 | 0.0% |
| GLM-4-Plus | 中文→排湾语 | 40 | 0.000 | 2.5% |
| ICL 5-shot | 中文→排湾语 | 40 | 0.000 | 0.0% |
| ICL 10-shot | 中文→排湾语 | 40 | 0.000 | 0.0% |
| ICL 20-shot | 中文→排湾语 | 40 | 0.000 | 7.5% |

即使提供 20 个示例（ICL），BLEU 仍为 0——排湾语在 LLM 预训练数据中几乎不存在。In-Context Learning 路线被彻底排除。

## 1.2 RAG 的希望与脆弱性

检索增强生成（RAG）是唯一可行路线。仅用 242 条语料，关键词 RAG 即可将完全匹配率提升至 90%（表 2）。

**表 2：RAG 基线实验**

| 方法 | 语料量 | BLEU | 完全匹配率 |
|------|--------|------|-----------|
| 关键词 RAG | 242 条 | 0.452 | 90.0% |
| 向量 RAG | 242 条 | 0.416 | 85.0% |
| 混合 RAG | 4,844 条 | 0.873 | 77.5% |

然而，RAG 在低资源场景中极其脆弱——检索语料太少，容不下任何模糊匹配。当 LLM 在 ReAct 循环中改写查询时（如"月亮"→"月亮的族语怎么说"），精确匹配立刻失败。这就是查询漂移问题。

## 1.3 贡献

本文的贡献有三：

1. **量化查询漂移现象**：在 ReAct 单 Agent 系统中，LLM 改写查询的漂移率达 35%，其中 71% 直接导致翻译错误
2. **提出 LEAF 框架**：四层品质保证 + Multi-Agent 编排，架构级约束查询透传，将漂移率降至 0%
3. **完整消融验证**：8 配置 Truth Table + 双语言交叉验证，证明改进来自架构设计而非单一组件

\newpage

# 2 相关工作

**低资源语言 NLP**。传统方法依赖平行语料训练统计机器翻译模型（Koehn et al., 2007），但数千条语料远不足以训练可用模型。近年工作探索零样本跨语言迁移（Conneau et al., 2020），但效果依赖于目标语言与高资源语言的类型学相似性。

**RAG 在翻译中的应用**。检索增强生成（Lewis et al., 2020）通过外部知识库补充 LLM 知识盲区。在翻译场景中，RAG 可以将低资源语言语料作为检索源，实现"无需训练"的翻译服务。但现有工作未关注检索查询被 LLM 改写导致的失败问题。

**Multi-Agent 系统**。AutoGen（Wu et al., 2023）、CrewAI 等框架通过 Agent 分工协作处理复杂任务。现有工作多关注任务分解和 Agent 间通信，较少研究 Multi-Agent 编排对检索品质的影响。

**ReAct 范式**。Yao et al. (2023) 提出的 ReAct（Reasoning + Acting）让 LLM 在推理过程中调用工具。本文发现，正是 ReAct 的"自由推理"特性在低资源检索场景中引发了查询漂移。

\newpage

# 3 方法

## 3.1 系统架构

LEAF 采用 Multi-Agent 架构，包含 3 个核心 Agent：

- **Coordinator Agent（协调智能体）**：系统协调者，负责 ReAct 流程控制与工具路由。核心职责是**保护用户原始 Query**——无论后续进行多少轮推理或工具调用，原始问题都会被完整保留，不被 LLM 自动改写
- **Knowledge Agent（知识智能体）**：封装所有语言知识来源，包括 Exact Match、RAG Search 和 CoreVocab 词汇库，负责找到候选翻译结果
- **Quality Agent（品质智能体）**：通过反向翻译验证检查候选答案是否正确。验证失败则触发下一轮检索与修正

简言之：Knowledge Agent 负责「找得到」，Quality Agent 负责「找对了」，Coordinator Agent 负责确保整个流程不会因 Query Drift 而偏离用户意图。

Agent 间通过结构化 `AgentMessage` 通信（非自然语言），由 `MessageBus` 统一调度。

## 3.2 四层品质保证

| 层级 | 名称 | 功能 | 关键机制 |
|------|------|------|---------|
| L1 | CoreVocab | 词汇约束 | 关键词词典 + 精确匹配，将输出约束在已知词汇空间内 |
| L2 | RAG | 检索增强 | 关键词 + 向量（FAISS）混合检索，从语料库获取翻译参考 |
| L3 | Grammar | 语法注入 | 音韵规则引擎（tj→t, lj→l）+ 缀词分析，处理拼写变体 |
| L4 | Verify | 反向验证 | 将候选翻译反向翻译，检查是否回到原文，捕获幻觉 |

## 3.3 查询漂移的发现与定义

在 ReAct 单 Agent 系统中，LLM 在推理过程中会改写传给检索工具的查询。我们将这种行为定义为**查询漂移（Query Drift）**：

$$\text{Drift}(q_{orig}, q_{retrieval}) = \mathbb{1}[q_{orig} \neq q_{retrieval}]$$

其中 $q_{orig}$ 是用户原始输入，$q_{retrieval}$ 是实际传给检索工具的查询。

**漂移类型分类**：

| 类型 | 示例 | 机制 |
|------|------|------|
| **扩展型** | 月亮→月亮的族语怎么说 | LLM 添加解释性后缀 |
| **改写型** | 孩子→小孩的→小孩子 | LLM 替换为同义词 |
| **语义漂移型** | 朋友→朋友的族语→朋友说什么语言 | LLM 逐步偏离原意 |

## 3.4 架构级约束机制

LEAF 解决查询漂移的核心策略不是 prompt engineering，而是**架构设计**：

1. **查询透传**：Coordinator 的 system prompt 明确指示"用户原始输入直接作为 translate 的 text 参数"，且在代码层面确保查询原封不动传递
2. **结构化工具调用**：Coordinator 的工具 schema 设计使得 LLM 无需（也无法）改写查询参数
3. **失败回退策略**：翻译失败时不改写查询，而是切换检索策略（translate → rag_search → lookup）

\newpage

# 4 实验

## 4.1 实验设置

- **验证语言**：排湾语（Paiwan, ISO 639-3: pag），5,923 条语料
- **迁移验证**：阿美语（Amis, ISO 639-3: ami），6,413 条语料
- **基座模型**：GLM-4-Flash（智谱 AI）
- **测试集**：20 题中文→排湾语翻译（benchmark_v2.json）
- **评估指标**：BLEU、完全匹配率（Exact Match）、查询漂移率

## 4.2 实验 A：8 配置消融 Truth Table

从"无工具"到"完整 Multi-Agent"，逐步添加组件（表 3）。

**表 3：8 配置消融实验**

| # | 配置 | 准确率 | 正确/20 | 与前一步差值 |
|---|------|--------|---------|-------------|
| 1 | LLM Direct（无工具） | 0% | 0/20 | — |
| 2 | RAG Only（直接匹配） | 80% | 16/20 | +80pp |
| 3 | Single Agent（无 ReAct） | 80% | 16/20 | 0pp |
| 4 | SA + ReAct（free） | 65% | 13/20 | **-15pp** |
| 5 | SA + Constrained ReAct | ~80% | 12/15 | +15pp vs #4 |
| 6 | Multi-Agent w/ SA prompt | 90% | 18/20 | +25pp |
| 7 | Multi-Agent w/o Quality | 95% | 19/20 | +5pp |
| 8 | **Multi-Agent（完整）** | **95%** | **19/20** | 0pp |

**关键发现**：配置 #3→#4 是唯一出现准确率**下降**的步骤——引入 ReAct 后反而降了 15pp。这是因为 LLM 的"自由推理"能力改写了检索查询。而配置 #4→#6 的 +25pp 则是 Coordinator 架构约束带来的恢复。

## 4.3 实验 B：查询漂移量化

**表 4：查询漂移率对比**

| 配置 | 漂移次数 / 20 | 漂移率 | 漂移→错误率 |
|------|-------------|--------|------------|
| SA 无 ReAct | 0/20 | 0% | N/A |
| SA + ReAct（free） | **7/20** | **35%** | **71%（5/7）** |
| SA + Constrained ReAct | 6/20 | 30% | — |
| Multi-Agent + ReAct | **0/20** | **0%** | N/A |

即使使用 Constrained Prompt（明确要求"不要改写查询"），漂移率仍达 30%。这证明 prompt engineering 无法完全阻止查询漂移，**架构级约束是必要的**。

## 4.4 实验 C：组件归因

**表 5：组件消融（从完整 Multi-Agent 逐步移除）**

| 移除组件 | 准确率 | Δ vs 完整 | 结论 |
|---------|--------|----------|------|
| —（完整） | 95% | — | — |
| Quality Agent | 95% | 0pp | 验证层对当前模型影响有限 |
| Pre-routing | 95% | 0pp | 快捷路径不影响准确性 |
| System Prompt（换 SA prompt） | 90% | -5pp | prompt 指导有 5pp 贡献 |
| **Architecture（整个 Coordinator→SA+ReAct）** | **65%** | **-30pp** | **架构本身是核心贡献者** |

**95% 的提升归因**：架构控制 +25pp（Coordinator 确保查询不漂移），系统 prompt +5pp，其余组件 0pp。

## 4.5 实验 D：四层翻译品质消融（双语言验证）

**表 6：四层翻译品质消融**

| 实验组 | 排湾语 | Δ | 阿美语 | Δ |
|--------|--------|---|--------|---|
| Control（四层全开） | 79.2% | — | 73.5% | — |
| 关闭 L1 CoreVocab | 45.8% | **-33.4pp** | 38.8% | **-34.7pp** |
| 关闭 L3 Grammar | 75.0% | -4.2pp | 73.5% | 0pp |
| 关闭 L4 Verify | 83.3% | +4.1pp | 73.5% | 0pp |

L1（CoreVocab）是系统的绝对支柱，关闭后两语言均下降 33-35pp。框架可迁移性验证成功：两语言 Control 仅差 5.7pp（79.2% vs 73.5%）。

## 4.6 实验 E：Query Drift 恢复（B 线）

构造 6 个刻意设计的 Query Drift 失败案例，测试恢复能力：

| 方法 | 恢复率 |
|------|--------|
| Prompt-only（要求 LLM 注意） | 0/6 |
| **LEAF Coordinator + ReAct** | **6/6** |

\newpage

# 5 讨论

## 5.1 为什么架构设计优于 Prompt Engineering

实验数据清楚地表明，查询漂移是 LLM 的内在行为倾向——即使 Constrained Prompt 也只能将漂移率从 35% 降至 30%（表 4）。根本原因在于 ReAct 范式让 LLM 在"推理"步骤中自然地改写查询，这是推理能力的副作用，无法通过 prompt 完全抑制。

LEAF 的 Coordinator 通过以下架构设计解决了这个问题：

- **职责分离**：Coordinator 负责"决定做什么"，Knowledge Agent 负责"怎么做"，查询内容不在 Coordinator 的修改范围内
- **结构化传递**：查询作为结构化参数（而非自然语言文本）在 Agent 间传递
- **失败恢复**：翻译失败时切换检索策略而非改写查询

## 5.2 可迁移性

框架在排湾语和阿美语两种南岛语族语言上验证成功。两种语言的语料规模相近（5,923 vs 6,413 条），但语法结构差异显著。四层消融中 CoreVocab 的贡献在两语言上一致（-33pp vs -35pp），说明该层捕捉到了低资源翻译的通用瓶颈。

## 5.3 局限性

1. **测试规模**：主要消融实验使用 20 题测试集，规模较小，但覆盖了常见词汇类别
2. **基座模型依赖**：部分实验（如反向验证）在 GLM-4-Flash 上效果不稳定，更强模型可能提升 L4 贡献
3. **单方向翻译**：实验主要验证中文→排湾语方向，反向翻译的查询漂移行为可能有差异

\newpage

# 6 结论

本文提出了 LEAF 框架，量化并解决了 ReAct 检索系统中的查询漂移问题。核心发现是：

1. **查询漂移是 ReAct 的固有缺陷**：在低资源检索场景中，LLM 改写查询的漂移率达 35%，其中 71% 直接导致翻译错误。Prompt engineering 最多将漂移率降至 30%，无法根治。

2. **架构级约束是解决方案**：Multi-Agent 编排通过职责分离和结构化传递，将查询漂移率降至 0%，翻译准确率从 65% 提升至 95%（+30pp）。消融实验证实，这一提升全部来自架构设计本身。

3. **框架可迁移**：在排湾语和阿美语上完成双语言验证，四层架构的贡献模式一致。

LEAF 为低资源语言翻译提供了一种"少语料、高品质"的技术路线，核心洞察——**架构设计优于 prompt engineering** ——对其他需要精确检索的 ReAct 应用场景亦有参考价值。

\newpage

# 参考文献

1. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
2. Koehn, P., Hoang, H., Birch, A., Callison-Burch, C., Federico, M., Bertoldi, N., ... & Herbst, E. (2007). Moses: Open source toolkit for statistical machine translation. *ACL 2007*.
3. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
4. Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., ... & Awadallah, A. H. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.
5. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*.

\newpage

# 附录 A：系统架构详细设计

## A.1 Agent 通信机制

Agent 间通过 `AgentMessage` 结构体通信，包含以下字段：

```
AgentMessage:
  - type: MessageType (TRANSLATE / RETRIEVE / VERIFY / EVALUATE / ERROR)
  - sender: str (agent role)
  - receiver: str (target agent role)
  - content: dict (structured data, NOT natural language)
  - metadata: dict (trace info, timestamps)
```

`MessageBus` 负责路由和日志记录，确保所有 Agent 交互可追溯。

## A.2 Coordinator ReAct 循环

```
1. 接收用户输入 → 保存原始查询 q_orig
2. LLM 决策：选择工具 + 参数
   - translate(text=q_orig, direction="c2p")  ← 查询原封不动
   - rag_search(query=q_orig)                  ← 查询原封不动
   - lookup(word=...)                           ← 从 RAG 结果提取
3. 执行工具 → 获取结果
4. 若结果有效 → 返回
   若结果无效 → ReAct 自纠正（切换检索策略，不改写查询）
5. 最多重试 3 次
```

## A.3 Quality Agent 反向验证 Pipeline

```
1. 接收候选翻译 candidate = "kina"（声称是"母亲"的排湾语）
2. 反向翻译：kina → (排湾语→中文) → "母亲"
3. 比对：反向结果 ≈ 原文 "母亲" → PASS
4. 反例：candidate = "tjina"
   → 反向翻译：tjina → "???" ≠ "母亲" → FAIL → 触发重试
```

# 附录 B：完整实验数据

## B.1 Query Drift 失败案例详细分析

| # | 原始查询 | 漂移后查询 | 漂移类型 | 正确答案 | 实际结果 | SA+ReAct | MA |
|---|---------|-----------|---------|---------|---------|----------|-----|
| q06 | 月亮 | 月亮的族语怎么说 | 扩展型 | qiljas | ❌ | ❌ | ✅ |
| q12 | 父亲 | 爸爸的排湾语 | 改写型 | ama | ❌ | ❌ | ✅ |
| q13 | 孩子 | 小孩子的怎么说 | 改写型 | aljak | ❌ | ❌ | ✅ |
| q16 | 人 | 人的族语是什么 | 扩展型 | caucau | ❌ | ❌ | ✅ |
| q17 | 朋友 | 朋友说什么语言 | 语义漂移型 | kabang | ❌ | ❌ | ✅ |
| q19 | 名字 | 名字怎么念 | 改写型 | ngadan | ❌ | ❌ | ✅ |

所有 6 个漂移失败案例在 Multi-Agent 配置中全部恢复。

## B.2 双语言消融详细数据

**排湾语**（50 条测试，klokah 交叉验证）：

| 配置 | 正确数 | 准确率 |
|------|--------|--------|
| 四层全开 | 39.6/50 | 79.2% |
| - CoreVocab | 22.9/50 | 45.8% |
| - Grammar | 37.5/50 | 75.0% |
| - Verify | 41.7/50 | 83.3% |

**阿美语**（相同实验设置）：

| 配置 | 正确数 | 准确率 |
|------|--------|--------|
| 四层全开 | 36.8/50 | 73.5% |
| - CoreVocab | 19.4/50 | 38.8% |
| - Grammar | 36.8/50 | 73.5% |
| - Verify | 36.8/50 | 73.5% |

# 附录 C：代码仓库结构

GitHub: https://github.com/yoanahappy2/LEAF

```
LEAF/
├── src/                    # 框架核心代码
│   ├── core/              # BaseAgent, MessageBus, ReAct Loop
│   ├── agents/            # Coordinator, Knowledge, Quality
│   └── rag_service.py     # RAG 检索服务
├── benchmark/             # 评估套件
│   ├── run_baseline.py    # 零样本基线
│   ├── run_rag_benchmark.py # RAG 基线
│   ├── run_benchmark_v2.1.py # 完整 Multi-Agent 评估
│   ├── metrics.py         # BLEU + Exact Match
│   └── results/           # 原始实验数据 (JSON)
├── experiments/           # 消融实验脚本
│   ├── ablation_v4.py     # 四层消融
│   ├── agent_ablation.py  # Agent 级消融
│   ├── constrained_react.py # 约束 ReAct 对比
│   └── critical_experiment.py # Query Drift 复现
├── data/                  # 语料库与语言资源
│   ├── merged_corpus.json # 主语料 (4,844 条)
│   └── grammar_rules.json # 排湾语语法规则
└── docs/                  # 实验报告与分析
```
