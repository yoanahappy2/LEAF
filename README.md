# LEAF: Layered Evaluation and Assurance Framework

**Discovering and Mitigating Query Drift in ReAct-based Retrieval Systems for Low-Resource Language Translation**

> A Multi-Agent framework that achieves 95% translation accuracy for Paiwan (an endangered Austronesian language) through layered quality assurance, using only 242 corpus entries.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Course](https://img.shields.io/badge/Course-大模型驱动的软件开发-orange)]()

---

## TL;DR

| Metric | Baseline (LLM zero-shot) | RAG only | **LEAF (Full Pipeline)** |
|--------|--------------------------|----------|--------------------------|
| BLEU | 0.000 | 0.452 | — |
| Exact Match | 0% | 90% | **95%** |
| Recovery Rate | N/A | N/A | **6/6** (failed cases rescued) |

**Key insight**: Standard RAG is fragile — a single paraphrase by the LLM causes *Query Drift*, derailing retrieval. LEAF's four-layer architecture prevents and recovers from this failure mode.

---

## What is Query Drift?

In ReAct-based translation systems, the LLM may paraphrase the source query before retrieving:

```
Original:  "tjima" (誰)        →  RAG retrieves "tjima" → 誰  ✓
Paraphrased: "想知道是誰" (想知道是谁) →  RAG retrieves unrelated entries → ✗
```

This *Query Drift* phenomenon is a stable failure mode in low-resource settings, where the retrieval corpus is too small for fuzzy matching to recover.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │    Orchestrator Agent     │
                    │  (ReAct + Tool Routing)  │
                    └────────┬────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Knowledge   │  │   Quality    │  │  Teaching    │
   │   Agent     │  │   Agent      │  │   Agent      │
   │ (RAG+Trans) │  │ (Verify+Fix) │  │ (Learn+Eval) │
   └─────────────┘  └──────────────┘  └──────────────┘
            │                │
            ▼                ▼
   ┌─────────────────────────────────────┐
   │     Four-Layer Quality Assurance    │
   │                                     │
   │  L1: CoreVocab  — 關鍵詞詞典約束    │
   │  L2: RAG        — 語料庫檢索增強    │
   │  L3: Grammar    — 音韻/綴詞規則引擎  │
   │  L4: Verify     — 反向翻譯驗證      │
   └─────────────────────────────────────┘
            │
            ▼
   ┌─────────────────────────────────────┐
   │         Corpus Layer                │
   │  242 entries · Paiwan↔Chinese       │
   │  Keyword + Vector (FAISS) index     │
   └─────────────────────────────────────┘
```

### Four Layers Explained

| Layer | Name | Role | Contribution |
|-------|------|------|-------------|
| **L1** | CoreVocab | Constrain output to known vocabulary | +33pp (largest single contributor) |
| **L2** | RAG | Retrieve relevant corpus entries | Enables context-aware translation |
| **L3** | Grammar | Phonological rules (tj→t, lj→l) + affix analysis | Handles spelling variation |
| **L4** | Verify | Reverse-translate and compare | Catches hallucination; enables recovery |

---

## Project Structure

```
LEAF/
├── src/                          # Core framework
│   ├── framework/                # Framework __init__
│   ├── core/                     # Base classes
│   │   ├── agent.py              # BaseAgent + AgentTrace
│   │   ├── message.py            # MessageBus + AgentMessage
│   │   ├── loop.py               # ReAct execution loop
│   │   ├── state.py              # Conversation state management
│   │   ├── decision.py           # Decision logging for ablation
│   │   ├── strategy.py           # Learning strategies
│   │   └── rate_limiter.py       # API rate limiting
│   ├── agents/                   # Specialized agents
│   │   ├── orchestrator.py       # Orchestrator (ReAct + routing)
│   │   ├── knowledge_agent.py    # Knowledge retrieval + translation
│   │   ├── quality_agent.py      # Quality control + reverse verification
│   │   └── teaching_agent.py     # Teaching & evaluation
│   ├── rag_service.py            # RAG retrieval service
│   ├── llm_service.py            # LLM API wrapper
│   └── translate_service.py      # Translation pipeline
│
├── benchmark/                    # Evaluation suite
│   ├── run_baseline.py           # Zero-shot baseline
│   ├── run_rag_benchmark.py      # RAG keyword retrieval
│   ├── run_rag_vector_benchmark.py # RAG vector retrieval
│   ├── run_icl_benchmark.py      # ICL comparison
│   ├── run_benchmark_v2.1.py     # Full multi-agent benchmark
│   ├── run_trace.py              # Query Drift trace analysis
│   ├── metrics.py                # BLEU + exact match metrics
│   ├── gen_plot.py               # Ablation charts
│   ├── results/                  # Raw experiment results (JSON)
│   ├── test_set.json             # 40-item test set
│   ├── ground_truth.json         # Ground truth annotations
│   └── lexicon_v2.json           # Paiwan-Chinese lexicon
│
├── experiments/                  # Ablation & critical experiments
│   ├── ablation_study.py         # Layer-by-layer ablation
│   ├── ablation_v3.py            # 4-layer ablation (L1-L4)
│   ├── ablation_v4.py            # Extended ablation
│   ├── agent_ablation.py         # Agent-level ablation
│   ├── strategy_ablation.py      # Strategy comparison
│   ├── bilingual_ablation.py     # Cross-language (Paiwan + Amis)
│   ├── critical_experiment.py    # Query Drift reproduction
│   ├── constrained_react.py      # Constrained ReAct (no paraphrase)
│   ├── multi_agent_benchmark.py  # Multi-agent evaluation
│   └── *.json                    # Experiment result files
│
├── data/                         # Corpus & linguistic resources
│   ├── merged_corpus.json        # Primary corpus (4,844 entries)
│   ├── klokah_paiwan_corpus.json # Klokah dictionary source
│   ├── paiwan_dictionary.json    # Structured dictionary
│   ├── grammar_rules.json        # Paiwan grammar rules
│   └── prompt_vocab.json         # Prompt vocabulary constraints
│
└── docs/                         # Documentation & reports
    ├── BASELINE_REPORT.md        # Baseline experiment report
    ├── A_LINE_FULL_RECORD.md     # Complete A-line experiment log
    ├── agent_advantage_analysis.md # Multi-agent advantage analysis
    ├── error_analysis.md         # Error categorization
    ├── case_study.md             # Query Drift case studies
    ├── LEAF_實驗回顧.md          # Full experiment retrospective (ZH)
    └── LEAF_演講稿_v7.md        # Presentation script (ZH)
```

---

## Key Results

### A-Line: Accuracy

| Configuration | Exact Match | BLEU |
|--------------|-------------|------|
| Zero-shot (GLM-4-Flash) | 0% | 0.000 |
| Zero-shot (GLM-4-Plus) | 2.5% | 0.000 |
| RAG keyword (242 entries) | 90% | 0.452 |
| RAG vector (242 entries) | 85% | 0.416 |
| ICL 20-shot | 7.5% | 0.000 |
| **LEAF Full Pipeline** | **95%** | — |

### B-Line: Recovery (Query Drift)

In 6 deliberately constructed Query Drift failure cases:

| Method | Recovery Rate |
|--------|--------------|
| Prompt-only (ask LLM to be careful) | 0/6 |
| **LEAF Orchestrator + ReAct** | **6/6** |

### Ablation: Layer Contributions

| Layer Removed | Δ Accuracy | Key Insight |
|--------------|-----------|-------------|
| L1 (CoreVocab) | −33pp | Largest single contributor |
| L4 (Verify) | −5pp | Catches remaining hallucination |
| L2 (RAG) | −60pp | Essential; without it, BLEU = 0 |

### Cross-Language Validation

Verified on Amis (阿美語, ISO 639-3: `ami`) — another endangered Austronesian language — confirming framework transferability.

---

## Quick Start

### Prerequisites

```bash
pip install openai faiss-cpu numpy matplotlib pyyaml
```

### Set API Key

```bash
export ZHIPU_API_KEY="your-key-here"  # GLM-4 API
```

### Run Benchmark

```bash
# Baseline (zero-shot)
python benchmark/run_baseline.py

# RAG benchmark
python benchmark/run_rag_benchmark.py

# Full LEAF pipeline
python experiments/multi_agent_benchmark_v2.py

# Ablation study
python experiments/ablation_v4.py
```

---

## Course Information

- **Course**: 大模型驱动的软件开发 (Large Model-Driven Software Development)
- **University**: 清华大学 (Tsinghua University)
- **Student**: 黄咏郁 (Huang Yong-Yu), 2026110008
- **Semester**: 2026 Spring (Exchange Student)

---

## Citation

```bibtex
@misc{leaf2026,
  title={LEAF: Layered Evaluation and Assurance Framework for Low-Resource Language Translation},
  author={Huang, Yong-Yu},
  year={2026},
  howpublished={\\url{https://github.com/yoanahappy2/LEAF}}
}
```

## License

MIT License — see [LICENSE](LICENSE).
