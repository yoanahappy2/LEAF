# LEAF: A ReAct-Orchestrated Framework for Endangered Language Learning
## 完整專案紀錄（A線）

---

# 一、專案概述

## 1.1 專案名稱
**LEAF** — Layered Evaluation & Adaptive Framework for Endangered Language Learning

## 1.2 一句話定位
> 一個基於 ReAct 循環的排灣語翻譯框架，透過迭代推理與自適應檢索，將低資源語言翻譯準確率從 70% 提升至 85%。

## 1.3 核心論點
> The performance gain comes from **iterative reasoning and adaptive retrieval**, not single-shot RAG.

## 1.4 課程資訊
- 課程：大模型驅動的軟件開發（唐杰老師）
- 學校：北京清華大學
- 展示：Poster 1.2m×0.9m
- 語料：排灣語（Paiwan），台灣原住民族語，瀕危語言

---

# 二、系統架構

## 2.1 架構圖

```
用戶輸入
    ↓
Orchestrator (ReAct Loop)
    ├─ Decision Log: 決策記錄
    ├─ Strategy: Mastery/Exploration/Exam
    ↓
KnowledgeAgent (語言知識服務)
    ├─ translate() — 精確匹配 + FAISS 語意搜尋
    ├─ rag_search() — 語料庫檢索
    ├─ lookup() — 詞彙深度查詢
    └─ pronunciation() — 發音查詢
    ↓
TeachingAgent (學習路徑服務)
    ├─ suggest_next() — 推薦下一個學習詞彙
    ├─ generate_quiz() — 生成測驗
    └─ record_learning() — 記錄學習行為
    ↓
QualityAgent (品質保證服務)
    ├─ review_translation() — 審核翻譯品質
    └─ self_judge() — 目標是否達成
    ↓
Orchestrator 整合結果 → 輸出
```

## 2.2 通訊協議
- AgentMessage（結構化 JSON，非自然語言）
- MessageBus（路由中心）
- 每條訊息含：from, to, type, payload, meta

## 2.3 ReAct 循環
```
while not resolved and turns < max_turns:
    1. LLM 觀察當前狀態（前次工具結果）
    2. LLM 決定下一步（調用哪個工具、什麼參數）
    3. 執行工具，獲得結果
    4. 判斷是否已解決 → 是則輸出，否則繼續
```

## 2.4 Decision Log
每一步記錄四要素：
- Situation: 當前狀態
- Options: 可用選項
- Chosen: 選擇了什麼
- Reasoning: 為什麼這樣選
- Confidence: 信心分數

---

# 三、語料庫

## 3.1 規模
- 原始語料：10,195 筆
- 清洗後：5,923 筆
- 核心詞彙表：47 筆（含 44 反向映射）
- 精確匹配索引：5,069 排灣語詞、5,014 中文詞
- 變體索引：66 個拼寫變體

## 3.2 來源
- klokah 教材語料（對話、閱讀、文化）
- 族語E樂園教材
- 政府原住民族語言資料

## 3.3 語料清洗
- 移除 4,165 筆重複/無效條目
- FAISS 向量索引（all-MiniLM-L6-v2, 384 維）
- 本地 embedding，zero API cost

---

# 四、Benchmark 設計

## 4.1 設計原則
- **variant-aware word-boundary match**：禁止 substring、fuzzy、LLM judge
- **語料庫驗證**：每個 variant 都必須在語料庫中存在
- **控制變因**：同模型（glm-4-flash）、同溫度（0.3）、同評分函數

## 4.2 評分標準
| 匹配類型 | 分數 | 定義 |
|---------|------|------|
| Exact Match | 1.0 | pred == preferred |
| Variant Match | 0.8 | pred in variants |
| No Match | 0.0 | 不在任何 variant 中 |

## 4.3 測試集
20 題基礎詞彙翻譯（中文 → 排灣語）

| ID | 中文 | 正確答案 | 難度 |
|----|------|---------|------|
| q01 | 你好 | djavadjavai/djavadjavay | easy |
| q02 | 謝謝 | masalu | easy |
| q03 | 水 | zaljum | easy |
| q04 | 吃 | keman | easy |
| q05 | 太陽 | qadaw | easy |
| q06 | 月亮 | qiljas | easy |
| q07 | 手 | lima | easy |
| q08 | 眼睛 | maca | easy |
| q09 | 火 | sapuy/sapui | easy |
| q10 | 星星 | vitjuqan | easy |
| q11 | 母親 | kina | medium |
| q12 | 父親 | kama | medium |
| q13 | 孩子 | aljak | easy |
| q14 | 房子 | umaq | medium |
| q15 | 道路 | djalan | medium |
| q16 | 人 | caucau | easy |
| q17 | 朋友 | drangi | medium |
| q18 | 山 | gadu | medium |
| q19 | 名字 | ngadan | medium |
| q20 | 狗 | vatu | easy |

## 4.4 Lexicon 審計
- 總 variant 數：28
- ✅ VERIFIED：28（100%）
- ❌ INVALID：0
- 已移除的錯誤 variants：tjina, ina, tama, ama, mata, kan, sadju, kadu
- 審計報告：`benchmark/corpus_audit_report.md`

---

# 五、實驗結果

## 5.1 消融實驗結果

| 配置 | 準確率 | avg_score | 正確/總數 | 錯題數 |
|------|--------|-----------|----------|--------|
| LLM Direct | 0.0% | 0.00 | 0/20 | 20 |
| Single-Shot RAG | 70.0% | 0.68 | 14/20 | 6 |
| Single Agent | 70.0% | 0.68 | 14/20 | 6 |
| **ReAct Orchestrator** | **85.0%** | 0.83 | 17/20 | 3 |

## 5.2 逐題結果

| ID | 中文 | 答案 | LLM | RAG | Single | ReAct | 備註 |
|----|------|------|-----|-----|--------|-------|------|
| q01 | 你好 | djavadjavai | ✗ | ✓ | ✓ | ✓ | |
| q02 | 謝謝 | masalu | ✗ | ✓ | ✓ | ✓ | |
| q03 | 水 | zaljum | ✗ | ✓ | ✓ | ✓ | |
| q04 | 吃 | keman | ✗ | ✓ | ✓ | ✓ | |
| q05 | 太陽 | qadaw | ✗ | ✓ | ✓ | ✓ | |
| q06 | 月亮 | qiljas | ✗ | ✓ | ✓ | ✓ | |
| q07 | 手 | lima | ✗ | ✓ | ✓ | ✓ | |
| q08 | 眼睛 | maca | ✗ | ✓ | ✓ | ✓ | |
| q09 | 火 | sapuy | ✗ | ✓ | ✓ | ✓ | |
| q10 | 星星 | vitjuqan | ✗ | ✓ | ✓ | ✓ | |
| q11 | 母親 | kina | ✗ | ✗ | ✗ | ✗ | 知識庫返回 tjina（髒數據）|
| q12 | 父親 | kama | ✗ | ✗ | ✗ | ✗ | 知識庫返回 tama（髒數據）|
| q13 | 孩子 | aljak | ✗ | ✓ | ✓ | ✓ | |
| q14 | 房子 | umaq | ✗ | ✗ | ✗ | **✓** | **ReAct 救回** |
| q15 | 道路 | djalan | ✗ | ✓ | ✓ | ✓ | |
| q16 | 人 | caucau | ✗ | ✗ | ✗ | **✓** | **ReAct 救回** |
| q17 | 朋友 | drangi | ✗ | ✓ | ✓ | ✓ | |
| q18 | 山 | gadu | ✗ | ✗ | ✗ | ✗ | 全部 hallucinate |
| q19 | 名字 | ngadan | ✗ | ✗ | ✗ | **✓** | **ReAct 救回** |
| q20 | 狗 | vatu | ✗ | ✓ | ✓ | ✓ | |

## 5.3 錯題分布

| 配置 | 錯題列表 |
|------|---------|
| LLM Direct | 全部 20 題 |
| RAG Only | 母親、父親、房子、人、山、名字 |
| Single Agent | 母親、父親、房子、人、山、名字 |
| ReAct Orchestrator | 母親、父親、山 |

---

# 六、ReAct 救回案例

## 6.1 Case q14：房子 → umaq

**場景**：RAG 精確匹配索引中「房子」沒有直接對應（語料庫登記為「家」「家屋」）

**Single-Shot RAG 流程**：
```
translate("房子", c2p) → 精確匹配失敗 → FAISS 搜尋返回不相關結果 → ""
結果：✗
```

**ReAct Orchestrator 流程**：
```
Turn 1: translate("房子", c2p) → "" (空)
Turn 2: rag_search("房子 house") → 語料例句（含「家」相關句子）
Turn 3: rag_search("house") → 更多語料
最終：LLM 從語料中提取 "umaq"
結果：✓
```

**救回機制**：
1. 偵測 translate 失敗（返回空）
2. 改用 rag_search + 英文 query 擴大搜尋範圍
3. LLM 從語料例句中推理出正確答案

**工具序列**：translate → rag_search → rag_search（3 次工具調用）

---

## 6.2 Case q16：人 → caucau

**場景**：RAG 精確匹配找不到「人」（語料釋義缺失）

**Single-Shot RAG 流程**：
```
translate("人", c2p) → 精確匹配失敗 → ""
結果：✗
```

**ReAct Orchestrator 流程**：
```
Turn 1: rag_search("人") → 語料例句（含 caucau 的句子）
Turn 2: translate("caucau", p2c) → 反向驗證（排灣→中文）
Turn 3: translate("人", c2p) → 再試一次
最終：LLM 從 rag_search 結果中識別 "caucau"
結果：✓
```

**救回機制**：
1. 先用 rag_search 語意搜尋找到候選詞 caucau
2. 做反向翻譯驗證（caucau → 中文）
3. LLM 綜合判斷輸出

**工具序列**：rag_search → translate(p2c 反向驗證) → translate(c2p)（3 次工具調用）

---

## 6.3 Case q19：名字 → ngadan

**場景**：RAG 精確匹配找不到「名字」

**Single-Shot RAG 流程**：
```
translate("名字", c2p) → 精確匹配失敗 → ""
結果：✗
```

**ReAct Orchestrator 流程**：
```
Turn 1: translate("名字", c2p) → "" (空)
Turn 2: rag_search("名字 name") → 語料例句
Turn 3: rag_search("個人名稱 personal name") → 更多語料
最終：LLM 從語料中提取 "ngadan"
結果：✓
```

**救回機制**：
1. translate 失敗後改用 rag_search
2. 用中英雙語 query 擴大覆蓋（「名字 name」「個人名稱 personal name」）
3. LLM query reformulation 是關鍵

**工具序列**：translate → rag_search(中英query) → rag_search(同義詞query)（3 次工具調用）

---

# 七、失敗案例分析

## 7.1 知識庫髒數據（q11 母親、q12 父親）

**問題**：RAG 知識庫的精確匹配索引映射了錯誤答案
- 母親 → tjina（不是排灣語）
- 父親 → tama（不是排灣語）

**ReAct 也救不回**：因為 KnowledgeAgent 返回的 tjina/tama 就是錯的，Orchestrator 沒有機制偵測「這個答案是知識庫的錯誤數據」

**已修正**：prompt_vocab.json 中 tjina→kina, tama→kama

## 7.2 全部 Hallucinate（q18 山）

**問題**：所有模型都輸出 qungiljaw（不是「山」的意思）
- RAG：FAISS 語意搜尋返回錯誤的詞
- ReAct：也拿到同樣的錯誤結果

**原因**：「山」這個 query 太短，語意搜尋匹配到不相關的語料

---

# 八、時間線

## 2026-06-04（完整開發日）

| 時間 | 事件 |
|------|------|
| 20:00 | 開始：確認系統架構（4 Agent + MessageBus + ReAct）|
| 20:05 | 審查 Decision Log + Strategy System |
| 20:15 | 完成 strategy_ablation.py 消融腳本 |
| 20:27 | Pre-flight check 全部通過 |
| 20:34 | 第 1 輪實驗：0%/0%/0%/85%（AgentMessage 參數 bug）|
| 20:47 | 修復 AgentMessage 構造函數 |
| 21:00 | 第 2 輪實驗：0%/75%/80%/90% |
| 21:10 | 確認語料庫中 djalan = 道路 |
| 21:25 | 開始 LEAF Benchmark v2.0 重構 |
| 21:30 | 建立 lexicon_v2.json + benchmark_v2.json + run_benchmark_v2.py |
| 21:36 | 第 3 輪實驗：0%/0%/0%/85%（AgentMessage bug again）|
| 21:38 | 修復 from_agent/to_agent |
| 21:40 | 第 4 輪實驗：0%/75%/80%/90% |
| 21:42 | 發現 djavadjavay 變體缺失，加進 lexicon |
| 21:47 | 第 5 輪實驗：0%/80%/85%/90% |
| 21:48 | 修復 str(dict) fallback bug |
| 22:05 | 第 6 輪實驗：0%/85%/85%/95% |
| 22:16 | **關鍵發現**：tjina/tama 不是排灣語 |
| 22:23 | Lexicon 全面審計：移除 8 個假 variants |
| 22:27 | 第 7 輪實驗：0%/70%/70%/85% |
| 22:34 | prompt_vocab 修正 tjina→kina, tama→kama |
| 22:37 | 第 8 輪實驗：0%/80%/80%/90%（tjina/tama 作為 variant）|
| 22:40 | 確認 tjina/tama 完全不是排灣語，移除 |
| 22:42 | **最終實驗：0%/70%/70%/85%** |
| 22:52 | Corpus Audit 完成：28/28 VERIFIED |
| 23:09 | 生成 Ground Truth + Error Analysis + Case Study |
| 23:12 | 生成 ablation_plot.png |
| 23:15 | 跑 Multi-Agent Trace（6 題）|
| 23:19 | **核心發現**：3 個 MA 成功案例全部只用 KnowledgeAgent |
| 23:19 | 確認優勢來自 ReAct 循環，不是多 Agent 協作 |
| 00:22 | 決定 A線/B線雙軌策略 |
| 00:26 | A線定位：ReAct-Orchestrated Framework |

---

# 九、控制變因

| 變因 | 值 |
|------|------|
| 模型 | glm-4-flash（智譜）|
| 溫度 | 0.3 |
| 評分函數 | evaluate_v2()（variant-aware word-boundary match）|
| Lexicon | lexicon_v2.json（28 variants, 全部 corpus-verified）|
| 語料庫 | merged_corpus.json（5,923 筆清洗後）|
| 測試集 | benchmark_v2.json（20 題）|

---

# 十、檔案清單

## 核心代碼
```
~/Desktop/paiwan_competition_2026/
├── agent_framework/
│   ├── core/
│   │   ├── agent.py          # BaseAgent
│   │   ├── message.py        # AgentMessage + MessageBus
│   │   ├── strategy.py       # LearningStrategy (Mastery/Exploration/Exam)
│   │   ├── decision.py       # Decision + DecisionLogger
│   │   ├── loop.py           # ReAct loop
│   │   ├── state.py          # State management
│   │   └── rate_limiter.py   # API rate limiting
│   └── agents/
│       ├── orchestrator.py   # Orchestrator (ReAct loop)
│       ├── knowledge_agent.py  # KnowledgeAgent (translate/rag/lookup)
│       ├── teaching_agent.py   # TeachingAgent (learning path)
│       └── quality_agent.py    # QualityAgent (review/self_judge)
├── translate_service.py      # PaiwanTranslator (RAG + FAISS)
└── strategy_ablation.py      # v1 消融腳本（舊版）
```

## Benchmark v2
```
benchmark/
├── lexicon_v2.json            # 24 詞詞彙表（corpus-verified）
├── benchmark_v2.json          # 20 題測試集
├── ground_truth.json          # Ground Truth + corpus evidence
├── run_benchmark_v2.py        # v2 實驗腳本
├── corpus_audit_report.md     # 語料庫審計報告
├── error_analysis.md          # 錯題分析
├── case_study.md              # 案例研究
├── agent_advantage_analysis.md # 優勢分析（ReAct vs multi-agent）
├── multi_agent_traces.json    # 6 題完整 trace
├── ablation_plot_A.png        # A線圖表（ReAct-Orchestrated）
├── ablation_plot.png          # 原始圖表（Multi-Agent 標籤）
├── ablation_summary.json      # 機器可讀摘要
├── run_trace.py               # Trace 捕獲腳本
├── gen_plot_A.py              # A線圖表生成
└── gen_plot.py                # 原始圖表生成
```

## 實驗結果
```
experiment_results/
├── ablation_20260604_201340.json  # v1 第 1 輪
├── ablation_20260604_202123.json  # v1 第 2 輪
├── ablation_20260604_202731.json  # v1 第 3 輪
├── ablation_20260604_203658.json  # v1 第 4 輪
└── ablation_v2_20260604_224233.json  # v2 最終結果 ★
```

## 數據源
```
data/
├── merged_corpus.json         # 主語料庫（5,923 筆）
├── klokah_paiwan_corpus.json  # klokah 教材語料
├── expanded_corpus.json       # 擴充語料
├── prompt_vocab.json          # 官方翻譯映射（已修正 tjina/tama）
├── local_faiss.index          # FAISS 索引
└── rag_local_metadata.json    # RAG 索引元數據
```

---

# 十一、Poster 可用素材

## 標題
LEAF: A ReAct-Orchestrated Framework for Endangered Language Learning

## 副標題
Iterative Reasoning + Adaptive Retrieval for Low-Resource Language Translation

## 核心數據
- Single-Shot RAG: **70%** → ReAct Orchestrator: **85%** (+15pp)
- 救回 3/6 RAG 失敗案例
- 語料：5,923 筆清洗後排灣語語料

## 三個亮點
1. **Query Reformulation**: translate 失敗後自動改用中英雙語 rag_search
2. **Reverse Verification**: 先搜尋候選詞再做反向翻譯驗證（p2c）
3. **Corpus-Verified Evaluation**: 28 variants 全部經語料庫驗證

## 圖表
- `benchmark/ablation_plot_A.png`（三面板：Accuracy / Error Breakdown / Match Type）

## Case Study
- q14 房子：translate(空) → rag_search(中英query) → umaq
- q19 名字：translate(空) → rag_search(同義詞) → ngadan
