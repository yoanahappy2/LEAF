# LEAF Benchmark v2.0 — Multi-Agent 優勢分析

## 核心發現

**三個 MA-only 成功案例（q14/q16/q19）的優勢全部來自 Orchestrator 的 ReAct 循環，不涉及多 Agent 協作。**

優勢本質：**多次檢索 > 單次檢索**，不是多 Agent 協作。

---

## 逐案例分析

---

### Case 1: q14「房子」→ umaq

**1. 問題**
- Input: 「房子的排灣語是什麼？」
- Target: 房子

**2. Ground Truth**
- accepted_answers: ["umaq"]
- preferred: umaq
- 語料證據: vocabulary 條目 umaq=家/家屋

**3. RAG Only Trace**
- 調用: `translate("房子", direction="c2p")`
- 精確匹配索引: 「房子」不在索引中（語料庫只有「家」「家屋」）
- FAISS 語意搜尋: 返回不相關結果
- 輸出: `""`（空字串）
- 結果: ✗ score=0.0

**4. Multi-Agent Trace**
```
Turn 1: orchestrator → knowledge: translate("房子", c2p)
        knowledge → orchestrator: translation="" (空)
Turn 2: orchestrator → knowledge: rag_search("房子 house")
        knowledge → orchestrator: results=[...] (語料例句)
Turn 3: orchestrator → knowledge: rag_search("house")
        knowledge → orchestrator: results=[...] (更多語料)
最終: LLM 整合所有 rag_search 結果 → 輸出 "umaq"
```

**5. Tool Sequence**
```
translate(c2p) → rag_search("房子 house") → rag_search("house") → LLM合成
```

**6. Agent Sequence**
```
Orchestrator → KnowledgeAgent → Orchestrator → KnowledgeAgent → Orchestrator → KnowledgeAgent → Orchestrator
```
只涉及 KnowledgeAgent。TeachingAgent 和 QualityAgent 未參與。

**7. 為何 Multi-Agent 成功**
RAG Only 做一次精確匹配+FAISS 搜尋就結束了。Orchestrator 偵測到第一次 translate 返回空結果後，改用不同 query（加英文 "house"、直接用英文）重新 rag_search，LLM 最終從語料例句中提取出 umaq。

**8. 是否涉及多 Agent 協作**
**否。** 只有 Orchestrator + KnowledgeAgent。

**9. 是否只是額外一次檢索**
**是。** 本質是 translate 失敗後用 rag_search 補了 2 次檢索。

**10. Poster Case Study Summary**

> **Case: 房子 → umaq**
> 
> RAG 精確匹配找不到「房子」（語料庫登記為「家」）。Multi-Agent 的 Orchestrator 偵測到翻譯失敗，自動切換策略用英文關鍵詞 "house" 做 rag_search，最終從語料例句中提取正確答案 umaq。
> 
> **優勢來源**: ReAct 循環的多步重試，不是多 Agent 協作。

---

### Case 2: q16「人」→ caucau

**1. 問題**
- Input: 「人的排灣語是什麼？」
- Target: 人

**2. Ground Truth**
- accepted_answers: ["caucau"]
- preferred: caucau
- 語料證據: custom 條目 caucau（釋義缺失但 expanded_corpus 大量出現）

**3. RAG Only Trace**
- 調用: `translate("人", direction="c2p")`
- 精確匹配索引: 「人」匹配到結果但翻譯為空
- 輸出: `""`（空字串）
- 結果: ✗ score=0.0

**4. Multi-Agent Trace**
```
Turn 1: orchestrator → knowledge: rag_search("人")
        knowledge → orchestrator: results=[...] (含 caucau 的例句)
Turn 2: orchestrator → knowledge: translate("caucau", p2c)  ← 反向驗證
        knowledge → orchestrator: translation="" (p2c 也查不到)
Turn 3: orchestrator → knowledge: translate("人", c2p)
        knowledge → orchestrator: translation="" (仍然空)
最終: LLM 從 rag_search 的例句中直接識別出 caucau
```

**5. Tool Sequence**
```
rag_search("人") → translate("caucau", p2c) → translate("人", c2p) → LLM合成
```

**6. Agent Sequence**
```
Orchestrator → KnowledgeAgent × 3
```
只涉及 KnowledgeAgent。

**7. 為何 Multi-Agent 成功**
RAG 的 translate 精確匹配失敗（「人」這個單字太短，語料庫登記不完整）。Orchestrator 先用 rag_search 做語意搜尋找到含 caucau 的例句，甚至做了反向驗證（caucau → 中文），最後 LLM 從例句中正確提取 caucau。

**8. 是否涉及多 Agent 協作**
**否。** 只有 Orchestrator + KnowledgeAgent。

**9. 是否只是額外一次檢索**
**基本上是。** 但這裡有個有趣的步驟：反向驗證（p2c 方向），這是單純「多檢索幾次」做不到的——需要 Orchestrator 有「先搜尋→反向驗證→再搜尋」的推理邏輯。

**10. Poster Case Study Summary**

> **Case: 人 → caucau**
> 
> RAG 的精確匹配找不到「人」（語料釋義缺失）。Multi-Agent 先用 rag_search 語意搜尋找到例句，再做反向翻譯驗證（caucau → 中文），最終 LLM 從語料例句中正確識別 caucau。
> 
> **優勢來源**: ReAct 的搜尋→驗證→推理鏈，涉及多步工具調用策略。

---

### Case 3: q19「名字」→ ngadan

**1. 問題**
- Input: 「名字用排灣語怎麼說？」
- Target: 名字

**2. Ground Truth**
- accepted_answers: ["ngadan"]
- preferred: ngadan
- 語料證據: custom 條目 ngadan=名字

**3. RAG Only Trace**
- 調用: `translate("名字", direction="c2p")`
- 精確匹配索引: 「名字」匹配不到直接翻譯
- 輸出: `""`（空字串）
- 結果: ✗ score=0.0

**4. Multi-Agent Trace**
```
Turn 1: orchestrator → knowledge: translate("名字", c2p)
        knowledge → orchestrator: translation="" (空)
Turn 2: orchestrator → knowledge: rag_search("名字 name")
        knowledge → orchestrator: results=[...] (含名字相關例句)
Turn 3: orchestrator → knowledge: rag_search("個人名稱 personal name")
        knowledge → orchestrator: results=[...] (含 ngadan 的例句)
最終: LLM 從 rag_search 結果中提取 ngadan
```

**5. Tool Sequence**
```
translate(c2p) → rag_search("名字 name") → rag_search("個人名稱 personal name") → LLM合成
```

**6. Agent Sequence**
```
Orchestrator → KnowledgeAgent × 3
```
只涉及 KnowledgeAgent。

**7. 為何 Multi-Agent 成功**
translate 精確匹配失敗後，Orchestrator 用中英雙語 query 做 rag_search（"名字 name"、"個人名稱 personal name"），擴大了搜尋覆蓋範圍，LLM 從語料例句中正確提取 ngadan。

**8. 是否涉及多 Agent 協作**
**否。** 只有 Orchestrator + KnowledgeAgent。

**9. 是否只是額外一次檢索**
**是。** 但 query 策略有變化（加英文、換同義詞），這是 Orchestrator 的 LLM 推理結果。

**10. Poster Case Study Summary**

> **Case: 名字 → ngadan**
> 
> RAG 精確匹配找不到「名字」。Multi-Agent 在 translate 失敗後，用中英雙語 query 做兩次 rag_search（"名字 name"、"個人名稱 personal name"），擴大語意覆蓋範圍，最終從語料中提取 ngadan。
> 
> **優勢來源**: LLM 驅動的 query 改寫 + 多次檢索重試。

---

## 總結

### 優勢本質

| Case | 優勢來源 | 多 Agent? | 本質 |
|------|---------|----------|------|
| q14 房子 | 多次 rag_search + 英文 query | 否 | 多步檢索 |
| q16 人 | rag_search + 反向驗證(p2c) | 否 | 多步檢索 + 推理 |
| q19 名字 | 中英雙語 rag_search | 否 | 多步檢索 + query 改寫 |

### 核心結論

**LEAF Benchmark v2.0 中 Multi-Agent 的 +15pp 優勢（70% → 85%）主要來自 Orchestrator 的 ReAct 循環，而非多 Agent 協作。**

具體來說：
1. **Orchestrator 的價值**：偵測失敗 → 改變策略（換工具/換 query）→ 重試
2. **KnowledgeAgent 的價值**：提供 translate 和 rag_search 兩種檢索方式
3. **TeachingAgent / QualityAgent 未參與**：因為 benchmark 只有翻譯題，不涉及學習路徑或品質審核

### 對 Poster 的啟示

**誠實的做法**：
- 標題不要寫「Multi-Agent 協作提升翻譯準確率」
- 改寫「ReAct 循環 + 工具鏈提升低資源語言翻譯準確率」
- 或「Orchestrator 驅動的多步推理彌補 RAG 檢索不足」

**如果要展示多 Agent 協作的價值**：
- 需要設計需要 Teaching + Quality 參與的題目（如：學習路徑推薦、翻譯品質審核、錯誤修正）
- 當前的 benchmark 只有翻譯題，只測到 Orchestrator + KnowledgeAgent
