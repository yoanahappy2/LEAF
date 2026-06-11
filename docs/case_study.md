# LEAF v2.0 — Multi-Agent Case Study

## Multi-Agent 獨有成功案例（RAG 失敗但 Multi-Agent 成功）

這三題展示了 Multi-Agent 架構的核心價值：
當 RAG 第一次檢索取不到結果時，Orchestrator 的 ReAct 循環能做多步推理補救。

---

## Case q14: '房子' -> umaq

| 模型 | 結果 | 回覆 |
|------|------|------|
| RAG Only | X | `` |
| Single Agent | X | `` |
| **Multi-Agent** | **O** | `
房子的排灣語是 **umaq**。` |

### 救回原因

RAG 精確匹配索引找不到「房子」的直接翻譯。
Multi-Agent 的 Orchestrator 偵測到第一次翻譯失敗後，啟動 ReAct 循環：
1. **Turn 1**: translate(房子) -> 返回空/錯誤
2. **Turn 2**: rag_search(房子) -> 語料庫語意搜尋
3. **Turn 3**: 整合結果 -> 輸出正確答案 `umaq`

這是**單步 RAG 做不到的**——它只能做一次檢索。

---

## Case q16: '人' -> caucau

| 模型 | 結果 | 回覆 |
|------|------|------|
| RAG Only | X | `` |
| Single Agent | X | `ca` |
| **Multi-Agent** | **O** | `
人的排灣語是「caucau」。這個詞在句子中經常出現，例如「我們有10個人」就是「malje tapuluq amen, izua anan a kakaiz` |

### 救回原因

RAG 精確匹配索引找不到「人」的直接翻譯。
Multi-Agent 的 Orchestrator 偵測到第一次翻譯失敗後，啟動 ReAct 循環：
1. **Turn 1**: translate(人) -> 返回空/錯誤
2. **Turn 2**: rag_search(人) -> 語料庫語意搜尋
3. **Turn 3**: 整合結果 -> 輸出正確答案 `caucau`

這是**單步 RAG 做不到的**——它只能做一次檢索。

---

## Case q19: '名字' -> ngadan

| 模型 | 結果 | 回覆 |
|------|------|------|
| RAG Only | X | `` |
| Single Agent | X | `` |
| **Multi-Agent** | **O** | `
名字的排灣語是 **ngadan**。

例如：
- 我的名字是 savan：**ti savan a ku ngadan**
- 我叫 sakinu：**t` |

### 救回原因

RAG 精確匹配索引找不到「名字」的直接翻譯。
Multi-Agent 的 Orchestrator 偵測到第一次翻譯失敗後，啟動 ReAct 循環：
1. **Turn 1**: translate(名字) -> 返回空/錯誤
2. **Turn 2**: rag_search(名字) -> 語料庫語意搜尋
3. **Turn 3**: 整合結果 -> 輸出正確答案 `ngadan`

這是**單步 RAG 做不到的**——它只能做一次檢索。

---

## 反例：Multi-Agent 也救不回的題目

### q11 '母親' -> expected: kina
- MA reply: `
母親的排灣語是「tjina」。`
- 失敗原因: RAG 知識庫返回了錯誤答案（tjina/tama），Knowledge Agent 無法自行修正知識庫的髒數據

### q12 '父親' -> expected: kama
- MA reply: `
父親的排灣語是「tama」。`
- 失敗原因: RAG 知識庫返回了錯誤答案（tjina/tama），Knowledge Agent 無法自行修正知識庫的髒數據

### q18 '山' -> expected: gadu
- MA reply: `
山的排灣語是 **qungiljaw**。`
- 失敗原因: 所有模型都 hallucinate（qungiljaw 不是「山」），RAG 語意檢索返回了錯誤的詞
