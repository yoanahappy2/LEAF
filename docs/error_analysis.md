# LEAF Benchmark v2.0 — Error Analysis

## 總覽

| 配置 | 準確率 | avg_score | 錯題數 |
|------|--------|-----------|--------|
| llm_direct | 0% | 0.00 | 20/20 |
| rag_only | 70% | 0.68 | 6/20 |
| single_agent | 70% | 0.68 | 6/20 |
| multi_agent | 85% | 0.83 | 3/20 |

## 逐題結果

| ID | 中文 | 答案 | LLM | RAG | Single | Multi | 備註 |
|----|------|------|-----|-----|--------|-------|------|
| q01 | 你好 | djavadjavai | ✗ | ✓ | ✓ | ✓ |  |
| q02 | 謝謝 | masalu | ✗ | ✓ | ✓ | ✓ |  |
| q03 | 水 | zaljum | ✗ | ✓ | ✓ | ✓ |  |
| q04 | 吃 | keman | ✗ | ✓ | ✓ | ✓ |  |
| q05 | 太陽 | qadaw | ✗ | ✓ | ✓ | ✓ |  |
| q06 | 月亮 | qiljas | ✗ | ✓ | ✓ | ✓ |  |
| q07 | 手 | lima | ✗ | ✓ | ✓ | ✓ |  |
| q08 | 眼睛 | maca | ✗ | ✓ | ✓ | ✓ |  |
| q09 | 火 | sapuy | ✗ | ✓ | ✓ | ✓ |  |
| q10 | 星星 | vitjuqan | ✗ | ✓ | ✓ | ✓ |  |
| q11 | 母親 | kina | ✗ | ✗ | ✗ | ✗ | 全部失敗 |
| q12 | 父親 | kama | ✗ | ✗ | ✗ | ✗ | 全部失敗 |
| q13 | 孩子 | aljak | ✗ | ✓ | ✓ | ✓ |  |
| q14 | 房子 | umaq | ✗ | ✗ | ✗ | ✓ | **MA救回** |
| q15 | 道路 | djalan | ✗ | ✓ | ✓ | ✓ |  |
| q16 | 人 | caucau | ✗ | ✗ | ✗ | ✓ | **MA救回** |
| q17 | 朋友 | drangi | ✗ | ✓ | ✓ | ✓ |  |
| q18 | 山 | gadu | ✗ | ✗ | ✗ | ✗ | 全部失敗 |
| q19 | 名字 | ngadan | ✗ | ✗ | ✗ | ✓ | **MA救回** |
| q20 | 狗 | vatu | ✗ | ✓ | ✓ | ✓ |  |

---

## RAG / Single Agent 錯題詳情

### q11 「母親」→ expected: ['kina']
- 正確答案: ['kina']
- rag_only: ✗ reply="tjina" score=0.0
- single_agent: ✗ reply="tjina" score=0.0
- multi_agent: ✗ reply="
母親的排灣語是「tjina」。" score=0.0

### q12 「父親」→ expected: ['kama']
- 正確答案: ['kama']
- rag_only: ✗ reply="tama" score=0.0
- single_agent: ✗ reply="tama" score=0.0
- multi_agent: ✗ reply="
父親的排灣語是「tama」。" score=0.0

### q14 「房子」→ expected: ['umaq']
- 正確答案: ['umaq']
- rag_only: ✗ reply="" score=0.0
- single_agent: ✗ reply="" score=0.0
- multi_agent: ✓ reply="
房子的排灣語是 **umaq**。" score=1.0

### q16 「人」→ expected: ['caucau']
- 正確答案: ['caucau']
- rag_only: ✗ reply="" score=0.0
- single_agent: ✗ reply="ca" score=0.0
- multi_agent: ✓ reply="
人的排灣語是「caucau」。這個詞在句子中經常出現，例如「我們有10個人」就是「malje tapuluq amen, izua anan a kakaizuavan」。" score=1.0

### q18 「山」→ expected: ['gadu']
- 正確答案: ['gadu']
- rag_only: ✗ reply="qungiljaw" score=0.0
- single_agent: ✗ reply="qungiljaw" score=0.0
- multi_agent: ✗ reply="
山的排灣語是 **qungiljaw**。" score=0.0

### q19 「名字」→ expected: ['ngadan']
- 正確答案: ['ngadan']
- rag_only: ✗ reply="" score=0.0
- single_agent: ✗ reply="" score=0.0
- multi_agent: ✓ reply="
名字的排灣語是 **ngadan**。

例如：
- 我的名字是 savan：**ti savan a ku ngadan**
- 我叫 sakinu：**ti sakinu a ku ngadan" score=1.0

---

## Multi-Agent 獨有成功案例

### q14 「房子」→ umaq
- RAG reply: ""
- Single reply: ""
- Multi reply: "
房子的排灣語是 **umaq**。"
- 救回原因: Multi-Agent 的 ReAct 循環允許多次工具調用（translate → rag_search → lookup），單次 RAG 查不到時能補救

### q16 「人」→ caucau
- RAG reply: ""
- Single reply: "ca"
- Multi reply: "
人的排灣語是「caucau」。這個詞在句子中經常出現，例如「我們有10個人」就是「malje tapuluq amen, izua anan a kakaizuavan」。"
- 救回原因: Multi-Agent 的 ReAct 循環允許多次工具調用（translate → rag_search → lookup），單次 RAG 查不到時能補救

### q19 「名字」→ ngadan
- RAG reply: ""
- Single reply: ""
- Multi reply: "
名字的排灣語是 **ngadan**。

例如：
- 我的名字是 savan：**ti savan a ku ngadan**
- 我叫 sakinu：**ti sakinu a ku ngadan"
- 救回原因: Multi-Agent 的 ReAct 循環允許多次工具調用（translate → rag_search → lookup），單次 RAG 查不到時能補救
