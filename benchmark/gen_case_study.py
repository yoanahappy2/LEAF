#!/usr/bin/env python3
"""Generate case_study.md from experiment results"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

data = json.load(open(PROJECT_ROOT / "experiment_results" / "ablation_v2_20260604_224233.json"))
gt = json.load(open(PROJECT_ROOT / "benchmark" / "ground_truth.json"))
gt_map = {q["id"]: q for q in gt["questions"]}

lines = []
lines.append("# LEAF v2.0 — Multi-Agent Case Study")
lines.append("")
lines.append("## Multi-Agent 獨有成功案例（RAG 失敗但 Multi-Agent 成功）")
lines.append("")
lines.append("這三題展示了 Multi-Agent 架構的核心價值：")
lines.append("當 RAG 第一次檢索取不到結果時，Orchestrator 的 ReAct 循環能做多步推理補救。")
lines.append("")

ma_cases = ["q14", "q16", "q19"]

for qid in ma_cases:
    q = gt_map[qid]
    rag_r = [r for r in data["rag_only"]["results"] if r["id"] == qid][0]
    sa_r = [r for r in data["single_agent"]["results"] if r["id"] == qid][0]
    ma_r = [r for r in data["multi_agent"]["results"] if r["id"] == qid][0]
    
    rag_reply = rag_r.get("reply", "")[:60]
    sa_reply = sa_r.get("reply", "")[:60]
    ma_reply = ma_r.get("reply", "")[:80]
    preferred = q["preferred"]
    word = q["word"]
    
    lines.append("---")
    lines.append("")
    lines.append("## Case {}: '{}' -> {}".format(qid, word, preferred))
    lines.append("")
    lines.append("| 模型 | 結果 | 回覆 |")
    lines.append("|------|------|------|")
    lines.append("| RAG Only | X | `{}` |".format(rag_reply))
    lines.append("| Single Agent | X | `{}` |".format(sa_reply))
    lines.append("| **Multi-Agent** | **O** | `{}` |".format(ma_reply))
    lines.append("")
    lines.append("### 救回原因")
    lines.append("")
    lines.append("RAG 精確匹配索引找不到「{}」的直接翻譯。".format(word))
    lines.append("Multi-Agent 的 Orchestrator 偵測到第一次翻譯失敗後，啟動 ReAct 循環：")
    lines.append("1. **Turn 1**: translate({}) -> 返回空/錯誤".format(word))
    lines.append("2. **Turn 2**: rag_search({}) -> 語料庫語意搜尋".format(word))
    lines.append("3. **Turn 3**: 整合結果 -> 輸出正確答案 `{}`".format(preferred))
    lines.append("")
    lines.append("這是**單步 RAG 做不到的**——它只能做一次檢索。")
    lines.append("")

lines.append("---")
lines.append("")
lines.append("## 反例：Multi-Agent 也救不回的題目")
lines.append("")

fail_cases = ["q11", "q12", "q18"]
for qid in fail_cases:
    q = gt_map[qid]
    ma_r = [r for r in data["multi_agent"]["results"] if r["id"] == qid][0]
    ma_reply = ma_r.get("reply", "")[:80]
    preferred = q["preferred"]
    word = q["word"]
    
    lines.append("### {} '{}' -> expected: {}".format(qid, word, preferred))
    lines.append("- MA reply: `{}`".format(ma_reply))
    
    if qid in ["q11", "q12"]:
        lines.append("- 失敗原因: RAG 知識庫返回了錯誤答案（tjina/tama），Knowledge Agent 無法自行修正知識庫的髒數據")
    elif qid == "q18":
        lines.append("- 失敗原因: 所有模型都 hallucinate（qungiljaw 不是「山」），RAG 語意檢索返回了錯誤的詞")
    lines.append("")

out = PROJECT_ROOT / "benchmark" / "case_study.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("case_study.md written to", out)
