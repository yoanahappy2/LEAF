#!/usr/bin/env python3
"""
test_quality_v2.py — QualityAgent v2 Reverse Verification 測試

驗證目標：
1. tjina (不是排灣語「母親」) → FAIL
2. tama (不是排灣語「父親」) → FAIL
3. masalu (不是「你好」，是「謝謝」) → FAIL
4. kina (正確的「母親」) → PASS
5. kama (正確的「父親」) → PASS
6. djavadjavai (正確的「你好」) → PASS
7. masalu (正確的「謝謝」) → PASS
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "agent_framework"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from openai import OpenAI
from agents.quality_agent import QualityAgent
from core.message import AgentMessage

client = OpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

agent = QualityAgent(
    client=client,
    project_root=Path(__file__).parent,
)

# 測試用例
test_cases = [
    # (original, translation, direction, expected_pass, description)
    ("母親", "tjina", "c2p", False, "tjina 不是排灣語「母親」（應為 kina）"),
    ("父親", "tama", "c2p", False, "tama 不是排灣語「父親」（應為 kama）"),
    ("你好", "masalu", "c2p", False, "masalu 是「謝謝」不是「你好」"),
    ("母親", "kina", "c2p", True, "kina 是正確的「母親」"),
    ("父親", "kama", "c2p", True, "kama 是正確的「父親」"),
    ("你好", "djavadjavai", "c2p", True, "djavadjavai 是正確的「你好」"),
    ("謝謝", "masalu", "c2p", True, "masalu 是正確的「謝謝」"),
    ("水", "zaljum", "c2p", True, "zaljum 是正確的「水」"),
    ("吃", "keman", "c2p", True, "keman 是正確的「吃」"),
    ("太陽", "qadaw", "c2p", True, "qadaw 是正確的「太陽」"),
]

print("=" * 70)
print("  🔍 QualityAgent v2 — Reverse Verification 測試")
print("=" * 70)

results = []
correct = 0
total = len(test_cases)

for original, translation, direction, expected_pass, description in test_cases:
    msg = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="quality",
        task="review_translation",
        params={"original": original, "translation": translation, "direction": direction},
    )

    resp = agent.handle_message(msg)
    data = resp.payload.get("data", {})
    passed = data.get("passed", None)
    score = data.get("score", None)
    method = data.get("method", "")
    rv = data.get("reverse_verification", {})
    feedback = data.get("feedback", "")

    match = passed == expected_pass
    correct += 1 if match else 0

    status = "✅" if match else "❌"
    print(f"\n{status} {original} → {translation}")
    print(f"   預期: {'PASS' if expected_pass else 'FAIL'} | 實際: {'PASS' if passed else 'FAIL'} | score={score} | method={method}")
    print(f"   Reverse: {rv.get('reverse_translation', '')} | rv_score={rv.get('match_score', '')} | rv_method={rv.get('method', '')}")
    if feedback:
        print(f"   Feedback: {feedback}")
    print(f"   說明: {description}")

    results.append({
        "original": original,
        "translation": translation,
        "expected_pass": expected_pass,
        "actual_pass": passed,
        "score": score,
        "method": method,
        "reverse_verification": rv,
        "match": match,
        "description": description,
    })

print("\n" + "=" * 70)
print(f"  結果: {correct}/{total} 正確 ({correct/total*100:.1f}%)")
print("=" * 70)

# 保存結果
result_file = Path(__file__).parent / "experiment_results" / "quality_v2_test.json"
result_file.parent.mkdir(parents=True, exist_ok=True)
with open(result_file, "w", encoding="utf-8") as f:
    json.dump({"results": results, "summary": {"correct": correct, "total": total, "accuracy": correct/total}}, f, ensure_ascii=False, indent=2)
print(f"\n結果已保存: {result_file}")
