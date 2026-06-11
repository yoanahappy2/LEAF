"""
agent_ablation.py — Multi-Agent 消融實驗

證明每個 Agent 的存在價值：
1. Full System（4 Agent）
2. Single Agent（只有 Orchestrator，不分派）
3. No Quality（關掉 Quality Agent）
4. No Teaching（關掉 Teaching Agent）

對比：翻譯準確率、回覆品質、Token 消耗、響應時間

作者: 地陪
日期: 2026-05-30
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ablation")

client = OpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

sys.path.insert(0, str(Path(__file__).parent / "agent_framework"))
from agents.orchestrator import OrchestratorAgent
from core.rate_limiter import get_api_guard

# ============================================
# 測試集 — 涵蓋翻譯、教學、品質三類
# ============================================

TEST_CASES = [
    # 翻譯類（需要 Knowledge Agent）
    {"id": "T01", "input": "謝謝的排灣語怎麼說", "category": "translate", "expected_keyword": "masalu"},
    {"id": "T02", "input": "你好嗎排灣語", "category": "translate", "expected_keyword": "tarivak"},
    {"id": "T03", "input": "水用排灣語怎麼說", "category": "translate", "expected_keyword": "zaljum"},
    {"id": "T04", "input": "再見排灣語", "category": "translate", "expected_keyword": "pacunan"},
    {"id": "T05", "input": "朋友排灣語", "category": "translate", "expected_keyword": "drangi"},

    # 知識查詢類（需要 Knowledge Agent）
    {"id": "K01", "input": "masalu 是什麼意思", "category": "knowledge", "expected_keyword": "謝謝"},
    {"id": "K02", "input": "排灣語的家族稱謂有哪些", "category": "knowledge", "expected_keyword": "kina"},
    {"id": "K03", "input": "排灣語的前綴有什麼", "category": "knowledge", "expected_keyword": "前綴"},

    # 教學類（需要 Teaching Agent）
    {"id": "L01", "input": "我想從零開始學排灣語", "category": "teaching", "expected_keyword": ""},
    {"id": "L02", "input": "幫我出一道排灣語測驗題", "category": "teaching", "expected_keyword": ""},
    {"id": "L03", "input": "推薦我下一個排灣語詞彙", "category": "teaching", "expected_keyword": ""},

    # 品質相關（需要 Quality Agent）
    {"id": "Q01", "input": "審核翻譯：原句「你好」→ 翻譯「djavadjavay」", "category": "quality", "expected_keyword": ""},
    {"id": "Q02", "input": "審核翻譯：原句「謝謝」→ 翻譯「masalu」", "category": "quality", "expected_keyword": ""},

    # 混合/對話
    {"id": "M01", "input": "你好", "category": "chat", "expected_keyword": ""},
    {"id": "M02", "input": "排灣族有什麼文化", "category": "chat", "expected_keyword": ""},
]


# ============================================
# 消融配置
# ============================================

ABLATION_CONFIGS = [
    {
        "name": "Full_System",
        "label": "4 Agent 全開（Control）",
        "setup": "full",
    },
    {
        "name": "No_Quality",
        "label": "關掉 Quality Agent",
        "setup": "no_quality",
    },
    {
        "name": "No_Teaching",
        "label": "關掉 Teaching Agent",
        "setup": "no_teaching",
    },
    {
        "name": "Single_Agent",
        "label": "單 Agent（Orchestrator 獨立處理）",
        "setup": "single",
    },
]


def setup_orchestrator(config_name: str) -> OrchestratorAgent:
    """根據消融配置建立 Orchestrator"""
    api_guard = get_api_guard()
    orch = OrchestratorAgent(
        client=client,
        api_guard=api_guard,
        project_root=Path(__file__).parent,
    )

    config = next(c for c in ABLATION_CONFIGS if c["name"] == config_name)
    setup = config["setup"]

    if setup == "full":
        orch._ensure_agents()
    elif setup == "no_quality":
        from agents.knowledge_agent import KnowledgeAgent
        from agents.teaching_agent import TeachingAgent
        knowledge = KnowledgeAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        teaching = TeachingAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        orch.register_agents(knowledge=knowledge, teaching=teaching, quality=None)
    elif setup == "no_teaching":
        from agents.knowledge_agent import KnowledgeAgent
        from agents.quality_agent import QualityAgent
        knowledge = KnowledgeAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        quality = QualityAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        orch.register_agents(knowledge=knowledge, teaching=None, quality=quality)
    elif setup == "single":
        # 不註冊任何 Agent
        orch._agents_registered = True
        # Override chat to bypass agent dispatch
        orch._single_mode = True

    return orch


def single_agent_chat(orch, user_input: str) -> str:
    """Single Agent 模式：Orchestrator 直接用 LLM 回覆，不經過其他 Agent"""
    system_prompt = (
        "你是排灣族語教學助手。請直接回答用戶的問題。\n"
        "如果涉及排灣語翻譯，盡你所能回答。如果你不確定，請說明。\n"
        "簡潔回覆，不超過 5 句話。\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    result = orch.call_llm(
        messages=messages,
        tools=None,
        temperature=0.4,
        max_tokens=300,
        model="glm-4.5-air",
    )
    return result["message"].content or ""


def score_reply(test_case: dict, reply: str) -> dict:
    """評分回覆品質"""
    scores = {}
    issues = []

    # 1. 有回覆
    scores["has_reply"] = len(reply.strip()) > 0
    if not scores["has_reply"]:
        issues.append("無回覆")

    # 2. 回覆長度合理（10-500字）
    scores["length_ok"] = 10 <= len(reply) <= 500
    if len(reply) < 10:
        issues.append(f"太短({len(reply)}字)")
    elif len(reply) > 500:
        issues.append(f"太長({len(reply)}字)")

    # 3. 關鍵詞匹配
    expected = test_case.get("expected_keyword", "")
    if expected:
        scores["keyword_match"] = expected in reply
        if not scores["keyword_match"]:
            issues.append(f"缺少關鍵詞: {expected}")
    else:
        scores["keyword_match"] = True

    # 4. 沒有工具調用洩漏（rag_search / translate 等格式）
    tool_leak = any(marker in reply for marker in ["rag_search", "translate(", "query=", ""])
    scores["no_tool_leak"] = not tool_leak
    if tool_leak:
        issues.append("工具調用洩漏")

    # 5. 沒有明顯錯誤（包含排灣語但標記⚠️，或不確定標記）
    has_uncertainty = "不確定" in reply or "⚠️" in reply or "[不確定]" in reply
    # 如果是翻譯/知識類且有 expected keyword，不確定 = 扣分
    if test_case["category"] in ("translate", "knowledge") and expected and has_uncertainty:
        scores["confidence"] = False
        issues.append("標記不確定")
    else:
        scores["confidence"] = True

    total = sum(1 for v in scores.values() if v)
    max_score = len(scores)
    return {"scores": scores, "total": total, "max": max_score, "issues": issues}


def run_ablation():
    """執行完整消融實驗"""
    print("=" * 70)
    print("  🧪 Multi-Agent 消融實驗")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  測試集: {len(TEST_CASES)} 條")
    print(f"  消融配置: {len(ABLATION_CONFIGS)} 組")
    print("=" * 70)

    all_results = {}

    for config in ABLATION_CONFIGS:
        name = config["name"]
        label = config["label"]

        print(f"\n{'─' * 60}")
        print(f"  📊 {name}: {label}")
        print(f"{'─' * 60}")

        orch = setup_orchestrator(name)
        guard = orch.api_guard
        tokens_before = guard.get_total_tokens_used()

        config_results = {
            "label": label,
            "interactions": [],
            "tokens_used": 0,
        }

        pass_count = 0
        fail_count = 0

        for i, tc in enumerate(TEST_CASES):
            tid = tc["id"]
            user_input = tc["input"]
            print(f"  [{i+1}/{len(TEST_CASES)}] {tid} 「{user_input[:30]}」", end=" ... ", flush=True)

            start = time.time()
            try:
                if getattr(orch, '_single_mode', False):
                    reply = single_agent_chat(orch, user_input)
                else:
                    reply = orch.chat(user_input, user_id="ablation_test")
                elapsed = (time.time() - start) * 1000

                scoring = score_reply(tc, reply)
                passed = scoring["total"] >= scoring["max"] - 1

                if passed:
                    pass_count += 1
                    status = "✅"
                else:
                    fail_count += 1
                    status = "❌"

                print(f"{status} ({scoring['total']}/{scoring['max']}) [{elapsed:.0f}ms]")

                if scoring["issues"]:
                    for issue in scoring["issues"]:
                        print(f"         {issue}")

                config_results["interactions"].append({
                    "id": tid,
                    "input": user_input,
                    "category": tc["category"],
                    "reply": reply[:500],
                    "elapsed_ms": round(elapsed),
                    "scores": scoring["scores"],
                    "score_total": scoring["total"],
                    "score_max": scoring["max"],
                    "passed": passed,
                    "issues": scoring["issues"],
                })

            except Exception as e:
                fail_count += 1
                elapsed = (time.time() - start) * 1000
                print(f"💥 {e}")
                config_results["interactions"].append({
                    "id": tid,
                    "input": user_input,
                    "category": tc["category"],
                    "error": str(e),
                    "elapsed_ms": round(elapsed),
                    "passed": False,
                })

            time.sleep(0.5)

        tokens_after = guard.get_total_tokens_used()
        config_results["tokens_used"] = tokens_after - tokens_before
        config_results["pass_count"] = pass_count
        config_results["fail_count"] = fail_count
        config_results["pass_rate"] = pass_count / (pass_count + fail_count) * 100 if (pass_count + fail_count) > 0 else 0

        print(f"\n  📊 {name}: {pass_count}/{pass_count+fail_count} 通過 ({config_results['pass_rate']:.0f}%) | {config_results['tokens_used']:,} tokens")

        all_results[name] = config_results

    # ============================================
    # 摘要對比表
    # ============================================
    print(f"\n{'═' * 70}")
    print("  📊 消融實驗結果對比")
    print(f"{'═' * 70}\n")

    print(f"  {'配置':<25} {'通過率':>8} {'Token':>10} {'品質分':>8}")
    print(f"  {'─' * 55}")

    for name, data in all_results.items():
        label = data["label"][:22]
        rate = f"{data['pass_rate']:.0f}%"
        tokens = f"{data['tokens_used']:,}"
        # 加權品質分
        all_scores = [i["score_total"] / i["score_max"] for i in data["interactions"] if "score_total" in i]
        avg_quality = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        quality = f"{avg_quality:.0f}%"
        print(f"  {label:<25} {rate:>8} {tokens:>10} {quality:>8}")

    # 按類別統計
    print(f"\n  按類別通過率:")
    categories = sorted(set(tc["category"] for tc in TEST_CASES))
    for cat in categories:
        cat_tests = [tc for tc in TEST_CASES if tc["category"] == cat]
        print(f"    {cat}: ", end="")
        for name, data in all_results.items():
            cat_results = [i for i in data["interactions"] if i.get("category") == cat]
            cat_pass = sum(1 for i in cat_results if i.get("passed"))
            cat_total = len(cat_results)
            rate = cat_pass / cat_total * 100 if cat_total else 0
            print(f"{name}={rate:.0f}%  ", end="")
        print()

    # 保存結果
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": len(TEST_CASES),
        "configs": {name: {
            "label": data["label"],
            "pass_rate": data["pass_rate"],
            "pass_count": data["pass_count"],
            "fail_count": data["fail_count"],
            "tokens_used": data["tokens_used"],
        } for name, data in all_results.items()},
        "details": all_results,
    }

    report_path = Path(__file__).parent / "results" / "agent_ablation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 報告: {report_path}")

    return report


if __name__ == "__main__":
    run_ablation()
