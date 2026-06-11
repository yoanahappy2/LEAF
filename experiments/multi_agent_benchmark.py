#!/usr/bin/env python3
"""
multi_agent_benchmark.py — Multi-Agent 50 Loops 完整跑分

實驗設計：
- 4 組消融配置（Full / No_Quality / No_Teaching / Single_Agent）
- 15 條測試集 × 3 次重複 = 每組 45 次交互
- 模型：glm-4.5-air（主力）
- 記錄：通過率、Token 消耗、響應時間、錯誤率、按類別統計

用法：
    python3 multi_agent_benchmark.py                  # 跑全部
    python3 multi_agent_benchmark.py --config Full    # 只跑某一組
    python3 multi_agent_benchmark.py --loops 10       # 快速測試
    python3 multi_agent_benchmark.py --model glm-4-flash

作者: 地陪
日期: 2026-06-03
"""

import sys
import json
import time
import logging
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("benchmark")

client = OpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

sys.path.insert(0, str(Path(__file__).parent / "agent_framework"))
from agents.orchestrator import OrchestratorAgent
from core.rate_limiter import get_api_guard

# ============================================
# 測試集 — 15 條，涵蓋 5 類
# ============================================

TEST_CASES = [
    # 翻譯類
    {"id": "T01", "input": "謝謝的排灣語怎麼說", "category": "translate", "expected_keyword": "masalu"},
    {"id": "T02", "input": "你好嗎排灣語", "category": "translate", "expected_keyword": "tarivak"},
    {"id": "T03", "input": "水用排灣語怎麼說", "category": "translate", "expected_keyword": "zaljum"},
    {"id": "T04", "input": "再見排灣語", "category": "translate", "expected_keyword": "pacunan"},
    {"id": "T05", "input": "朋友排灣語", "category": "translate", "expected_keyword": "drangi"},

    # 知識查詢
    {"id": "K01", "input": "masalu 是什麼意思", "category": "knowledge", "expected_keyword": "謝謝"},
    {"id": "K02", "input": "排灣語的家族稱謂有哪些", "category": "knowledge", "expected_keyword": "kina"},
    {"id": "K03", "input": "排灣語的前綴有什麼", "category": "knowledge", "expected_keyword": "前綴"},

    # 教學
    {"id": "L01", "input": "我想從零開始學排灣語", "category": "teaching", "expected_keyword": ""},
    {"id": "L02", "input": "幫我出一道排灣語測驗題", "category": "teaching", "expected_keyword": ""},
    {"id": "L03", "input": "推薦我下一個排灣語詞彙", "category": "teaching", "expected_keyword": ""},

    # 品質
    {"id": "Q01", "input": "審核翻譯：原句「你好」→ 翻譯「djavadjavay」", "category": "quality", "expected_keyword": ""},
    {"id": "Q02", "input": "審核翻譯：原句「謝謝」→ 翻譯「masalu」", "category": "quality", "expected_keyword": ""},

    # 混合
    {"id": "M01", "input": "你好", "category": "chat", "expected_keyword": ""},
    {"id": "M02", "input": "排灣族有什麼文化", "category": "chat", "expected_keyword": ""},
]

# ============================================
# 消融配置
# ============================================

ABLATION_CONFIGS = {
    "Full_System": {
        "label": "4 Agent 全開（Control）",
        "setup": "full",
    },
    "No_Quality": {
        "label": "關掉 Quality Agent",
        "setup": "no_quality",
    },
    "No_Teaching": {
        "label": "關掉 Teaching Agent",
        "setup": "no_teaching",
    },
    "Single_Agent": {
        "label": "單 Agent（Orchestrator 獨立）",
        "setup": "single",
    },
}


def setup_orchestrator(config_name: str, model: str = "glm-4.5-air") -> OrchestratorAgent:
    api_guard = get_api_guard()
    orch = OrchestratorAgent(
        client=client,
        api_guard=api_guard,
        project_root=Path(__file__).parent,
    )

    config = ABLATION_CONFIGS[config_name]
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
        orch._agents_registered = True
        orch._single_mode = True

    orch._model = model
    return orch


def single_agent_chat(orch, user_input: str, model: str = "glm-4.5-air") -> str:
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
        model=model,
    )
    return result["message"].content or ""


def score_reply(test_case: dict, reply: str) -> dict:
    scores = {}
    issues = []

    scores["has_reply"] = len(reply.strip()) > 0
    if not scores["has_reply"]:
        issues.append("無回覆")

    scores["length_ok"] = 10 <= len(reply) <= 500
    if len(reply) < 10:
        issues.append(f"太短({len(reply)}字)")
    elif len(reply) > 500:
        issues.append(f"太長({len(reply)}字)")

    expected = test_case.get("expected_keyword", "")
    if expected:
        scores["keyword_match"] = expected in reply
        if not scores["keyword_match"]:
            issues.append(f"缺少關鍵詞: {expected}")
    else:
        scores["keyword_match"] = True

    tool_leak = any(marker in reply for marker in ["rag_search", "translate(", "query="])
    scores["no_tool_leak"] = not tool_leak
    if tool_leak:
        issues.append("工具調用洩漏")

    has_uncertainty = "不確定" in reply or "⚠️" in reply or "[不確定]" in reply
    if test_case["category"] in ("translate", "knowledge") and expected and has_uncertainty:
        scores["confidence"] = False
        issues.append("標記不確定")
    else:
        scores["confidence"] = True

    total = sum(1 for v in scores.values() if v)
    max_score = len(scores)
    return {"scores": scores, "total": total, "max": max_score, "issues": issues}


def run_single_config(config_name: str, loops: int, model: str) -> dict:
    """跑一組消融配置，loops 次重複"""
    config = ABLATION_CONFIGS[config_name]
    label = config["label"]
    print(f"\n{'═' * 70}")
    print(f"  🧪 {config_name}: {label}")
    print(f"  Loops: {loops} | Model: {model} | 測試集: {len(TEST_CASES)} 條")
    print(f"{'═' * 70}")

    all_interactions = []
    total_pass = 0
    total_fail = 0
    total_error = 0
    all_latencies = []
    all_scores_pct = []

    for loop_idx in range(loops):
        print(f"\n  ── Loop {loop_idx + 1}/{loops} ──")

        # 每個 loop 重新建立 orchestrator
        orch = setup_orchestrator(config_name, model=model)
        guard = orch.api_guard

        for i, tc in enumerate(TEST_CASES):
            tid = f"{tc['id']}_L{loop_idx+1}"
            user_input = tc["input"]
            print(f"    [{i+1}/{len(TEST_CASES)}] {tid} 「{user_input[:25]}」", end=" ... ", flush=True)

            start = time.time()
            try:
                if getattr(orch, '_single_mode', False):
                    reply = single_agent_chat(orch, user_input, model=model)
                else:
                    reply = orch.chat(user_input, user_id=f"bench_{loop_idx}")
                elapsed = (time.time() - start) * 1000

                scoring = score_reply(tc, reply)
                passed = scoring["total"] >= scoring["max"] - 1

                if passed:
                    total_pass += 1
                    status = "✅"
                else:
                    total_fail += 1
                    status = "❌"

                score_pct = scoring["total"] / scoring["max"] * 100
                all_scores_pct.append(score_pct)
                all_latencies.append(elapsed)

                print(f"{status} ({scoring['total']}/{scoring['max']}) [{elapsed:.0f}ms]")

                all_interactions.append({
                    "loop": loop_idx + 1,
                    "id": tid,
                    "test_id": tc["id"],
                    "input": user_input,
                    "category": tc["category"],
                    "reply": reply[:300],
                    "elapsed_ms": round(elapsed),
                    "scores": scoring["scores"],
                    "score_total": scoring["total"],
                    "score_max": scoring["max"],
                    "score_pct": round(score_pct, 1),
                    "passed": passed,
                    "issues": scoring["issues"],
                })

            except Exception as e:
                total_error += 1
                elapsed = (time.time() - start) * 1000
                all_latencies.append(elapsed)
                print(f"💥 {e}")

                all_interactions.append({
                    "loop": loop_idx + 1,
                    "id": tid,
                    "test_id": tc["id"],
                    "input": user_input,
                    "category": tc["category"],
                    "error": str(e),
                    "elapsed_ms": round(elapsed),
                    "passed": False,
                })

            # Rate limit: 0.8s between requests
            time.sleep(0.8)

        # Brief pause between loops
        if loop_idx < loops - 1:
            print(f"    ⏳ Loop {loop_idx + 1} 完成，暫停 3 秒...")
            time.sleep(3)

    # ─── 統計 ───
    n_total = total_pass + total_fail + total_error
    pass_rate = total_pass / n_total * 100 if n_total else 0
    error_rate = total_error / n_total * 100 if n_total else 0

    # 按類別統計
    by_category = {}
    for cat in sorted(set(tc["category"] for tc in TEST_CASES)):
        cat_items = [it for it in all_interactions if it.get("category") == cat and "error" not in it]
        cat_pass = sum(1 for it in cat_items if it.get("passed"))
        cat_total = len(cat_items)
        cat_errors = sum(1 for it in all_interactions if it.get("category") == cat and "error" in it)
        cat_scores = [it["score_pct"] for it in cat_items]
        by_category[cat] = {
            "pass": cat_pass,
            "total": cat_total,
            "errors": cat_errors,
            "pass_rate": cat_pass / cat_total * 100 if cat_total else 0,
            "avg_score": round(statistics.mean(cat_scores), 1) if cat_scores else 0,
        }

    # Latency stats
    lat_p50 = statistics.median(all_latencies) if all_latencies else 0
    lat_p95 = sorted(all_latencies)[int(len(all_latencies)*0.95)] if len(all_latencies) >= 20 else max(all_latencies) if all_latencies else 0
    lat_mean = statistics.mean(all_latencies) if all_latencies else 0

    # Score distribution
    score_mean = statistics.mean(all_scores_pct) if all_scores_pct else 0
    score_std = statistics.stdev(all_scores_pct) if len(all_scores_pct) >= 2 else 0

    result = {
        "label": label,
        "config": config_name,
        "model": model,
        "loops": loops,
        "total_interactions": n_total,
        "pass": total_pass,
        "fail": total_fail,
        "errors": total_error,
        "pass_rate": round(pass_rate, 2),
        "error_rate": round(error_rate, 2),
        "avg_score": round(score_mean, 2),
        "score_std": round(score_std, 2),
        "latency_ms": {
            "mean": round(lat_mean),
            "p50": round(lat_p50),
            "p95": round(lat_p95),
        },
        "by_category": by_category,
        "interactions": all_interactions,
    }

    # Print summary
    print(f"\n  📊 {config_name} Summary:")
    print(f"     通過率: {pass_rate:.1f}% ({total_pass}/{n_total})")
    print(f"     錯誤率: {error_rate:.1f}% ({total_error}/{n_total})")
    print(f"     平均品質分: {score_mean:.1f}% (σ={score_std:.1f})")
    print(f"     延遲: mean={lat_mean:.0f}ms, p50={lat_p50:.0f}ms, p95={lat_p95:.0f}ms")
    print(f"     按類別:")
    for cat, data in by_category.items():
        print(f"       {cat}: {data['pass_rate']:.0f}% ({data['pass']}/{data['total']})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent 50 Loops Benchmark")
    parser.add_argument("--config", type=str, default=None,
                        help="只跑某一組 (Full_System/No_Quality/No_Teaching/Single_Agent)")
    parser.add_argument("--loops", type=int, default=3,
                        help="重複次數（每次 15 條 × loops）")
    parser.add_argument("--model", type=str, default="glm-4.5-air",
                        help="模型名稱")
    args = parser.parse_args()

    print("=" * 70)
    print("  🏁 Multi-Agent Benchmark")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型: {args.model}")
    print(f"  Loops: {args.loops} × {len(TEST_CASES)} 條 = {args.loops * len(TEST_CASES)} 次/組")
    print("=" * 70)

    if args.config:
        configs = [args.config]
    else:
        configs = list(ABLATION_CONFIGS.keys())

    all_results = {}
    for cfg in configs:
        result = run_single_config(cfg, loops=args.loops, model=args.model)
        all_results[cfg] = result

    # ─── 對比表 ───
    print(f"\n{'═' * 80}")
    print("  📊 總對比表")
    print(f"{'═' * 80}\n")
    print(f"  {'配置':<20} {'通過率':>8} {'品質分':>8} {'延遲p50':>10} {'錯誤率':>8}")
    print(f"  {'─' * 60}")
    for name, data in all_results.items():
        print(f"  {data['label'][:18]:<20} {data['pass_rate']:>7.1f}% {data['avg_score']:>7.1f}% "
              f"{data['latency_ms']['p50']:>8}ms {data['error_rate']:>7.1f}%")

    # 按類別對比
    print(f"\n  按類別通過率:")
    cats = sorted(set(tc["category"] for tc in TEST_CASES))
    header = f"  {'配置':<20}" + "".join(f" {cat:>10}" for cat in cats)
    print(header)
    print(f"  {'─' * (20 + 10 * len(cats))}")
    for name, data in all_results.items():
        row = f"  {data['label'][:18]:<20}"
        for cat in cats:
            if cat in data["by_category"]:
                rate = data["by_category"][cat]["pass_rate"]
                row += f" {rate:>9.0f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # ─── Save ───
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "loops": args.loops,
            "test_cases": len(TEST_CASES),
            "total_per_config": args.loops * len(TEST_CASES),
        },
        "summary": {
            name: {
                "label": data["label"],
                "pass_rate": data["pass_rate"],
                "avg_score": data["avg_score"],
                "score_std": data["score_std"],
                "error_rate": data["error_rate"],
                "latency_ms": data["latency_ms"],
                "by_category": data["by_category"],
            }
            for name, data in all_results.items()
        },
        "details": all_results,
    }

    report_path = Path(__file__).parent / "results" / "multi_agent_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 完整報告: {report_path}")

    return report


if __name__ == "__main__":
    main()
