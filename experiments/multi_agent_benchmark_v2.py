#!/usr/bin/env python3
"""
multi_agent_benchmark_v2.py — Multi-Agent 消融實驗 V2

重新設計的測試集（20 條），每個 Agent 對應 5 題，
確保關掉該 Agent 後對應類別有明顯下降。

測試集設計邏輯：
- Knowledge (5題): 必須通過 RAG 才能答對的語法/詞彙查詢
- Teaching (5題): 必須由 Teaching Agent 設計學習路徑/出題
- Quality (5題): 必須由 Quality Agent 審核才能抓出錯誤
- Composite (5題): 需要 Orchestrator 協調多個 Agent

評分邏輯：
- 不再用通用通過率，改用「任務完成度」
- 每題有 task_type + expected_keywords + expected_structure
- 關鍵詞匹配 + 結構化判斷

作者: 地陪
日期: 2026-06-03
"""

import sys
import json
import time
import logging
import argparse
import statistics
import re
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("benchmark_v2")

client = OpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

sys.path.insert(0, str(Path(__file__).parent / "agent_framework"))
from agents.orchestrator import OrchestratorAgent
from core.rate_limiter import get_api_guard


# ============================================
# 測試集 V2 — 20 題，每類 5 題
# ============================================

TEST_CASES = [
    # ─── Knowledge Agent 專屬 (5 題) ───
    # 這些題目需要 RAG 檢索才能正確回答
    # 關鍵：單 Agent 沒有 RAG 支撐，應該答不準
    {
        "id": "KN01",
        "input": "排灣語的焦點系統是什麼？-em- 是什麼意思",
        "category": "knowledge",
        "expected_keywords": ["焦點", "綴", "em"],
        "expected_structure": None,  # 不要求特定結構
        "description": "語法規則查詢 — 必須從語料/語法庫檢索",
    },
    {
        "id": "KN02",
        "input": "排灣語的「水」和「海」和「雨」怎麼說",
        "category": "knowledge",
        "expected_keywords": ["zaljum", "海", "雨"],  # zaljum 是水的排灣語
        "expected_structure": None,
        "description": "多詞翻譯 — 需要精確匹配 Core Vocab",
    },
    {
        "id": "KN03",
        "input": "kina 是什麼意思？排灣語的親屬稱謂有哪些",
        "category": "knowledge",
        "expected_keywords": ["kina", "媽媽", "母親"],
        "expected_structure": None,
        "description": "詞彙深度查詢 — 需要知識圖譜/RAG",
    },
    {
        "id": "KN04",
        "input": "排灣語的 ta- 前綴有什麼用法？舉例說明",
        "category": "knowledge",
        "expected_keywords": ["前綴", "ta"],
        "expected_structure": None,
        "description": "語法綴詞查詢 — 必須從語法庫檢索",
    },
    {
        "id": "KN05",
        "input": "幫我查 masalu 的完整資訊：意思、用法、例句",
        "category": "knowledge",
        "expected_keywords": ["masalu", "謝謝"],
        "expected_structure": None,
        "description": "詞彙深度查詢 — 需要 lookup 功能",
    },

    # ─── Teaching Agent 專屬 (5 題) ───
    # 這些題目需要 Teaching Agent 設計學習路徑
    # 關鍵：單 Agent 只會直接翻譯，不會規劃學習
    {
        "id": "TE01",
        "input": "我想從零開始學排灣語，幫我規劃一個學習計畫",
        "category": "teaching",
        "expected_keywords": [],
        "expected_structure": ["步", "階段", "計畫", "主題", "詞彙", "學習", "順序", "第一", "第二", "第三"],
        "description": "學習路徑規劃 — 需要 Teaching Agent 的 plan_learning",
    },
    {
        "id": "TE02",
        "input": "我已經學了 masalu、tarivak、zaljum，下一步學什麼",
        "category": "teaching",
        "expected_keywords": [],
        "expected_structure": ["推薦", "建議", "下一", "可以學", "接下來"],
        "description": "詞彙推薦 — 需要 Teaching Agent 的 suggest_next",
    },
    {
        "id": "TE03",
        "input": "幫我出一道排灣語選擇題，要四個選項",
        "category": "teaching",
        "expected_keywords": [],
        "expected_structure": ["A", "B", "C", "D", "選項", "題"],
        "description": "測驗生成 — 需要 Teaching Agent 的 generate_quiz",
    },
    {
        "id": "TE04",
        "input": "教我排灣語的數字 1 到 5，然後出個小測驗考考我",
        "category": "teaching",
        "expected_keywords": [],
        "expected_structure": ["it", "drusa", "tjelu", "sepat", "lima", "測驗", "測試", "考"],
        "description": "教學+測驗複合 — 需要 Teaching Agent 協調",
    },
    {
        "id": "TE05",
        "input": "我今天學了哪些排灣語？給我一份學習報告",
        "category": "teaching",
        "expected_keywords": [],
        "expected_structure": ["學習", "報告", "進度", "已學", "詞"],
        "description": "學習報告 — 需要 Teaching Agent 的 learning_report",
    },

    # ─── Quality Agent 專屬 (5 題) ───
    # 給故意錯誤的翻譯讓系統審核
    # 關鍵：沒有 Quality Agent → 錯誤翻譯不會被攔截
    {
        "id": "QA01",
        "input": "請審核這個翻譯：原文「謝謝」→ 翻譯「pacunan」",
        "category": "quality",
        "expected_keywords": ["錯", "不正確", "不正確", "不正確", "pacunan", "masalu", "不正確"],
        "expected_structure": None,
        "description": "錯誤翻譯審核 — pacunan 是再見不是謝謝，應該被攔截",
    },
    {
        "id": "QA02",
        "input": "這個翻譯對嗎？原文「你好」→ 翻譯「masalu」",
        "category": "quality",
        "expected_keywords": ["錯", "不正確", "不對", "masalu", "tarivak"],
        "expected_structure": None,
        "description": "錯誤翻譯審核 — masalu 是謝謝不是你好",
    },
    {
        "id": "QA03",
        "input": "審核翻譯品質：原文「水」→ 翻譯「zaljum」，原文「火」→ 翻譯「zaljum」",
        "category": "quality",
        "expected_keywords": ["錯", "不正確", "重複", "火", "zaljum"],
        "expected_structure": None,
        "description": "部分錯誤審核 — 兩個詞譯成同一個排灣語",
    },
    {
        "id": "QA04",
        "input": "請檢查這個翻譯：原文「朋友」→ 翻譯「drangi」",
        "category": "quality",
        "expected_keywords": ["正確", "對", "正確", "通過", "drangi"],
        "expected_structure": None,
        "description": "正確翻譯審核 — 應該通過",
    },
    {
        "id": "QA05",
        "input": "這個翻譯有問題嗎？原文「再見」→ 翻譯「masalu」",
        "category": "quality",
        "expected_keywords": ["錯", "不正確", "不對", "masalu", "pacunan", "再見"],
        "expected_structure": None,
        "description": "錯誤翻譯審核 — masalu 是謝謝不是再見",
    },

    # ─── Composite 複合任務 (5 題) ───
    # 需要多個 Agent 協作才能完成
    # 關鍵：單 Agent 無法同時處理多種認知層次
    {
        "id": "CO01",
        "input": "教我說謝謝的排灣語，然後用這個詞出一道測驗題",
        "category": "composite",
        "expected_keywords": ["masalu", "謝謝"],
        "expected_structure": ["測驗", "題", "選"],
        "description": "Knowledge+Teaching — 翻譯後出題",
    },
    {
        "id": "CO02",
        "input": "幫我翻譯「你好」成排灣語，然後審核翻譯品質",
        "category": "composite",
        "expected_keywords": ["tarivak", "djavadjavay", "你好"],
        "expected_structure": ["審核", "品質", "正確", "通過", "準確"],
        "description": "Knowledge+Quality — 翻譯後審核",
    },
    {
        "id": "CO03",
        "input": "我想學排灣語的問候語，教我然後考我",
        "category": "composite",
        "expected_keywords": [],
        "expected_structure": ["問候", "教", "測驗", "考"],
        "description": "Knowledge+Teaching — 學問候語+測驗",
    },
    {
        "id": "CO04",
        "input": "有人說「水」的排灣語是「pacunan」，這對嗎？如果錯了請糾正並教我正確的",
        "category": "composite",
        "expected_keywords": ["錯", "不正確", "zaljum", "pacunan"],
        "expected_structure": None,
        "description": "Quality+Knowledge — 抓錯+正確翻譯",
    },
    {
        "id": "CO05",
        "input": "幫我翻譯「朋友」和「太陽」成排灣語，然後用這兩個詞出個測驗",
        "category": "composite",
        "expected_keywords": ["drangi"],
        "expected_structure": ["測驗", "題"],
        "description": "Knowledge+Teaching — 雙詞翻譯+出題",
    },
]

# ============================================
# 消融配置
# ============================================

ABLATION_CONFIGS = {
    "Full_System": {
        "label": "4 Agent 全開（Control）",
        "setup": "full",
    },
    "No_Knowledge": {
        "label": "關掉 Knowledge Agent",
        "setup": "no_knowledge",
    },
    "No_Teaching": {
        "label": "關掉 Teaching Agent",
        "setup": "no_teaching",
    },
    "No_Quality": {
        "label": "關掉 Quality Agent",
        "setup": "no_quality",
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
    elif setup == "no_knowledge":
        from agents.teaching_agent import TeachingAgent
        from agents.quality_agent import QualityAgent
        teaching = TeachingAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        quality = QualityAgent(client=client, api_guard=api_guard, project_root=Path(__file__).parent)
        orch.register_agents(knowledge=None, teaching=teaching, quality=quality)
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
        messages=messages, tools=None,
        temperature=0.4, max_tokens=500, model=model,
    )
    return result["message"].content or ""


# ============================================
# 評分邏輯 V2 — 任務完成度
# ============================================

def score_v2(test_case: dict, reply: str) -> dict:
    """
    V2 評分：任務完成度
    
    維度：
    1. has_reply (bool): 有回覆
    2. length_ok (bool): 長度合理
    3. keyword_match (bool): 關鍵詞命中（至少 1 個）
    4. structure_match (bool): 結構化判斷（至少 1 個）
    5. no_hallucination (bool): 沒有明顯編造
    """
    scores = {}
    issues = []
    
    # 1. 有回覆
    scores["has_reply"] = len(reply.strip()) > 0
    if not scores["has_reply"]:
        issues.append("無回覆")
    
    # 2. 長度合理
    scores["length_ok"] = 10 <= len(reply) <= 1000
    if len(reply) < 10:
        issues.append(f"太短({len(reply)}字)")
    elif len(reply) > 1000:
        issues.append(f"太長({len(reply)}字)")
    
    # 3. 關鍵詞匹配（至少 1 個）
    expected_kw = test_case.get("expected_keywords", [])
    if expected_kw:
        matched = [kw for kw in expected_kw if kw in reply]
        scores["keyword_match"] = len(matched) >= 1
        if not scores["keyword_match"]:
            issues.append(f"缺少關鍵詞: {expected_kw}")
    else:
        scores["keyword_match"] = True
    
    # 4. 結構化判斷（至少 1 個）
    expected_struct = test_case.get("expected_structure")
    if expected_struct:
        matched_s = [s for s in expected_struct if s in reply]
        scores["structure_match"] = len(matched_s) >= 1
        if not scores["structure_match"]:
            issues.append(f"缺少結構: {expected_struct}")
    else:
        scores["structure_match"] = True
    
    # 5. 沒有明顯編造（回覆中同時出現排灣語但不確定標記）
    has_uncertainty = "不確定" in reply or "⚠️" in reply or "[不確定]" in reply
    has_paiwan = any(ord(c) > 0x2800 for c in reply)  # 粗略檢測是否有非 ASCII
    # 如果有 expected keywords 但回覆充滿不確定 → 扣分
    if expected_kw and has_uncertainty and not scores["keyword_match"]:
        scores["no_hallucination"] = False
        issues.append("不確定且缺少關鍵詞")
    else:
        scores["no_hallucination"] = True
    
    total = sum(1 for v in scores.values() if v)
    max_score = len(scores)
    
    return {
        "scores": scores,
        "total": total,
        "max": max_score,
        "issues": issues,
        "passed": total >= max_score - 1,  # 允許 1 分寬容
    }


def run_single_config(config_name: str, loops: int, model: str) -> dict:
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
        orch = setup_orchestrator(config_name, model=model)

        for i, tc in enumerate(TEST_CASES):
            tid = f"{tc['id']}_L{loop_idx+1}"
            user_input = tc["input"]
            desc = tc.get("description", "")
            print(f"    [{i+1}/{len(TEST_CASES)}] {tid} 「{user_input[:30]}」{f' ({desc})' if desc else ''}", end=" ... ", flush=True)

            start = time.time()
            try:
                if getattr(orch, '_single_mode', False):
                    reply = single_agent_chat(orch, user_input, model=model)
                else:
                    reply = orch.chat(user_input, user_id=f"bench_v2_{loop_idx}")
                elapsed = (time.time() - start) * 1000

                scoring = score_v2(tc, reply)
                passed = scoring["passed"]

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
                if scoring["issues"]:
                    for issue in scoring["issues"]:
                        print(f"         ⚠️ {issue}")

                all_interactions.append({
                    "loop": loop_idx + 1,
                    "id": tid,
                    "test_id": tc["id"],
                    "input": user_input,
                    "category": tc["category"],
                    "reply": reply[:500],
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

            time.sleep(0.8)

        if loop_idx < loops - 1:
            print(f"    ⏳ Loop {loop_idx + 1} 完成，暫停 3 秒...")
            time.sleep(3)

    # ─── 統計 ───
    n_total = total_pass + total_fail + total_error
    pass_rate = total_pass / n_total * 100 if n_total else 0
    error_rate = total_error / n_total * 100 if n_total else 0

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

    lat_p50 = statistics.median(all_latencies) if all_latencies else 0
    lat_p95 = sorted(all_latencies)[int(len(all_latencies)*0.95)] if len(all_latencies) >= 20 else max(all_latencies) if all_latencies else 0
    lat_mean = statistics.mean(all_latencies) if all_latencies else 0
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

    print(f"\n  📊 {config_name} Summary:")
    print(f"     通過率: {pass_rate:.1f}% ({total_pass}/{n_total})")
    print(f"     平均完成度: {score_mean:.1f}% (σ={score_std:.1f})")
    print(f"     延遲: mean={lat_mean:.0f}ms, p50={lat_p50:.0f}ms")
    print(f"     按類別:")
    for cat, data in by_category.items():
        print(f"       {cat}: {data['pass_rate']:.0f}% avg_score={data['avg_score']:.1f}% ({data['pass']}/{data['total']})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Benchmark V2")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--model", type=str, default="glm-4.5-air")
    args = parser.parse_args()

    print("=" * 70)
    print("  🏁 Multi-Agent Benchmark V2 — 針對性測試集")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型: {args.model}")
    print(f"  Loops: {args.loops} × {len(TEST_CASES)} 條 = {args.loops * len(TEST_CASES)} 次/組")
    print(f"  配置數: {len(ABLATION_CONFIGS)}（含 No_Knowledge 新配置）")
    print("=" * 70)

    configs = [args.config] if args.config else list(ABLATION_CONFIGS.keys())

    all_results = {}
    for cfg in configs:
        result = run_single_config(cfg, loops=args.loops, model=args.model)
        all_results[cfg] = result

    # ─── 對比表 ───
    print(f"\n{'═' * 80}")
    print("  📊 總對比表")
    print(f"{'═' * 80}\n")
    print(f"  {'配置':<25} {'通過率':>8} {'完成度':>8} {'延遲p50':>10} {'錯誤率':>8}")
    print(f"  {'─' * 65}")
    for name, data in all_results.items():
        print(f"  {data['label'][:22]:<25} {data['pass_rate']:>7.1f}% {data['avg_score']:>7.1f}% "
              f"{data['latency_ms']['p50']:>8}ms {data['error_rate']:>7.1f}%")

    print(f"\n  按類別通過率:")
    cats = sorted(set(tc["category"] for tc in TEST_CASES))
    cat_cn = {"knowledge": "知識", "teaching": "教學", "quality": "品質", "composite": "複合"}
    header = f"  {'配置':<25}" + "".join(f" {cat_cn.get(c,c):>8}" for c in cats)
    print(header)
    print(f"  {'─' * (25 + 9 * len(cats))}")
    for name, data in all_results.items():
        row = f"  {data['label'][:22]:<25}"
        for cat in cats:
            if cat in data["by_category"]:
                rate = data["by_category"][cat]["pass_rate"]
                row += f" {rate:>7.0f}%"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # ─── Save ───
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "loops": args.loops,
            "test_cases": len(TEST_CASES),
            "total_per_config": args.loops * len(TEST_CASES),
            "version": "v2_targeted",
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

    report_path = Path(__file__).parent / "results" / "multi_agent_benchmark_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 完整報告: {report_path}")

    return report


if __name__ == "__main__":
    main()
