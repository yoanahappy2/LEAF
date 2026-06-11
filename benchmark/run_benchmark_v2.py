#!/usr/bin/env python3
"""
LEAF Benchmark v2.0 — 穩定可發表版

改動：
1. Evaluation: variant-aware word-boundary match（禁止 substring）
2. Lexicon: 唯一真實來源（benchmark/lexicon_v2.json）
3. Dataset: benchmark/benchmark_v2.json
4. 四層消融: LLM Direct → RAG → Single Agent → Multi-Agent

用法：
    python3 benchmark/run_benchmark_v2.py --experiment ablation
    python3 benchmark/run_benchmark_v2.py --experiment ablation --test-count 10
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── 路徑設定 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 控制變因 ──
CONTROLLED_MODEL = "glm-4-flash"
CONTROLLED_TEMPERATURE = 0.3

# ── 載入 Lexicon 和 Benchmark ──
def load_lexicon(path: str = None) -> dict:
    path = path or str(PROJECT_ROOT / "benchmark" / "lexicon_v2.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 去掉 _meta
    return {k: v for k, v in data.items() if not k.startswith("_")}

def load_benchmark(path: str = None) -> list:
    path = path or str(PROJECT_ROOT / "benchmark" / "benchmark_v2.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================
# 評分標準 v2 — variant-aware word-boundary match
# ============================================
def evaluate_v2(pred: str, target: str, lexicon: dict) -> dict:
    """
    v2 評分：
    - 從 pred 中提取排灣語詞彙（word-boundary，不是 substring）
    - 比對 lexicon[target] 的 variants
    - exact_match: pred == preferred
    - variant_match: pred in variants (但不是 preferred)
    - correct: pred in variants

    ❌ 禁止 substring match（"in" in "kina" 這種）
    ✅ 只認 word-boundary 匹配
    """
    entry = lexicon.get(target)
    if not entry:
        return {
            "match_level": "error",
            "correct": False,
            "detail": f"Target '{target}' not in lexicon",
            "score": 0.0,
        }

    variants = entry["variants"]
    preferred = entry["preferred"]

    # Step 1: 清理 pred（去掉中文描述、標點、空白）
    pred_clean = pred.strip().lower()

    # Step 2: 提取排灣語詞彙（word-boundary match）
    # 用 regex 找所有拉丁字母序列（排除太短的）
    candidates = re.findall(r'[a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+', pred_clean)
    candidates = [c for c in candidates if len(c) >= 2]

    # Step 3: 比對 variants（exact match，不是 substring）
    matched_variant = None
    for candidate in candidates:
        # 完全匹配某個 variant
        if candidate in variants:
            matched_variant = candidate
            break

    # Step 4: 判定
    if matched_variant is None:
        return {
            "match_level": "none",
            "correct": False,
            "detail": f"No variant match. Candidates: {candidates[:5]}, Expected: {variants}",
            "score": 0.0,
            "candidates": candidates[:5],
        }
    elif matched_variant == preferred:
        return {
            "match_level": "exact",
            "correct": True,
            "detail": f"Exact match: '{matched_variant}' == preferred '{preferred}'",
            "score": 1.0,
            "matched": matched_variant,
        }
    else:
        return {
            "match_level": "variant",
            "correct": True,
            "detail": f"Variant match: '{matched_variant}' in {variants} (preferred: '{preferred}')",
            "score": 0.8,
            "matched": matched_variant,
        }


# ============================================
# Baseline 1: LLM Direct（無 RAG，無工具）
# ============================================
def _run_llm_direct(client: OpenAI, test_cases: list, lexicon: dict) -> dict:
    prompt = (
        "你是排灣語翻譯助手。用戶會問你排灣語怎麼說，"
        "你只能回答排灣語詞彙本身，不要加中文解釋。"
        "如果你不知道，就回答「我不知道」。"
    )

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            resp = client.chat.completions.create(
                model=CONTROLLED_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": tc["input"]},
                ],
                temperature=CONTROLLED_TEMPERATURE,
                max_tokens=100,
            )
            reply = resp.choices[0].message.content.strip()
            ev = evaluate_v2(reply, tc["target"], lexicon)

            results.append({
                "id": tc["id"], "target": tc["target"],
                "input": tc["input"],
                "reply": reply[:200],
                "match_level": ev["match_level"],
                "correct": ev["correct"],
                "score": ev["score"],
                "detail": ev["detail"],
            })
            if ev["correct"]:
                correct += 1
            score_sum += ev["score"]
            total += 1
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1

        time.sleep(0.3)

    return {
        "name": "llm_direct",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


# ============================================
# Baseline 2: RAG Only（語料庫直接匹配）
# ============================================
def _run_rag_only(test_cases: list, lexicon: dict, client: OpenAI) -> dict:
    """RAG Only：用 KnowledgeAgent 的 translate，但不經 Orchestrator"""
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.core.rate_limiter import get_api_guard
    from agent_framework.core.message import AgentMessage, MessageType

    agent = KnowledgeAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
    )

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            msg = AgentMessage(
                from_agent="evaluator", to_agent="knowledge",
                type=MessageType.TASK_ASSIGN,
                payload={"task": "translate", "params": {"text": tc["target"], "direction": "c2p"}},
            )
            resp = agent.handle_message(msg)
            reply = ""
            if resp and resp.payload:
                d = resp.payload.get("data", {})
                if isinstance(d, dict):
                    reply = d.get("translation", "")
                else:
                    reply = str(d) if d else ""
            reply = str(reply)

            ev = evaluate_v2(reply, tc["target"], lexicon)
            results.append({
                "id": tc["id"], "target": tc["target"],
                "reply": reply[:200],
                "match_level": ev["match_level"],
                "correct": ev["correct"],
                "score": ev["score"],
                "detail": ev["detail"],
            })
            if ev["correct"]:
                correct += 1
            score_sum += ev["score"]
            total += 1
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1

    return {
        "name": "rag_only",
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


# ============================================
# Baseline 3: Single Agent (Knowledge Only)
# ============================================
def _run_single_agent(client: OpenAI, test_cases: list, lexicon: dict) -> dict:
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.core.rate_limiter import get_api_guard
    from agent_framework.core.message import AgentMessage, MessageType

    agent = KnowledgeAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
    )

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            msg = AgentMessage(
                from_agent="evaluator", to_agent="knowledge",
                type=MessageType.TASK_ASSIGN,
                payload={"task": "translate", "params": {"text": tc["target"], "direction": "c2p"}},
            )
            resp = agent.handle_message(msg)
            reply = ""
            if resp and resp.payload:
                d = resp.payload.get("data", {})
                if isinstance(d, dict):
                    reply = d.get("translation", "")
                else:
                    reply = str(d) if d else ""
            reply = str(reply)

            ev = evaluate_v2(reply, tc["target"], lexicon)
            results.append({
                "id": tc["id"], "target": tc["target"],
                "reply": reply[:200],
                "match_level": ev["match_level"],
                "correct": ev["correct"],
                "score": ev["score"],
                "detail": ev["detail"],
            })
            if ev["correct"]:
                correct += 1
            score_sum += ev["score"]
            total += 1
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1

        time.sleep(0.3)

    return {
        "name": "single_agent",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


# ============================================
# Full: Multi-Agent (Orchestrator + Knowledge + Quality + Teaching)
# ============================================
def _run_multi_agent(client: OpenAI, test_cases: list, lexicon: dict) -> dict:
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.core.rate_limiter import get_api_guard

    orchestrator = OrchestratorAgent(
        client=client,
        api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
        strategy_name="mastery_first",
    )
    orchestrator._ensure_agents()

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            # Orchestrator 完整 ReAct 循環
            reply = orchestrator.chat(tc["input"])

            # 從 reply 中提取排灣語詞彙
            ev = evaluate_v2(reply, tc["target"], lexicon)

            # 如果直接評分不通過，嘗試提取 markdown bold 中的詞再評一次
            if not ev["correct"]:
                bold_words = re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', reply)
                for bw in bold_words:
                    ev2 = evaluate_v2(bw, tc["target"], lexicon)
                    if ev2["correct"]:
                        ev = ev2
                        break

            results.append({
                "id": tc["id"], "target": tc["target"],
                "input": tc["input"],
                "reply": str(reply)[:300],
                "match_level": ev["match_level"],
                "correct": ev["correct"],
                "score": ev["score"],
                "detail": ev["detail"],
            })
            if ev["correct"]:
                correct += 1
            score_sum += ev["score"]
            total += 1
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1

        time.sleep(0.5)

    return {
        "name": "multi_agent",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


# ============================================
# 主程式
# ============================================
def run_ablation(client: OpenAI, test_cases: list, lexicon: dict):
    """跑四層消融"""
    all_results = {}

    # Pre-flight
    dataset_hash = hashlib.md5(
        json.dumps(test_cases, ensure_ascii=False).encode()
    ).hexdigest()
    logger.info(f"🧪 LEAF Benchmark v2.0 — Agent 消融實驗")
    logger.info(f"   控制: model={CONTROLLED_MODEL}, temp={CONTROLLED_TEMPERATURE}")
    logger.info(f"   題數: {len(test_cases)}, hash: {dataset_hash[:8]}")
    logger.info(f"   評分: variant-aware word-boundary match")

    # 1. LLM Direct
    logger.info("\n==================================================")
    logger.info("  🔬 Baseline 1: LLM Direct")
    logger.info("==================================================")
    r1 = _run_llm_direct(client, test_cases, lexicon)
    all_results["llm_direct"] = r1
    logger.info(f"  結果: {r1['accuracy']*100:.1f}% ({r1['correct']}/{r1['total']}) avg_score={r1['avg_score']:.2f}")

    # 2. RAG Only
    logger.info("\n==================================================")
    logger.info("  🔬 Baseline 2: RAG Only")
    logger.info("==================================================")
    r2 = _run_rag_only(test_cases, lexicon, client)
    all_results["rag_only"] = r2
    logger.info(f"  結果: {r2['accuracy']*100:.1f}% ({r2['correct']}/{r2['total']}) avg_score={r2['avg_score']:.2f}")

    # 3. Single Agent
    logger.info("\n==================================================")
    logger.info("  🔬 Baseline 3: Single Agent (Knowledge Only)")
    logger.info("==================================================")
    r3 = _run_single_agent(client, test_cases, lexicon)
    all_results["single_agent"] = r3
    logger.info(f"  結果: {r3['accuracy']*100:.1f}% ({r3['correct']}/{r3['total']}) avg_score={r3['avg_score']:.2f}")

    # 4. Multi-Agent
    logger.info("\n==================================================")
    logger.info("  🔬 Full: Multi-Agent")
    logger.info("==================================================")
    r4 = _run_multi_agent(client, test_cases, lexicon)
    all_results["multi_agent"] = r4
    logger.info(f"  結果: {r4['accuracy']*100:.1f}% ({r4['correct']}/{r4['total']}) avg_score={r4['avg_score']:.2f}")

    return all_results


def save_results(results: dict, prefix: str = "ablation"):
    out_dir = PROJECT_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_v2_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"📁 結果已保存: {path}")
    return path


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("  📊 LEAF Benchmark v2.0 — 消融實驗結果")
    print("  控制變因: model=glm-4-flash, temperature=0.3")
    print("  評分: variant-aware word-boundary match")
    print("=" * 60)
    print()
    print(f"  {'配置':<20} {'準確率':>8} {'avg_score':>10} {'正確/總數':>10}")
    print("  " + "-" * 50)
    for name in ["llm_direct", "rag_only", "single_agent", "multi_agent"]:
        r = results[name]
        acc = f"{r['accuracy']*100:.1f}%"
        avg = f"{r['avg_score']:.2f}"
        ct = f"{r['correct']}/{r['total']}"
        print(f"  {name:<20} {acc:>8} {avg:>10} {ct:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEAF Benchmark v2.0")
    parser.add_argument("--experiment", choices=["ablation"], default="ablation")
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--lexicon", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    # 載入
    lexicon = load_lexicon(args.lexicon)
    benchmark = load_benchmark(args.benchmark)
    test_cases = benchmark[:args.test_count]

    print(f"Lexicon: {len(lexicon)} 詞")
    print(f"Benchmark: {len(test_cases)} 題")

    # API client
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        # 嘗試從 .env 讀
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in open(env_path):
                if line.startswith("ZHIPUAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("❌ 找不到 ZHIPUAI_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")

    # 跑
    results = run_ablation(client, test_cases, lexicon)
    save_results(results, "ablation")
    print_summary(results)
    logger.info("✅ 所有實驗完成")
