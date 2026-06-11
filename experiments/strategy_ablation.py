"""
strategy_ablation.py — 策略對比 + Agent 消融實驗

P0 實驗腳本，回答兩個核心問題：
1. 不同教學策略對學習效果有什麼影響？
2. 每個 Agent 的貢獻是什麼？（消融實驗）

跑法：
    python strategy_ablation.py --experiment strategy     # 策略對比
    python strategy_ablation.py --experiment ablation      # Agent 消融
    python strategy_ablation.py --experiment all           # 全跑

作者: yu
日期: 2026-06-04
"""

import json
import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# 專案路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

from agent_framework.core.strategy import get_strategy, STRATEGIES
from agent_framework.core.decision import Decision, DecisionLogger
from agent_framework.core.rate_limiter import APIGuard, get_api_guard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================
# 實驗控制變因
# ============================================

# 所有消融實驗使用同一模型、同一 temperature
CONTROLLED_MODEL = "glm-4-flash"
CONTROLLED_TEMPERATURE = 0.3
CONTROLLED_MAX_TOKENS = 100

# ============================================
# 測試集
# ============================================

# 20 個基礎翻譯測試（中文→排灣語）
# 每條包含 accepted_variants 容許變體拼法
TRANSLATION_TESTS = [
    {"chinese": "你好", "paiwan": "djavadjavai", "variants": ["djavadjavai", "tjavatjavai", "nanguaq", "masalu", "aiyanga"]},
    {"chinese": "謝謝", "paiwan": "masalu", "variants": ["masalut", "masalu"]},
    {"chinese": "再見", "paiwan": "sadju", "variants": ["pacunan", "sadju"]},
    {"chinese": "水", "paiwan": "zanim", "variants": ["zaljum", "qanip", "zanim"]},
    {"chinese": "吃", "paiwan": "kan", "variants": ["keman", "kakan", "kan"]},
    {"chinese": "房子", "paiwan": "umaq", "variants": ["umaq"]},
    {"chinese": "人", "paiwan": "caucau", "variants": ["caucau", "kakacaucauan"]},
    {"chinese": "眼睛", "paiwan": "mata", "variants": ["mata", "maca"]},
    {"chinese": "手", "paiwan": "lama", "variants": ["lima", "lama"]},
    {"chinese": "太陽", "paiwan": "qadaw", "variants": ["qadaw"]},
    {"chinese": "月亮", "paiwan": "vuralac", "variants": ["qiljas", "vurasiac", "vuralac"]},
    {"chinese": "山", "paiwan": "garanga", "variants": ["gadu", "qungiljaw", "kazangalan", "garanga"]},
    {"chinese": "名字", "paiwan": "ngadan", "variants": ["ngadan"]},
    {"chinese": "朋友", "paiwan": "kadu", "variants": ["drangi", "kadu"]},
    {"chinese": "孩子", "paiwan": "aljak", "variants": ["aljak"]},
    {"chinese": "母親", "paiwan": "ina", "variants": ["tjina", "kina", "ina"]},
    {"chinese": "父親", "paiwan": "ama", "variants": ["tama", "kama", "ama"]},
    {"chinese": "一", "paiwan": "itua", "variants": ["ita", "usa", "ituk"]},
    {"chinese": "二", "paiwan": "drusa", "variants": ["drusa", "drusa"]},
    {"chinese": "好", "paiwan": "tarivak", "variants": ["nanguaq", "tarivak", "ui"]},
]


# ============================================
# 評分標準（固定，三層）
# ============================================

def check_translation(reply: str, expected: str, variants: list = None) -> dict:
    """
    固定評分標準：
    - Exact Match: 回覆中包含完全一致的目標詞
    - Variant Match: 回覆中包含任何變體拼法
    - Semantic Match: 需要人工複核，標記 pending_human_review

    Returns: {"match_level": "exact"|"variant"|"none", "correct": bool, "detail": str}
    """
    reply_lower = reply.lower()
    expected_lower = expected.lower()

    # Level 1: Exact Match（回覆中包含目標詞）
    if expected_lower in reply_lower:
        return {"match_level": "exact", "correct": True, "detail": f"Exact: '{expected}' found"}

    # Level 2: Variant Match（回覆中包含任何變體）
    if variants:
        for v in variants:
            if v.lower() in reply_lower:
                return {"match_level": "variant", "correct": True, "detail": f"Variant: '{v}' found (expected: '{expected}')"}

    # Level 3: No match — 標記人工複核
    return {"match_level": "none", "correct": False, "detail": f"No match for '{expected}' or variants in reply", "pending_human_review": True}

# 10 個教學策略測試場景（模擬學生對話）
TEACHING_SCENARIOS = [
    "教我排灣語的你好怎麼說",
    "謝謝用排灣語怎麼說",
    "我想學排灣語的數字",
    "水這個詞排灣語怎麼說",
    "推薦我下一個應該學的詞",
    "排灣語的爸爸怎麼說",
    "教我一個新的排灣語詞",
    "我學了哪些詞了",
    "排灣語的吃怎麼說",
    "太陽的排灣語是什麼",
]


# ============================================
# 實驗 1: 策略對比
# ============================================

def run_strategy_experiment(client: OpenAI, test_words: list = None,
                            scenarios: list = None, loops: int = 1):
    """
    策略對比實驗

    對每個策略：
    1. 初始化 Orchestrator（帶策略）
    2. 跑翻譯測試集（準確率）
    3. 跑教學場景（Decision Log 分析）
    4. 記錄所有數據
    """
    from agent_framework.agents.orchestrator import OrchestratorAgent

    test_words = test_words or TRANSLATION_TESTS
    scenarios = scenarios or TEACHING_SCENARIOS
    results = {}

    for strategy_name in STRATEGIES:
        logger.info(f"\n{'='*50}")
        logger.info(f"  策略: {strategy_name}")
        logger.info(f"{'='*50}")

        decision_logger = DecisionLogger(
            storage_dir=PROJECT_ROOT / "agent_framework" / "storage" / "decisions" / f"strategy_{strategy_name}"
        )

        orchestrator = OrchestratorAgent(
            client=client,
            api_guard=get_api_guard(),
            project_root=PROJECT_ROOT,
            strategy_name=strategy_name,
            decision_logger=decision_logger,
        )
        orchestrator._ensure_agents()

        # A. 翻譯測試
        translation_results = []
        correct = 0
        total = 0

        for test in test_words:
            user_input = f"{test['chinese']}的排灣語怎麼說？"
            try:
                reply = orchestrator.chat(user_input)
                # 使用固定評分標準
                match = check_translation(reply, test['paiwan'], test.get('variants', []))
                translation_results.append({
                    "chinese": test["chinese"],
                    "expected": test["paiwan"],
                    "reply": reply[:200],
                    "correct": match["correct"],
                    "match_level": match["match_level"],
                    "detail": match["detail"],
                })
                if match["correct"]:
                    correct += 1
                total += 1
                logger.info(f"  {'✅' if match['correct'] else '❌'} {test['chinese']} → {test['paiwan']} [{match['match_level']}] | {reply[:80]}")
            except Exception as e:
                translation_results.append({
                    "chinese": test["chinese"],
                    "expected": test["paiwan"],
                    "reply": f"ERROR: {e}",
                    "correct": False,
                    "match_level": "none",
                    "detail": str(e),
                    "pending_human_review": True,
                })
                total += 1
                logger.error(f"  ❌ {test['chinese']} 錯誤: {e}")

            time.sleep(1)  # API 限速

        accuracy = correct / total if total > 0 else 0

        # B. 教學場景測試（記錄 Decision Log）
        teaching_results = []
        for scenario in scenarios:
            try:
                reply = orchestrator.chat(scenario)
                teaching_results.append({
                    "input": scenario,
                    "reply": reply[:200],
                })
                logger.info(f"  📚 {scenario[:30]} → {reply[:60]}")
            except Exception as e:
                teaching_results.append({
                    "input": scenario,
                    "reply": f"ERROR: {e}",
                })
            time.sleep(1)

        decision_logger.flush()

        # 收集 Decision Log
        decisions = decision_logger.get_decisions(limit=100)

        results[strategy_name] = {
            "strategy": strategy_name,
            "translation_accuracy": round(accuracy, 4),
            "translation_correct": correct,
            "translation_total": total,
            "translation_results": translation_results,
            "teaching_results": teaching_results,
            "decision_count": len(decisions),
            "decisions_sample": [d.to_dict() for d in decisions[:10]],
        }

        logger.info(f"\n  📊 策略 {strategy_name} 結果:")
        logger.info(f"    翻譯準確率: {accuracy:.1%} ({correct}/{total})")
        logger.info(f"    決策記錄: {len(decisions)} 條")

        # 清理歷史，避免跨策略污染
        orchestrator.clear_history()

    return results


# ============================================
# 實驗 2: Agent 消融
# ============================================

def _run_translation_direct(client: OpenAI, test_words: list) -> dict:
    """Baseline 1: LLM 直翻（控制：同模型、同 temperature、無框架）"""
    correct = 0
    exact_correct = 0
    total = 0
    results = []

    for test in test_words:
        try:
            response = client.chat.completions.create(
                model=CONTROLLED_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "你是排灣語翻譯器。將用戶輸入的中文翻譯成排灣語。\n"
                        "規則：\n"
                        "1. 只輸出翻譯結果，不要解釋\n"
                        "2. 如果不確定，輸出 [不確定]\n"
                        "3. 不要編造不確定的翻譯\n"
                    )},
                    {"role": "user", "content": f"將以下中文翻譯成排灣語：{test['chinese']}"},
                ],
                temperature=CONTROLLED_TEMPERATURE,
                max_tokens=CONTROLLED_MAX_TOKENS,
            )
            reply = response.choices[0].message.content.strip()
            match = check_translation(reply, test['paiwan'], test.get('variants', []))
            results.append({
                "chinese": test["chinese"], "expected": test["paiwan"],
                "reply": reply, "correct": match["correct"],
                "match_level": match["match_level"], "detail": match["detail"],
            })
            if match["correct"]:
                correct += 1
            if match["match_level"] == "exact":
                exact_correct += 1
            total += 1
        except Exception as e:
            results.append({"chinese": test["chinese"], "error": str(e), "pending_human_review": True})
            total += 1
        time.sleep(0.5)

    return {
        "name": "llm_direct",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct/total if total else 0,
        "exact_accuracy": exact_correct/total if total else 0,
        "correct": correct, "exact_correct": exact_correct, "total": total,
        "results": results,
    }


def _run_rag_only(test_words: list) -> dict:
    """Baseline 2: RAG Only（語料匹配，無 LLM）"""
    try:
        from rag_service import PaiwanRAG
        rag = PaiwanRAG()
        rag.build_index()
    except Exception as e:
        logger.error(f"RAG 載入失敗: {e}")
        return {"name": "rag_only", "accuracy": 0, "error": str(e)}

    correct = 0
    total = 0
    results = []

    for test in test_words:
        try:
            matches = rag.search(test["chinese"], top_k=3)
            # 取最匹配的結果
            best = matches[0] if matches else {}
            reply = best.get("paiwan", "")
            match = check_translation(reply, test['paiwan'], test.get('variants', []))
            results.append({
                "chinese": test["chinese"], "expected": test["paiwan"],
                "reply": reply, "correct": match["correct"],
                "match_level": match["match_level"], "detail": match["detail"],
                "method": "rag",
            })
            if match["correct"]:
                correct += 1
            total += 1
        except Exception as e:
            results.append({"chinese": test["chinese"], "error": str(e), "pending_human_review": True})
            total += 1

    return {
        "name": "rag_only",
        "model": "N/A (corpus matching)",
        "temperature": "N/A",
        "accuracy": correct/total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


def _run_single_agent(client: OpenAI, test_words: list) -> dict:
    """Baseline 3: 單 Agent（Knowledge Agent 獨立運作）"""
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.core.message import AgentMessage

    agent = KnowledgeAgent(client=client, project_root=PROJECT_ROOT)
    correct = 0
    total = 0
    results = []

    for test in test_words:
        try:
            msg = AgentMessage.task_assign(
                from_agent="test",
                to_agent="knowledge",
                task="translate",
                params={"text": test["chinese"], "direction": "c2p"},
            )
            resp = agent.handle_message(msg)
            data = resp.payload.get("data", {})
            reply = data.get("translation", "")
            match = check_translation(reply, test['paiwan'], test.get('variants', []))
            results.append({
                "chinese": test["chinese"], "expected": test["paiwan"],
                "reply": reply, "correct": match["correct"],
                "match_level": match["match_level"], "detail": match["detail"],
                "method": data.get("method", "unknown"),
            })
            if match["correct"]:
                correct += 1
            total += 1
        except Exception as e:
            results.append({"chinese": test["chinese"], "error": str(e), "pending_human_review": True})
            total += 1
        time.sleep(0.5)

    return {
        "name": "single_agent",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct/total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


def _run_multi_agent(client: OpenAI, test_words: list) -> dict:
    """完整 Multi-Agent 系統（Knowledge + Quality 往返驗證）"""
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.agents.quality_agent import QualityAgent
    from agent_framework.core.message import AgentMessage, MessageType

    orchestrator = OrchestratorAgent(
        client=client,
        api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
        strategy_name="mastery_first",
    )
    orchestrator._ensure_agents()

    correct = 0
    total = 0
    results = []

    for test in test_words:
        try:
            # Step 1: Orchestrator 調用 Knowledge Agent 翻譯
            reply = orchestrator.chat(f"{test['chinese']}的排灣語怎麼說？")

            # Step 2: 從 Orchestrator 回覆中提取排灣語詞彙
            # 支持多種格式：markdown bold、引號、括號、純文本
            import re
            candidates = []

            # 優先提取 markdown **粗體** 中的詞
            bold_matches = re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', reply)
            if bold_matches:
                candidates.extend(bold_matches)

            # 提取「引號」中的詞
            quote_matches = re.findall(r'[「「]([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)[」」]', reply)
            if quote_matches:
                candidates.extend(quote_matches)

            # 提取括號中的詞
            paren_matches = re.findall(r'[（(]([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)[）)]', reply)
            if paren_matches:
                candidates.extend(paren_matches)

            # Fallback: 提取所有拉丁字母序列
            if not candidates:
                candidates = re.findall(r'[a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+', reply)

            # 過濾太短的（排除 a, the, is 等）
            candidates = [c for c in candidates if len(c) >= 3]

            # Step 3: Quality Agent 往返驗證
            verified_word = None
            if candidates:
                knowledge = orchestrator.bus._agents.get("knowledge")
                if knowledge:
                    for candidate in candidates[:5]:  # 最多驗證前5個候選詞
                        try:
                            # 反向翻譯：排灣語→中文
                            back_msg = AgentMessage(
                                sender="quality",
                                recipient="knowledge",
                                message_type=MessageType.TASK_ASSIGN,
                                task="translate",
                                data={"text": candidate, "direction": "p2c"},
                            )
                            back_result = knowledge.handle_message(back_msg)
                            back_chinese = ""
                            if back_result and back_result.data:
                                back_chinese = back_result.data.get("translation", "")

                            # 比對：反翻譯回來的中文是否包含原始中文
                            if test['chinese'] in back_chinese:
                                verified_word = candidate
                                break  # 往返驗證通過
                        except Exception:
                            continue

            # 如果往返驗證通過，用驗證後的詞；否則用原始 reply
            final_reply = verified_word if verified_word else reply
            match = check_translation(final_reply, test['paiwan'], test.get('variants', []))

            results.append({
                "chinese": test["chinese"], "expected": test["paiwan"],
                "reply": str(final_reply)[:200],
                "raw_reply": str(reply)[:200],
                "verified_word": verified_word,
                "correct": match["correct"],
                "match_level": match["match_level"], "detail": match["detail"],
                "round_trip": verified_word is not None,
            })
            if match["correct"]:
                correct += 1
            total += 1
        except Exception as e:
            results.append({"chinese": test["chinese"], "error": str(e), "pending_human_review": True})
            total += 1
        time.sleep(1)

    return {
        "name": "multi_agent",
        "model": CONTROLLED_MODEL,
        "temperature": CONTROLLED_TEMPERATURE,
        "accuracy": correct/total if total else 0,
        "correct": correct, "total": total,
        "results": results,
    }


def run_ablation_experiment(client: OpenAI, test_words: list = None):
    """
    Agent 消融實驗

    配置階梯：
    1. LLM Direct（無框架）
    2. RAG Only（語料匹配）
    3. Single Agent（Knowledge Agent 獨立）
    4. Multi-Agent（完整系統）
    """
    test_words = test_words or TRANSLATION_TESTS
    results = {}

    # 1. LLM Direct
    logger.info("\n" + "="*50)
    logger.info("  🔬 Baseline 1: LLM Direct")
    logger.info("="*50)
    r1 = _run_translation_direct(client, test_words)
    results["llm_direct"] = r1
    logger.info(f"  結果: {r1['accuracy']:.1%} ({r1['correct']}/{r1['total']})")

    # 2. RAG Only
    logger.info("\n" + "="*50)
    logger.info("  🔬 Baseline 2: RAG Only")
    logger.info("="*50)
    r2 = _run_rag_only(test_words)
    results["rag_only"] = r2
    logger.info(f"  結果: {r2['accuracy']:.1%} ({r2['correct']}/{r2['total']})")

    # 3. Single Agent
    logger.info("\n" + "="*50)
    logger.info("  🔬 Baseline 3: Single Agent (Knowledge Only)")
    logger.info("="*50)
    r3 = _run_single_agent(client, test_words)
    results["single_agent"] = r3
    logger.info(f"  結果: {r3['accuracy']:.1%} ({r3['correct']}/{r3['total']})")

    # 4. Multi-Agent
    logger.info("\n" + "="*50)
    logger.info("  🔬 Full: Multi-Agent")
    logger.info("="*50)
    r4 = _run_multi_agent(client, test_words)
    results["multi_agent"] = r4
    logger.info(f"  結果: {r4['accuracy']:.1%} ({r4['correct']}/{r4['total']})")

    return results


# ============================================
# 主程式
# ============================================

def save_results(results: dict, experiment_name: str):
    """保存實驗結果"""
    output_dir = PROJECT_ROOT / "experiment_results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    output_path = output_dir / filename

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"\n📁 結果已保存: {output_path}")
    return output_path


def print_summary(results: dict, experiment_name: str):
    """印出結果摘要表格"""
    print(f"\n{'='*60}")
    print(f"  📊 {experiment_name} 實驗結果摘要")
    print(f"  控制變因: model={CONTROLLED_MODEL}, temperature={CONTROLLED_TEMPERATURE}")
    print(f"  評分: Exact Match + Variant Match + pending_human_review")
    print(f"{'='*60}")

    if experiment_name == "strategy":
        print(f"\n  {'策略':<25} {'準確率':>8} {'正確/總數':>12} {'決策數':>8}")
        print(f"  {'-'*55}")
        for name, r in results.items():
            print(f"  {name:<25} {r['translation_accuracy']:>7.1%} "
                  f"{r['translation_correct']:>5}/{r['translation_total']:<5} "
                  f"{r['decision_count']:>8}")

    elif experiment_name == "ablation":
        print(f"\n  {'配置':<20} {'準確率':>8} {'正確/總數':>12} {'模型':>15}")
        print(f"  {'-'*57}")
        for name, r in results.items():
            print(f"  {r['name']:<20} {r['accuracy']:>7.1%} "
                  f"{r['correct']:>5}/{r['total']:<5} "
                  f"{r.get('model', 'N/A'):>15}")

    # 待人工複核數量
    pending = 0
    for r in results.values():
        for item in r.get("results", []):
            if item.get("pending_human_review"):
                pending += 1
    if pending > 0:
        print(f"\n  ⚠️  {pending} 筆結果待人工複核")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEAF 策略對比 + Agent 消融實驗")
    parser.add_argument("--experiment", choices=["strategy", "ablation", "all"],
                        default="all", help="要跑的實驗")
    parser.add_argument("--test-count", type=int, default=20, help="測試題數")
    args = parser.parse_args()

    # 初始化
    client = OpenAI(
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    test_words = TRANSLATION_TESTS[:args.test_count]

    if args.experiment in ("strategy", "all"):
        logger.info("🧪 開始策略對比實驗...")
        strategy_results = run_strategy_experiment(client, test_words)
        save_results(strategy_results, "strategy")
        print_summary(strategy_results, "strategy")

    if args.experiment in ("ablation", "all"):
        logger.info("🧪 開始 Agent 消融實驗...")
        ablation_results = run_ablation_experiment(client, test_words)
        save_results(ablation_results, "ablation")
        print_summary(ablation_results, "ablation")

    logger.info("✅ 所有實驗完成")
