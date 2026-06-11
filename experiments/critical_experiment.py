#!/usr/bin/env python3
"""
致命實驗：Single Agent + ReAct vs Multi-Agent + ReAct

目的：證明或證偽 Multi-Agent Decomposition 的額外貢獻

實驗設計：
  - 控制組：Single Agent（只有 Knowledge Agent 的工具）+ ReAct 循環（5 輪）
  - 實驗組：Multi-Agent（Orchestrator + Knowledge + Quality + Teaching）+ ReAct 循環（5 輪）
  - 同模型、同語料、同測試集、同評估協議

預期結果：
  - 情況 A：兩個都 ~95% → 提升來自 ReAct，不是 Multi-Agent → 誠實結論
  - 情況 B：Multi-Agent > Single Agent → 證明 Agent Decomposition 有價值 → 海報升級
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL = "glm-4-flash"
TEMPERATURE = 0.3
MAX_REACT_TURNS = 5
API_KEY = os.environ.get("ZHIPUAI_API_KEY", "")
if not API_KEY:
    # 嘗試從 .env 讀取
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in open(env_path):
            if line.startswith("ZHIPUAI_API_KEY="):
                API_KEY = line.strip().split("=", 1)[1]
                break

# ── 評分（和 run_benchmark_v2 完全一致）──
def load_lexicon():
    path = PROJECT_ROOT / "benchmark" / "lexicon_v2.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}

def load_benchmark():
    path = PROJECT_ROOT / "benchmark" / "benchmark_v2.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def evaluate_v2(pred: str, target: str, lexicon: dict) -> dict:
    entry = lexicon.get(target)
    if not entry:
        return {"match_level": "error", "correct": False, "detail": f"Target '{target}' not in lexicon", "score": 0.0}

    variants = entry["variants"]
    preferred = entry["preferred"]
    pred_clean = pred.strip().lower()
    candidates = re.findall(r'[a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+', pred_clean)
    candidates = [c for c in candidates if len(c) >= 2]

    matched_variant = None
    for candidate in candidates:
        if candidate in variants:
            matched_variant = candidate
            break

    if matched_variant is None:
        return {"match_level": "none", "correct": False, "detail": f"No match. Candidates: {candidates[:5]}, Expected: {variants}", "score": 0.0, "candidates": candidates[:5]}
    elif matched_variant == preferred:
        return {"match_level": "exact", "correct": True, "detail": f"Exact: '{matched_variant}'", "score": 1.0, "matched": matched_variant}
    else:
        return {"match_level": "variant", "correct": True, "detail": f"Variant: '{matched_variant}' in {variants}", "score": 0.8, "matched": matched_variant}


# ── Single Agent + ReAct ──
def _run_single_agent_react(client: OpenAI, test_cases: list, lexicon: dict) -> dict:
    """
    模擬 ReAct 循環，但只給 Knowledge Agent 的工具。
    Orchestrator 的循環邏輯複製過來，但工具只有 translate + rag_search + lookup。
    """
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.core.rate_limiter import get_api_guard
    from agent_framework.core.message import AgentMessage, MessageType

    # 初始化 Knowledge Agent
    ka = KnowledgeAgent(client=client, api_guard=get_api_guard(), project_root=PROJECT_ROOT)
    ka._ensure_services()

    # 只給 Knowledge Agent 的工具（和 Orchestrator 給的一樣）
    single_agent_tools = [
        {
            "type": "function",
            "function": {
                "name": "translate",
                "description": "翻譯中文到排灣語。先用語料庫精確匹配，匹配不到再用 LLM。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "要翻譯的中文詞或句"},
                        "direction": {"type": "string", "enum": ["c2p", "p2c", "auto"], "description": "翻譯方向，預設 c2p"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "在排灣語語料庫中搜索相關例句和翻譯。適合翻譯失敗後換一種方式搜索。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索關鍵詞（中文或英文）"},
                        "top_k": {"type": "integer", "description": "返回結果數量，預設 5"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "查詞典，查看一個排灣語詞的完整資訊（級詞、親屬、相關詞）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string", "description": "要查詢的排灣語詞"},
                    },
                    "required": ["word"],
                },
            },
        },
    ]

    system_prompt = (
        "你是排灣語翻譯助手。用戶會問你中文詞彙的排灣語怎麼說。\n"
        "你必須使用工具來翻譯，不要自己猜。\n"
        "如果第一次翻譯失敗或結果不確定，嘗試用不同方式搜索（換關鍵詞、用英文搜索等）。\n"
        "最終只輸出排灣語詞彙本身，不要加解釋。"
    )

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tc["input"]},
            ]

            final_reply = ""
            tool_calls_made = []

            # ReAct 循環（和 Orchestrator 一樣最多 5 輪）
            for turn in range(MAX_REACT_TURNS):
                is_final_round = (turn >= 3)

                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=single_agent_tools if not is_final_round else None,
                    temperature=TEMPERATURE,
                    max_tokens=500,
                )

                msg = resp.choices[0].message

                if not is_final_round and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    messages.append(msg)

                    for tc_item in msg.tool_calls:
                        tool_name = tc_item.function.name
                        tool_args = json.loads(tc_item.function.arguments)
                        tool_calls_made.append(f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:60]})")

                        # 路由到 Knowledge Agent
                        if tool_name == "translate":
                            ka_msg = AgentMessage(
                                from_agent="evaluator", to_agent="knowledge",
                                type=MessageType.TASK_ASSIGN,
                                payload={"task": "translate", "params": tool_args},
                            )
                            ka_resp = ka.handle_message(ka_msg)
                            result_data = ka_resp.payload.get("data", {}) if ka_resp and ka_resp.payload else {}
                            result = {"translation": result_data.get("translation", ""), "method": result_data.get("method", "")}

                        elif tool_name == "rag_search":
                            ka_msg = AgentMessage(
                                from_agent="evaluator", to_agent="knowledge",
                                type=MessageType.TASK_ASSIGN,
                                payload={"task": "rag_search", "params": tool_args},
                            )
                            ka_resp = ka.handle_message(ka_msg)
                            result_data = ka_resp.payload.get("data", {}) if ka_resp and ka_resp.payload else {}
                            result = result_data if isinstance(result_data, dict) else {"results": str(result_data)}

                        elif tool_name == "lookup":
                            ka_msg = AgentMessage(
                                from_agent="evaluator", to_agent="knowledge",
                                type=MessageType.TASK_ASSIGN,
                                payload={"task": "lookup", "params": tool_args},
                            )
                            ka_resp = ka.handle_message(ka_msg)
                            result_data = ka_resp.payload.get("data", {}) if ka_resp and ka_resp.payload else {}
                            result = result_data if isinstance(result_data, dict) else {"info": str(result_data)}
                        else:
                            result = {"error": f"Unknown tool: {tool_name}"}

                        messages.append({
                            "role": "tool",
                            "content": json.dumps(result, ensure_ascii=False),
                            "tool_call_id": tc_item.id,
                        })

                    continue

                # LLM 直接回覆（最終輪或沒有 tool_calls）
                reply = msg.content or ""

                if not reply.strip() and turn < MAX_REACT_TURNS - 1:
                    messages.append({"role": "user", "content": "請根據以上工具調用結果，直接輸出排灣語詞彙。"})
                    continue

                final_reply = reply
                break

            # 評分
            ev = evaluate_v2(final_reply, tc["target"], lexicon)

            # 嘗試提取 bold 中的詞
            if not ev["correct"]:
                bold_words = re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', final_reply)
                for bw in bold_words:
                    ev2 = evaluate_v2(bw, tc["target"], lexicon)
                    if ev2["correct"]:
                        ev = ev2
                        break

            results.append({
                "id": tc["id"], "target": tc["target"],
                "input": tc["input"],
                "reply": final_reply[:300],
                "tool_calls": tool_calls_made,
                "match_level": ev["match_level"],
                "correct": ev["correct"],
                "score": ev["score"],
                "detail": ev["detail"],
            })
            if ev["correct"]:
                correct += 1
            score_sum += ev["score"]
            total += 1

            logger.info(f"  [{tc['id']}] {tc['target']}: {'✓' if ev['correct'] else '✗'} ({ev['match_level']}) tools={len(tool_calls_made)}")

        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1
            logger.error(f"  [{tc['id']}] ERROR: {e}")

        time.sleep(0.5)

    return {
        "name": "single_agent_react",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_react_turns": MAX_REACT_TURNS,
        "tools": ["translate", "rag_search", "lookup"],
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }


# ── Multi-Agent + ReAct（複用 Orchestrator，和 ablation_v2 完全一致）──
def _run_multi_agent_react(client: OpenAI, test_cases: list, lexicon: dict) -> dict:
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.core.rate_limiter import get_api_guard

    orchestrator = OrchestratorAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT, strategy_name="mastery_first",
    )
    orchestrator._ensure_agents()

    correct = 0
    total = 0
    score_sum = 0.0
    results = []

    for tc in test_cases:
        try:
            reply = orchestrator.chat(tc["input"])
            ev = evaluate_v2(reply, tc["target"], lexicon)

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

            logger.info(f"  [{tc['id']}] {tc['target']}: {'✓' if ev['correct'] else '✗'} ({ev['match_level']})")

        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1
            logger.error(f"  [{tc['id']}] ERROR: {e}")

        time.sleep(0.5)

    return {
        "name": "multi_agent_react",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "accuracy": correct / total if total else 0,
        "avg_score": score_sum / total if total else 0,
        "correct": correct, "total": total,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }


# ── Main ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both",
                        help="跑哪個配置")
    parser.add_argument("--test-count", type=int, default=0,
                        help="測試題數（0=全部 20 題）")
    args = parser.parse_args()

    client = OpenAI(
        api_key=API_KEY or "dummy",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    lexicon = load_lexicon()
    benchmark = load_benchmark()
    test_cases = benchmark[:args.test_count] if args.test_count else benchmark

    logger.info(f"=== 致命實驗：Single Agent + ReAct vs Multi-Agent + ReAct ===")
    logger.info(f"測試集：{len(test_cases)} 題")
    logger.info(f"模型：{MODEL}")

    output = {"metadata": {
        "experiment": "critical_single_vs_multi",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "test_count": len(test_cases),
        "description": "控制變因實驗：Single Agent + ReAct (translate/rag_search/lookup) vs Multi-Agent + ReAct (Orchestrator 全工具)",
    }}

    if args.mode in ("single", "both"):
        logger.info("\n>>> Running: Single Agent + ReAct")
        output["single_agent_react"] = _run_single_agent_react(client, test_cases, lexicon)

    if args.mode in ("multi", "both"):
        logger.info("\n>>> Running: Multi-Agent + ReAct")
        output["multi_agent_react"] = _run_multi_agent_react(client, test_cases, lexicon)

    # 存結果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "experiment_results" / f"critical_experiment_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 摘要
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    if "single_agent_react" in output:
        s = output["single_agent_react"]
        logger.info(f"Single Agent + ReAct: {s['correct']}/{s['total']} = {s['accuracy']*100:.1f}%")
    if "multi_agent_react" in output:
        m = output["multi_agent_react"]
        logger.info(f"Multi-Agent + ReAct:  {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")

    if "single_agent_react" in output and "multi_agent_react" in output:
        sa = output["single_agent_react"]["accuracy"]
        ma = output["multi_agent_react"]["accuracy"]
        diff = ma - sa
        logger.info(f"\n差異: {diff*100:+.1f}pp")
        if abs(diff) < 0.05:
            logger.info("→ 結論：提升來自 ReAct，不是 Multi-Agent Decomposition")
        elif diff > 0.1:
            logger.info("→ 結論：Multi-Agent Decomposition 有顯著額外貢獻！")
        else:
            logger.info("→ 結論：Multi-Agent 有小幅額外貢獻")

    logger.info(f"\n結果已存：{out_path}")
