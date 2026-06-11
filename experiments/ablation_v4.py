#!/usr/bin/env python3
"""
補充實驗 v4：
1. Multi-Agent w/o pre-routing — 純粹測 routing 的貢獻
2. Constrained ReAct — SA 但禁止改寫 query
"""

import os, sys, re, json, time, logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL = "glm-4-flash"

def load_lexicon():
    with open(PROJECT_ROOT / "benchmark" / "lexicon_v2.json", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}

def load_benchmark():
    with open(PROJECT_ROOT / "benchmark" / "benchmark_v2.json", encoding="utf-8") as f:
        return json.load(f)

def evaluate_v2(pred, target, lexicon):
    entry = lexicon.get(target)
    if not entry:
        return {"match_level": "error", "correct": False, "score": 0.0}
    variants = entry["variants"]
    preferred = entry["preferred"]
    candidates = [c for c in re.findall(r'[a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+', pred.strip().lower()) if len(c) >= 2]
    matched = None
    for c in candidates:
        if c in variants:
            matched = c; break
    if matched is None:
        return {"match_level": "none", "correct": False, "score": 0.0, "candidates": candidates[:5]}
    elif matched == preferred:
        return {"match_level": "exact", "correct": True, "score": 1.0, "matched": matched}
    else:
        return {"match_level": "variant", "correct": True, "score": 0.8, "matched": matched}


# ── 1. Multi-Agent w/o pre-routing ──
def run_multi_agent_no_routing(client, test_cases, lexicon):
    """Multi-Agent 但去掉 _keyword_pre_route"""
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.core.rate_limiter import get_api_guard
    
    orch = OrchestratorAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT, strategy_name="mastery_first",
    )
    orch._ensure_agents()
    
    # Monkey-patch: 跳過 keyword pre-routing
    orch._keyword_pre_route = lambda user_input, user_id: []
    
    correct = 0; total = 0; results = []
    for tc in test_cases:
        try:
            reply = orch.chat(tc["input"])
            ev = evaluate_v2(reply, tc["target"], lexicon)
            if not ev["correct"]:
                for bw in re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', reply):
                    ev2 = evaluate_v2(bw, tc["target"], lexicon)
                    if ev2["correct"]: ev = ev2; break
            results.append({"id": tc["id"], "target": tc["target"], "correct": ev["correct"],
                          "match_level": ev["match_level"], "reply": str(reply)[:200]})
            if ev["correct"]: correct += 1
            total += 1
            logger.info(f"  [{tc['id']}] {tc['target']}: {'✓' if ev['correct'] else '✗'}")
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1
        time.sleep(0.5)
    
    return {"name": "multi_no_routing", "correct": correct, "total": total,
            "accuracy": correct/total if total else 0, "results": results}


# ── 2. Constrained ReAct (SA, keyword-only) ──
def run_sa_constrained_react(client, test_cases, lexicon):
    """Single Agent + ReAct，但 system prompt 禁止改寫 query"""
    from agent_framework.agents.knowledge_agent import KnowledgeAgent
    from agent_framework.core.rate_limiter import get_api_guard
    
    ka = KnowledgeAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
    )
    ka._ensure_services()
    
    # 取得原本的 tools
    tools = ka._get_tools()
    
    CONSTRAINED_PROMPT = (
        "你是排灣語翻譯助手。\n"
        "規則：\n"
        "1. 用戶給你一個中文詞，你必須用 translate 工具翻譯\n"
        "2. **嚴禁改寫、擴展、改寫用戶的查詢**——直接把用戶的原文作為 text 參數\n"
        "3. 如果 translate 返回空結果，可以用 rag_search，但 query 也必須是原始詞\n"
        "4. 不要在 query 中添加「的族語怎麼說」「是什麼」等後綴\n"
        "5. 直接輸出排灣語詞彙\n"
    )
    
    correct = 0; total = 0; results = []
    for tc in test_cases:
        try:
            messages = [
                {"role": "system", "content": CONSTRAINED_PROMPT},
                {"role": "user", "content": tc["input"]},
            ]
            
            for step in range(5):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    temperature=0.3,
                    max_tokens=500,
                )
                msg = resp.choices[0].message
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc_d for tc_d in ([msg.tool_calls] if msg.tool_calls else [])] if False else None})
                
                if not msg.tool_calls:
                    # 沒有工具調用，直接評估
                    break
                
                tool_results = []
                for tc_call in msg.tool_calls:
                    fn_name = tc_call.function.name
                    fn_args = json.loads(tc_call.function.arguments)
                    
                    # 記錄 tool call
                    logger.info(f"  [{tc['id']}] tool: {fn_name}({fn_args})")
                    
                    if fn_name == "translate":
                        result = ka._translator.translate(fn_args.get("text", ""), direction=fn_args.get("direction", "c2p"))
                        tool_result = json.dumps({"input": result["input"], "translation": result["translation"], "method": result.get("method", "")}, ensure_ascii=False)
                    elif fn_name == "rag_search":
                        results_rag = ka._translator.rag_search(fn_args.get("query", ""), top_k=fn_args.get("top_k", 5))
                        tool_result = json.dumps([{"paiwan": r.get("paiwan",""), "chinese": r.get("chinese","")} for r in results_rag], ensure_ascii=False)
                    elif fn_name == "lookup":
                        lookup_result = ka._translator.lookup(fn_args.get("word", ""))
                        tool_result = json.dumps(lookup_result, ensure_ascii=False) if lookup_result else "{}"
                    else:
                        tool_result = "{}"
                    
                    tool_results.append({"role": "tool", "tool_call_id": tc_call.id, "content": tool_result})
                
                messages.extend(tool_results)
                # 修正 assistant message
                messages[-len(tool_results)-1] = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{"id": tc_c.id, "type": "function", "function": {"name": tc_c.function.name, "arguments": tc_c.function.arguments}} for tc_c in msg.tool_calls]
                }
            
            final = messages[-1]["content"] if messages[-1]["role"] == "assistant" else str(messages[-1].get("content", ""))
            ev = evaluate_v2(final, tc["target"], lexicon)
            if not ev["correct"]:
                for bw in re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', final):
                    ev2 = evaluate_v2(bw, tc["target"], lexicon)
                    if ev2["correct"]: ev = ev2; break
            
            results.append({"id": tc["id"], "target": tc["target"], "correct": ev["correct"],
                          "match_level": ev["match_level"], "reply": final[:200]})
            if ev["correct"]: correct += 1
            total += 1
            logger.info(f"  [{tc['id']}] {tc['target']}: {'✓' if ev['correct'] else '✗'} ({ev['match_level']})")
        except Exception as e:
            results.append({"id": tc["id"], "target": tc["target"], "error": str(e)})
            total += 1
            logger.info(f"  [{tc['id']}] {tc['target']}: ERROR {e}")
        time.sleep(0.5)
    
    return {"name": "sa_constrained_react", "correct": correct, "total": total,
            "accuracy": correct/total if total else 0, "results": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["no_routing", "constrained", "all"], default="all")
    args = parser.parse_args()
    
    API_KEY = os.environ.get("ZHIPUAI_API_KEY", "")
    if not API_KEY:
        for line in open(PROJECT_ROOT / ".env"):
            if line.startswith("ZHIPUAI_API_KEY="):
                API_KEY = line.strip().split("=", 1)[1]; break
    
    client = OpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")
    lexicon = load_lexicon()
    benchmark = load_benchmark()
    
    output = {"metadata": {"timestamp": datetime.now().isoformat(), "model": MODEL}}
    
    if args.exp in ("no_routing", "all"):
        logger.info("\n=== Multi-Agent w/o pre-routing ===")
        result = run_multi_agent_no_routing(client, benchmark, lexicon)
        logger.info(f"Result: {result['correct']}/{result['total']} = {result['accuracy']*100:.0f}%")
        output["multi_no_routing"] = result
    
    if args.exp in ("constrained", "all"):
        logger.info("\n=== SA + Constrained ReAct ===")
        result = run_sa_constrained_react(client, benchmark, lexicon)
        logger.info(f"Result: {result['correct']}/{result['total']} = {result['accuracy']*100:.0f}%")
        output["sa_constrained_react"] = result
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "experiment_results" / f"ablation_v4_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION v4 SUMMARY")
    logger.info("=" * 60)
    logger.info("Reference: SA no-ReAct=80%, SA+ReAct=65%, Multi-Agent=95%")
    if "multi_no_routing" in output:
        r = output["multi_no_routing"]
        logger.info(f"Multi-Agent w/o routing: {r['correct']}/{r['total']} = {r['accuracy']*100:.0f}%")
    if "sa_constrained_react" in output:
        r = output["sa_constrained_react"]
        logger.info(f"SA + Constrained ReAct: {r['correct']}/{r['total']} = {r['accuracy']*100:.0f}%")
    logger.info(f"\nSaved: {out_path}")
