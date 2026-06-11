#!/usr/bin/env python3
"""
SA + Constrained ReAct：禁止改寫 query
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

# Tools schema（直接從 Orchestrator 複製，只用翻譯相關的 3 個）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "排灣語⇄中文雙向翻譯",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要翻譯的文字"},
                    "direction": {"type": "string", "enum": ["auto", "p2c", "c2p"], "default": "auto"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "搜尋排灣語知識庫，找例句和語料",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜尋關鍵詞"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "查詢排灣語詞彙深度資訊（綴詞分析、親屬關係、相關詞）",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string", "description": "要查的詞"}},
                "required": ["word"],
            },
        },
    },
]

CONSTRAINED_PROMPT = (
    "你是排灣語翻譯助手。\n"
    "規則：\n"
    "1. 用戶給你一個中文詞，你必須用 translate 工具翻譯\n"
    "2. **嚴禁改寫、擴展用戶的查詢**——直接把用戶的原文作為 text 參數\n"
    "3. 如果 translate 返回空結果，可以用 rag_search，但 query 也必須是原始詞\n"
    "4. 不要在 query 中添加「的族語怎麼說」「是什麼」等後綴\n"
    "5. 直接輸出排灣語詞彙，不要加解釋\n"
)


def run_constrained(client, test_cases, lexicon):
    from translate_service import PaiwanTranslator
    translator = PaiwanTranslator()
    
    correct = 0; total = 0; results = []
    for tc in test_cases:
        try:
            messages = [
                {"role": "system", "content": CONSTRAINED_PROMPT},
                {"role": "user", "content": tc["input"]},
            ]
            
            final_reply = ""
            tool_calls_log = []
            for step in range(5):
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    temperature=0.3,
                    max_tokens=500,
                )
                msg = resp.choices[0].message
                
                if not msg.tool_calls:
                    final_reply = msg.content or ""
                    break
                
                # Build assistant message
                asst_msg = {"role": "assistant", "content": msg.content or ""}
                asst_msg["tool_calls"] = [
                    {"id": tc_c.id, "type": "function", 
                     "function": {"name": tc_c.function.name, "arguments": tc_c.function.arguments}}
                    for tc_c in msg.tool_calls
                ]
                messages.append(asst_msg)
                
                tool_results = []
                for tc_call in msg.tool_calls:
                    fn_name = tc_call.function.name
                    fn_args = json.loads(tc_call.function.arguments)
                    tool_calls_log.append(f'{fn_name}({json.dumps(fn_args, ensure_ascii=False)})')
                    logger.info(f"  [{tc['id']}] tool: {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")
                    
                    if fn_name == "translate":
                        result = translator.translate(fn_args.get("text", ""), direction=fn_args.get("direction", "c2p"))
                        tool_result = json.dumps({
                            "input": result["input"], 
                            "translation": result["translation"], 
                            "method": result.get("method", "")
                        }, ensure_ascii=False)
                    elif fn_name == "rag_search":
                        results_rag = translator.rag_search(fn_args.get("query", ""), top_k=fn_args.get("top_k", 5))
                        tool_result = json.dumps([
                            {"paiwan": r.get("paiwan",""), "chinese": r.get("chinese","")} 
                            for r in results_rag
                        ], ensure_ascii=False)
                    elif fn_name == "lookup":
                        lookup_result = translator.lookup(fn_args.get("word", ""))
                        tool_result = json.dumps(lookup_result, ensure_ascii=False) if lookup_result else "{}"
                    else:
                        tool_result = "{}"
                    
                    tool_results.append({
                        "role": "tool", 
                        "tool_call_id": tc_call.id, 
                        "content": tool_result
                    })
                
                messages.extend(tool_results)
            
            if not final_reply:
                final_reply = messages[-1].get("content", "") if messages[-1]["role"] == "tool" else ""
                # 需要再調一次 LLM 生成最終回覆
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=200,
                )
                final_reply = resp.choices[0].message.content or ""
            
            ev = evaluate_v2(final_reply, tc["target"], lexicon)
            if not ev["correct"]:
                for bw in re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', final_reply):
                    ev2 = evaluate_v2(bw, tc["target"], lexicon)
                    if ev2["correct"]: ev = ev2; break
            
            results.append({
                "id": tc["id"], "target": tc["target"], "correct": ev["correct"],
                "match_level": ev["match_level"], "tool_calls": tool_calls_log,
                "reply": final_reply[:200]
            })
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
    API_KEY = os.environ.get("ZHIPUAI_API_KEY", "")
    if not API_KEY:
        for line in open(PROJECT_ROOT / ".env"):
            if line.startswith("ZHIPUAI_API_KEY="):
                API_KEY = line.strip().split("=", 1)[1]; break
    
    client = OpenAI(api_key=API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")
    lexicon = load_lexicon()
    benchmark = load_benchmark()
    
    logger.info("=== SA + Constrained ReAct ===")
    result = run_constrained(client, benchmark, lexicon)
    logger.info(f"Result: {result['correct']}/{result['total']} = {result['accuracy']*100:.0f}%")
    
    # Differential with free ReAct
    d = json.load(open(PROJECT_ROOT / "experiment_results" / "critical_experiment_20260608_172611.json"))
    sa_react = {r['id']: r for r in d['single_agent_react']['results']}
    logger.info("\n=== Differential: Free ReAct vs Constrained ReAct ===")
    for q in result['results']:
        sr = sa_react.get(q['id'], {})
        if q['correct'] != sr.get('correct', None):
            logger.info(f"  {q['id']} {q['target']}: constrained={'✓' if q['correct'] else '✗'}, free={'✓' if sr.get('correct') else '✗'}")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "experiment_results" / f"constrained_react_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"timestamp": datetime.now().isoformat(), "model": MODEL},
                   "sa_constrained_react": result}, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved: {out_path}")
