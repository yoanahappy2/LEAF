#!/usr/bin/env python3
"""
補充消融實驗 v2：
1. ReAct error diagnosis — 分類 SA+ReAct 的錯誤
2. Multi-Agent w/o Quality Agent (去掉 review_translation + _verify_translations)
3. Multi-Agent w/o Orchestrator prompt (用 SA prompt 但保留架構)

目的：找出 95% vs 65% 的 30pp 差異來源
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
TEMPERATURE = 0.3

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


# ── 1. ReAct Error Diagnosis ──
def diagnose_react_errors():
    """分析 SA+ReAct 的錯誤類型"""
    d = json.load(open(PROJECT_ROOT / "experiment_results" / "critical_experiment_20260608_172611.json"))
    ma = json.load(open(PROJECT_ROOT / "experiment_results" / "ablation_v2_20260605_152910.json"))
    
    sa_react = {r['id']: r for r in d['single_agent_react']['results']}
    sa_no_react = {r['id']: r for r in ma['single_agent']['results']}
    
    errors = []
    for i in range(1, 21):
        qid = f'q{i:02d}'
        sr = sa_react.get(qid, {})
        snr = sa_no_react.get(qid, {})
        
        if sr.get('correct') and not snr.get('correct'):
            errors.append({"id": qid, "target": sr.get('target',''), "type": "reReact_saved", 
                          "tools": sr.get('tool_calls', [])})
        elif not sr.get('correct') and snr.get('correct'):
            tool_calls = sr.get('tool_calls', [])
            # 分類錯誤
            error_type = "unknown"
            for tc in tool_calls:
                if 'direction' not in tc and len(tc) > 30:
                    error_type = "query_drift"
                elif '的' in tc or '？' in tc or '怎麼說' in tc or '什麼' in tc:
                    error_type = "query_drift"
            
            errors.append({"id": qid, "target": sr.get('target',''), "type": error_type,
                          "sa_react_tools": tool_calls, "reply": sr.get('reply','')[:200]})
    
    return errors


# ── 2. Multi-Agent w/o Quality Agent ──
def run_multi_agent_no_quality(client, test_cases, lexicon):
    """Multi-Agent 但去掉 review_translation 工具和 _verify_translations"""
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.core.rate_limiter import get_api_guard
    
    # 建構不含 review_translation 的 tools
    # 這需要 monkey-patch orchestrator 的 _get_agent_tools 和 _verify_translations
    orch = OrchestratorAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT, strategy_name="mastery_first",
    )
    orch._ensure_agents()
    
    # Monkey-patch: 移除 review_translation
    original_get_tools = orch._get_agent_tools
    def get_tools_no_review():
        tools = original_get_tools()
        return [t for t in tools if t["function"]["name"] != "review_translation"]
    orch._get_agent_tools = get_tools_no_review
    
    # Monkey-patch: 跳過 _verify_translations
    orch._verify_translations = lambda reply, msgs: reply
    
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
    
    return {"name": "multi_no_quality", "correct": correct, "total": total,
            "accuracy": correct/total if total else 0, "results": results}


# ── 3. Multi-Agent 但用 SA 的 system prompt ──
def run_multi_agent_sa_prompt(client, test_cases, lexicon):
    """Multi-Agent 架構但把 Orchestrator prompt 換成 SA 的簡單 prompt"""
    from agent_framework.agents.orchestrator import OrchestratorAgent
    from agent_framework.core.rate_limiter import get_api_guard
    
    orch = OrchestratorAgent(
        client=client, api_guard=get_api_guard(),
        project_root=PROJECT_ROOT, strategy_name="mastery_first",
    )
    orch._ensure_agents()
    
    # Monkey-patch: 替換 system prompt 為簡單版
    simple_prompt = (
        "你是排灣語翻譯助手。用戶會問你中文詞彙的排灣語怎麼說。\n"
        "你必須使用工具來翻譯，不要自己猜。\n"
        "如果第一次翻譯失敗或結果不確定，嘗試用不同方式搜索。\n"
        "最終只輸出排灣語詞彙本身，不要加解釋。"
    )
    orch._get_cached_prompt = lambda: simple_prompt
    
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
    
    return {"name": "multi_sa_prompt", "correct": correct, "total": total,
            "accuracy": correct/total if total else 0, "results": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["diagnose", "no_quality", "sa_prompt", "all"], default="all")
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
    
    if args.exp in ("diagnose", "all"):
        logger.info("=== 1. ReAct Error Diagnosis ===")
        errors = diagnose_react_errors()
        for e in errors:
            logger.info(f"  {e['id']} {e['target']}: {e['type']}")
        output["react_error_diagnosis"] = errors
    
    if args.exp in ("no_quality", "all"):
        logger.info("\n=== 2. Multi-Agent w/o Quality Agent ===")
        result = run_multi_agent_no_quality(client, benchmark, lexicon)
        logger.info(f"Result: {result['correct']}/{result['total']} = {result['accuracy']*100:.0f}%")
        output["multi_no_quality"] = result
    
    if args.exp in ("sa_prompt", "all"):
        logger.info("\n=== 3. Multi-Agent with SA prompt ===")
        result = run_multi_agent_sa_prompt(client, benchmark, lexicon)
        logger.info(f"Result: {result['correct']}/{result['total']} = {result['accuracy']*100:.0f}%")
        output["multi_sa_prompt"] = result
    
    # 存結果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "experiment_results" / f"ablation_v3_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # 摘要
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION v3 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Reference: SA no-ReAct = 80%, SA+ReAct = 65%, Multi-Agent = 95%")
    if "multi_no_quality" in output:
        r = output["multi_no_quality"]
        logger.info(f"Multi-Agent w/o Quality: {r['correct']}/{r['total']} = {r['accuracy']*100:.0f}%")
    if "multi_sa_prompt" in output:
        r = output["multi_sa_prompt"]
        logger.info(f"Multi-Agent w/ SA prompt: {r['correct']}/{r['total']} = {r['accuracy']*100:.0f}%")
    if "react_error_diagnosis" in output:
        errors = output["react_error_diagnosis"]
        query_drift = [e for e in errors if e['type'] == 'query_drift']
        logger.info(f"ReAct errors: {len(errors)} total, {len(query_drift)} query_drift")
    
    logger.info(f"\nSaved: {out_path}")
