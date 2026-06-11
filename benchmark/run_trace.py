#!/usr/bin/env python3
"""
LEAF Trace — 跑指定題目，捕獲 Multi-Agent 完整 ReAct trace
"""
import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

from openai import OpenAI
from agent_framework.agents.orchestrator import OrchestratorAgent
from agent_framework.core.rate_limiter import get_api_guard
from agent_framework.core.message import AgentMessage, MessageType
from agent_framework.core.decision import Decision, DecisionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 載入 benchmark
benchmark = json.load(open(PROJECT_ROOT / "benchmark" / "benchmark_v2.json"))
lexicon = json.load(open(PROJECT_ROOT / "benchmark" / "lexicon_v2.json"))
lexicon.pop("_meta", None)

# API
api_key = os.environ.get("ZHIPUAI_API_KEY")
if not api_key:
    for line in open(PROJECT_ROOT / ".env"):
        if line.startswith("ZHIPUAI_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")


def evaluate(pred: str, target: str) -> dict:
    entry = lexicon.get(target)
    if not entry:
        return {"correct": False, "score": 0.0}
    variants = entry["variants"]
    preferred = entry["preferred"]
    candidates = re.findall(r'[a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+', pred.strip().lower())
    candidates = [c for c in candidates if len(c) >= 2]
    matched = None
    for c in candidates:
        if c in variants:
            matched = c
            break
    # 也檢查 markdown bold
    if not matched:
        bold_words = re.findall(r'\*\*([a-zA-Záéíóúàèìòùâêîôûäëïöüāīū]+)\*\*', pred)
        for bw in bold_words:
            if bw in variants:
                matched = bw
                break
    if matched is None:
        return {"correct": False, "score": 0.0, "matched": None}
    elif matched == preferred:
        return {"correct": True, "score": 1.0, "matched": matched}
    else:
        return {"correct": True, "score": 0.8, "matched": matched}


def run_trace(questions: list):
    """跑 Multi-Agent，捕獲完整 trace"""
    orchestrator = OrchestratorAgent(
        client=client,
        api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
        strategy_name="mastery_first",
    )
    orchestrator._ensure_agents()

    traces = []

    for tc in questions:
        qid = tc["id"]
        target = tc["target"]
        user_input = tc["input"]

        trace = {
            "question_id": qid,
            "input": user_input,
            "target": target,
            "accepted_answers": lexicon.get(target, {}).get("variants", []),
            "preferred": lexicon.get(target, {}).get("preferred", ""),
            "start_time": datetime.now().isoformat(),
            "turns": [],
        }

        # Monkey-patch MessageBus.send to intercept all messages
        original_send = orchestrator.bus.send
        intercepted = []
        
        def capture_send(msg):
            entry = {
                "from": msg.from_agent,
                "to": msg.to_agent,
                "type": msg.type if isinstance(msg.type, str) else msg.type.value,
                "payload": msg.payload,
            }
            intercepted.append(entry)
            resp = original_send(msg)
            if resp:
                intercepted.append({
                    "from": resp.from_agent,
                    "to": resp.to_agent,
                    "type": resp.type if isinstance(resp.type, str) else resp.type.value,
                    "payload": resp.payload,
                })
            return resp
        
        orchestrator.bus.send = capture_send
        
        # 執行
        intercepted.clear()
        t0 = time.time()
        try:
            reply = orchestrator.chat(user_input)
            elapsed = time.time() - t0
        except Exception as e:
            reply = f"ERROR: {e}"
            elapsed = time.time() - t0

        trace["reply"] = str(reply)[:500]
        trace["elapsed_seconds"] = round(elapsed, 1)
        trace["intercepted_messages"] = intercepted
        
        # Decision Log
        if hasattr(orchestrator, 'decision_log') and orchestrator.decision_log:
            decisions = []
            if isinstance(orchestrator.decision_log, list):
                decisions = [
                    {
                        "agent": getattr(d, 'agent', ''),
                        "task": getattr(d, 'task', ''),
                        "situation": getattr(d, 'situation', ''),
                        "options": getattr(d, 'options', []),
                        "chosen": getattr(d, 'chosen', ''),
                        "reasoning": getattr(d, 'reasoning', ''),
                        "confidence": getattr(d, 'confidence', 0),
                    }
                    for d in orchestrator.decision_log
                ]
            trace["decisions"] = decisions
        
        # Evaluation
        ev = evaluate(str(reply), target)
        trace["evaluation"] = ev
        
        trace["end_time"] = datetime.now().isoformat()
        traces.append(trace)

        # 恢復
        orchestrator.bus.send = original_send

        print(f"  {qid} {target}: {'✓' if ev['correct'] else '✗'} score={ev['score']} turns={len(intercepted)//2} time={elapsed:.1f}s")

    return traces


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", default=["q14", "q16", "q19", "q11", "q12", "q18"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Filter questions
    target_ids = args.ids
    questions = [tc for tc in benchmark if tc["id"] in target_ids]
    
    print(f"Running trace for {len(questions)} questions: {[q['id'] for q in questions]}")
    
    traces = run_trace(questions)
    
    out_path = args.output or str(PROJECT_ROOT / "benchmark" / "multi_agent_traces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {out_path}")
    
    # Summary
    for t in traces:
        ev = t["evaluation"]
        n_msgs = len(t.get("intercepted_messages", []))
        print(f"\n{'='*60}")
        print(f"  {t['question_id']} 「{t['target']}」→ {t['preferred']}")
        print(f"  Result: {'✓' if ev['correct'] else '✗'} score={ev.get('score',0)} matched={ev.get('matched','')}")
        print(f"  Messages: {n_msgs} | Time: {t['elapsed_seconds']}s")
        print(f"  Reply: {t['reply'][:120]}")
        for msg in t.get("intercepted_messages", []):
            p = msg.get("payload", {})
            task = p.get("task", "")
            params = p.get("params", {})
            if msg.get("to") in ["knowledge", "teaching", "quality"]:
                print(f"    {msg['from']} → {msg['to']}: {task}({json.dumps(params, ensure_ascii=False)[:80]})")
            elif msg.get("from") in ["knowledge", "teaching", "quality"]:
                status = p.get("status", "")
                data_preview = str(p.get("data", ""))[:60]
                print(f"    {msg['from']} → {msg['to']}: {status} data={data_preview}")
