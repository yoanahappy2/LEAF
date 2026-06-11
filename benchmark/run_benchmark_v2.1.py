#!/usr/bin/env python3
"""
LEAF Benchmark v2.1 — Multi-Agent Collaboration Verification

核心目標：驗證 TeachingAgent + QualityAgent + KnowledgeAgent 是否真的被觸發
不是 accuracy，而是 agent_invocation

用法：
    python3 benchmark/run_benchmark_v2.1.py
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework"))

from openai import OpenAI
from agent_framework.agents.orchestrator import OrchestratorAgent
from agent_framework.core.rate_limiter import get_api_guard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_api_client():
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        for line in open(PROJECT_ROOT / ".env"):
            if line.startswith("ZHIPUAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
    return OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4")


def run_multi_agent_task(client, task):
    """跑一個 Multi-Agent task，捕獲所有 agent 調用"""
    orchestrator = OrchestratorAgent(
        client=client,
        api_guard=get_api_guard(),
        project_root=PROJECT_ROOT,
        strategy_name="mastery_first",
    )
    orchestrator._ensure_agents()

    # Intercept MessageBus.send
    agents_invoked = set()
    tools_called = []
    messages_log = []
    original_send = orchestrator.bus.send

    def capture_send(msg):
        to_agent = msg.to_agent
        task_name = msg.payload.get("task", "")
        params = msg.payload.get("params", {})

        if to_agent in ["knowledge", "teaching", "quality"]:
            agents_invoked.add(to_agent)
            tools_called.append({"agent": to_agent, "task": task_name, "params": params})
            messages_log.append({
                "direction": "dispatch",
                "from": msg.from_agent,
                "to": to_agent,
                "task": task_name,
                "params": params,
            })

        resp = original_send(msg)

        if resp and to_agent in ["knowledge", "teaching", "quality"]:
            payload = resp.payload or {}
            messages_log.append({
                "direction": "response",
                "from": resp.from_agent,
                "to": resp.to_agent,
                "status": payload.get("status", ""),
                "data_preview": str(payload.get("data", ""))[:100],
            })

        return resp

    orchestrator.bus.send = capture_send

    # Execute
    t0 = time.time()
    try:
        reply = orchestrator.chat(task["input"])
        elapsed = time.time() - t0
        error = None
    except Exception as e:
        reply = f"ERROR: {e}"
        elapsed = time.time() - t0
        error = str(e)

    orchestrator.bus.send = original_send

    # Evaluate
    expected_agents = set(task.get("expected_agents", []))
    agents_ok = expected_agents.issubset(agents_invoked)

    return {
        "id": task["id"],
        "type": task["type"],
        "input": task["input"],
        "reply": str(reply)[:500],
        "elapsed_seconds": round(elapsed, 1),
        "agents_invoked": sorted(agents_invoked),
        "agents_expected": sorted(expected_agents),
        "agents_ok": agents_ok,
        "tools_called": tools_called,
        "messages_log": messages_log,
        "error": error,
    }


if __name__ == "__main__":
    client = get_api_client()
    benchmark = json.load(open(PROJECT_ROOT / "benchmark" / "benchmark_v2.1.json"))
    tasks = benchmark["tasks"]

    print("=" * 60)
    print("  🧪 LEAF Benchmark v2.1 — Multi-Agent Collaboration Verification")
    print("=" * 60)
    print(f"  Tasks: {len(tasks)}")
    print()

    results = []
    for task in tasks:
        print(f"  Running {task['id']} ({task['type']})...")
        result = run_multi_agent_task(client, task)
        results.append(result)

        status = "✅ PASS" if result["agents_ok"] else "❌ FAIL"
        print(f"  {task['id']}: {status}")
        print(f"    Expected: {result['agents_expected']}")
        print(f"    Invoked:  {result['agents_invoked']}")
        tool_seq = [t['agent'] + ':' + t['task'] for t in result['tools_called']]
        print(f"    Tools:    {tool_seq}")
        print(f"    Reply:    {result['reply'][:120]}")
        print(f"    Time:     {result['elapsed_seconds']}s")
        print()

    # Summary
    passed = sum(1 for r in results if r["agents_ok"])
    total = len(results)

    print("=" * 60)
    print(f"  📊 Summary: {passed}/{total} tasks triggered all expected agents")
    print("=" * 60)

    # Agent coverage
    all_agents = set()
    for r in results:
        all_agents.update(r["agents_invoked"])
    print(f"  Agents used: {sorted(all_agents)}")
    print(f"  Knowledge: {sum(1 for r in results if 'knowledge' in r['agents_invoked'])}/{total}")
    print(f"  Teaching:  {sum(1 for r in results if 'teaching' in r['agents_invoked'])}/{total}")
    print(f"  Quality:   {sum(1 for r in results if 'quality' in r['agents_invoked'])}/{total}")

    # Save
    out_dir = PROJECT_ROOT / "experiment_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"multi_agent_v21_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 Saved: {out_path}")
