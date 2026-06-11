"""
demo_multi_agent.py — Multi-Agent 系統完整 Demo

展示三種長程任務模式 + 4 個 Agent 協作過程。
產出 JSON 報告 + 可讀的 Markdown 日誌。

作者: 地陪
日期: 2026-05-30
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "agent_framework"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("demo")

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

from agents.orchestrator import OrchestratorAgent
from core.rate_limiter import get_api_guard

# ============================================
# Demo 配置
# ============================================

DEMO_SCENARIOS = [
    # === Phase 1: ReAct（Orchestrator 自主決定）===
    {
        "phase": "ReAct",
        "description": "Orchestrator 根據用戶輸入自主分派 Agent",
        "inputs": [
            "你好嗎？排灣語怎麼說",
            "教我一個排灣語的動物詞",
            "謝謝用排灣語怎麼說？什麼時候用？",
        ],
    },
    # === Phase 2: Quality 審核 ===
    {
        "phase": "Quality_Review",
        "description": "Quality Agent 審核翻譯品質",
        "inputs": [
            "審核翻譯：原句「你好」→ 翻譯「djavadjavay」",
            "審核翻譯：原句「我愛你」→ 翻譯「tjengelay aken tjanusun」",
        ],
    },
    # === Phase 3: 教學推薦 ===
    {
        "phase": "Teaching",
        "description": "Teaching Agent 推薦學習詞彙和生成測驗",
        "inputs": [
            "我剛學了 masalu 和 djavadjavay，推薦下一個",
            "幫我出一道排灣語測驗題",
        ],
    },
]

def run_demo():
    """執行完整 Demo"""
    
    print("=" * 70)
    print("  🎬 Multi-Agent 系統 Demo")
    print(f"  時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    api_guard = get_api_guard()
    
    # 初始化 Orchestrator（會自動註冊 Knowledge/Teaching/Quality Agent）
    logger.info("初始化 Multi-Agent 系統...")
    orch = OrchestratorAgent(
        client=client,
        api_guard=api_guard,
        project_root=Path(__file__).parent,
    )
    orch._ensure_agents()
    
    registered = [k for k, v in orch.bus._agents.items() if v]
    logger.info(f"已註冊 Agent: {registered}")
    print(f"\n✅ Agent 已啟動: {', '.join(registered)}\n")
    
    # 記錄
    all_results = []
    total_tokens_start = api_guard.get_total_tokens_used()
    
    for phase_idx, scenario in enumerate(DEMO_SCENARIOS):
        phase = scenario["phase"]
        desc = scenario["description"]
        inputs = scenario["inputs"]
        
        print(f"\n{'═' * 70}")
        print(f"  📌 Phase {phase_idx+1}: {phase}")
        print(f"  {desc}")
        print(f"{'═' * 70}\n")
        
        phase_results = {
            "phase": phase,
            "description": desc,
            "interactions": [],
        }
        
        for i, user_input in enumerate(inputs):
            print(f"  [{i+1}/{len(inputs)}] 用戶: 「{user_input}」")
            
            start = time.time()
            try:
                reply = orch.chat(user_input, user_id="demo_user")
                elapsed = (time.time() - start) * 1000
                
                print(f"  🤖 系統: {reply[:200]}")
                if len(reply) > 200:
                    print(f"         ...（共 {len(reply)} 字）")
                print(f"  ⏱️ {elapsed:.0f}ms")
                
                phase_results["interactions"].append({
                    "input": user_input,
                    "reply": reply,
                    "elapsed_ms": round(elapsed),
                    "status": "success",
                })
                
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"  ❌ 錯誤: {e}")
                
                phase_results["interactions"].append({
                    "input": user_input,
                    "error": str(e),
                    "elapsed_ms": round(elapsed),
                    "status": "error",
                })
            
            print()
            time.sleep(1)  # 避免 API 限速
        
        all_results.append(phase_results)
    
    # 統計
    total_tokens = api_guard.get_total_tokens_used()
    tokens_used = total_tokens - total_tokens_start
    
    print(f"\n{'═' * 70}")
    print(f"  📊 Demo 完成")
    print(f"{'═' * 70}")
    print(f"  Token 消耗: {tokens_used:,}")
    print(f"  API 調用: {api_guard.get_usage_summary().get('total_requests', '?')}")
    
    # 保存 JSON 報告
    report = {
        "timestamp": datetime.now().isoformat(),
        "agents": registered,
        "phases": all_results,
        "summary": {
            "tokens_used": tokens_used,
            "total_interactions": sum(len(p["interactions"]) for p in all_results),
            "successful": sum(1 for p in all_results for i in p["interactions"] if i["status"] == "success"),
            "errors": sum(1 for p in all_results for i in p["interactions"] if i["status"] == "error"),
        },
    }
    
    report_path = Path(__file__).parent / "results" / "multi_agent_demo.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 報告: {report_path}")
    
    # 保存 Markdown 日誌
    md_lines = [
        f"# Multi-Agent 系統 Demo 報告",
        f"",
        f"**時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Agent**: {', '.join(registered)}",
        f"**Token 消耗**: {tokens_used:,}",
        f"**成功率**: {report['summary']['successful']}/{report['summary']['total_interactions']}",
        f"",
    ]
    
    for phase in all_results:
        md_lines.append(f"## {phase['phase']}: {phase['description']}")
        md_lines.append("")
        for inter in phase["interactions"]:
            status = "✅" if inter["status"] == "success" else "❌"
            md_lines.append(f"### {status} 用戶: {inter['input']}")
            md_lines.append("")
            if inter["status"] == "success":
                md_lines.append(f"**系統回覆** ({inter['elapsed_ms']}ms):")
                md_lines.append(f"> {inter['reply']}")
            else:
                md_lines.append(f"**錯誤**: {inter.get('error', 'unknown')}")
            md_lines.append("")
    
    md_path = Path(__file__).parent / "results" / "multi_agent_demo.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  📝 Markdown: {md_path}")
    
    return report


if __name__ == "__main__":
    run_demo()
