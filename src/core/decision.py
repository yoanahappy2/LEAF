"""
decision.py — Agent 自主決策記錄系統

每次 Agent 面臨多個選項時，記錄它的思考過程：
- 當前情境是什麼
- 有哪些可選行動
- 選了什麼
- 為什麼這樣選

這是展示 Multi-Agent 自主性的核心數據結構。
直接放海報、放 PPT、放論文。

作者: yu
日期: 2026-06-04
"""

import json
import time
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """
    單次 Agent 決策記錄

    使用方式（在 Agent 的 handle_message 或 react_loop 中）：

        decision = Decision(
            agent="teaching",
            situation="學生已學 5 詞，連續答對 3 次",
            options=["推薦新主題", "加深當前主題", "進行測驗"],
            chosen="推薦新主題",
            reasoning="學生掌握良好，可以拓展學習範圍",
            confidence=0.85,
        )
        decision.save()

    LLM 驅動模式（讓 LLM 自己輸出決策）：

        decision = Decision.from_llm(
            agent="teaching",
            situation="學生連錯 3 次",
            options=["繼續新單字", "回到舊單字複習", "提供提示"],
            llm_output=llm_response_text,
        )
    """
    # 身份
    id: str = field(default_factory=lambda: f"dec-{uuid.uuid4().hex[:8]}")
    agent: str = ""           # "orchestrator" | "knowledge" | "teaching" | "quality"

    # 決策內容
    situation: str = ""       # 當前狀態描述
    options: list = field(default_factory=list)    # 可選行動列表
    chosen: str = ""          # 選擇的行動
    reasoning: str = ""       # 為什麼這樣選
    confidence: float = 0.0   # 決策信心 0-1

    # 上下文
    task: str = ""            # 觸發此決策的任務
    task_params: dict = field(default_factory=dict)
    strategy: str = ""        # 當前使用的策略名稱
    language: str = "paiwan"  # 當前語言

    # 元數據
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    model: str = ""           # 使用的 LLM 模型

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent": self.agent,
            "situation": self.situation,
            "options": self.options,
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "task": self.task,
            "strategy": self.strategy,
            "language": self.language,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "model": self.model,
        }

    def to_json(self, indent: int = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_llm(cls, agent: str, situation: str, options: list[str],
                 llm_output: str, **kwargs) -> "Decision":
        """
        從 LLM 的自然語言輸出解析決策

        LLM 輸出格式（期望）：
        {
            "chosen": "回到舊單字複習",
            "reasoning": "學生遺忘率過高，優先鞏固已學內容",
            "confidence": 0.85
        }
        """
        try:
            # 嘗試解析 JSON
            content = llm_output.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())

            return cls(
                agent=agent,
                situation=situation,
                options=options,
                chosen=parsed.get("chosen", ""),
                reasoning=parsed.get("reasoning", ""),
                confidence=float(parsed.get("confidence", 0.5)),
                **kwargs,
            )
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失敗，把整段當 reasoning
            return cls(
                agent=agent,
                situation=situation,
                options=options,
                chosen=options[0] if options else "",
                reasoning=llm_output[:200],
                confidence=0.3,
                **kwargs,
            )


class DecisionLogger:
    """
    決策日誌管理器

    收集所有 Agent 的決策，持久化到 JSONL，
    提供查詢和統計接口。

    用法：
        logger = DecisionLogger()
        logger.log(decision)
        logger.get_decisions(agent="teaching")
        logger.export_for_poster()
    """

    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path(__file__).parent.parent / "storage" / "decisions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[Decision] = []
        self._flush_interval = 10  # 每 N 條決策 flush 一次

    def log(self, decision: Decision):
        """記錄一條決策"""
        self._buffer.append(decision)
        logging.getLogger(__name__).info(
            f"[Decision] {decision.agent}: {decision.chosen} "
            f"(confidence={decision.confidence:.2f}) — {decision.reasoning[:60]}"
        )
        if len(self._buffer) >= self._flush_interval:
            self.flush()

    def flush(self):
        """將緩衝區的決策寫入文件"""
        if not self._buffer:
            return

        log_file = self.storage_dir / f"decisions_{time.strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            for d in self._buffer:
                f.write(d.to_json() + "\n")

        self._buffer.clear()

    def get_decisions(self, agent: str = None, strategy: str = None,
                      language: str = None, limit: int = 100) -> list[Decision]:
        """查詢歷史決策"""
        decisions = []
        for f in sorted(self.storage_dir.glob("decisions_*.jsonl"), reverse=True):
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    d = Decision.from_dict(json.loads(line))
                    if agent and d.agent != agent:
                        continue
                    if strategy and d.strategy != strategy:
                        continue
                    if language and d.language != language:
                        continue
                    decisions.append(d)
                except (json.JSONDecodeError, TypeError):
                    continue
                if len(decisions) >= limit:
                    return decisions
        return decisions

    def get_stats(self) -> dict:
        """統計決策數據"""
        all_decisions = self.get_decisions(limit=10000)
        if not all_decisions:
            return {"total": 0}

        by_agent = {}
        by_strategy = {}
        confidences = []

        for d in all_decisions:
            by_agent[d.agent] = by_agent.get(d.agent, 0) + 1
            by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1
            confidences.append(d.confidence)

        return {
            "total": len(all_decisions),
            "by_agent": by_agent,
            "by_strategy": by_strategy,
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
        }

    def export_for_poster(self, output_path: Path = None, limit: int = 5) -> str:
        """
        導出最適合放海報的決策案例

        選擇標準：信心度高、reasoning 清晰、有教育意義
        """
        decisions = self.get_decisions(limit=100)

        # 按 confidence × reasoning 長度 排序（選最說服力強的）
        scored = []
        for d in decisions:
            score = d.confidence * 0.6 + min(len(d.reasoning) / 200, 1.0) * 0.4
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = [d for _, d in scored[:limit]]

        output = {
            "title": "LEAF Agent Decision Log — 自主決策示例",
            "description": "每個 Agent 面對多個選項時的思考過程",
            "examples": [d.to_dict() for d in best],
        }

        if output_path:
            output_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return json.dumps(output, ensure_ascii=False, indent=2)


# ============================================
# 全域實例
# ============================================

_global_logger: Optional[DecisionLogger] = None


def get_decision_logger(storage_dir: Path = None) -> DecisionLogger:
    """取得全域 DecisionLogger（延遲初始化）"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DecisionLogger(storage_dir)
    return _global_logger


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  🧠 Decision Log 測試")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        logger = DecisionLogger(storage_dir=Path(tmp))

        # 測試 1：基本決策
        d1 = Decision(
            agent="teaching",
            situation="學生連錯 3 次：你好→你好嗎→謝謝",
            options=["繼續新單字", "回到舊單字複習", "提供提示"],
            chosen="回到舊單字複習",
            reasoning="遺忘率過高（3/3 錯誤），優先鞏固已學內容",
            confidence=0.85,
            task="suggest_next",
            strategy="mastery_first",
        )
        logger.log(d1)
        print(f"\n  決策 1: {d1.chosen} (conf={d1.confidence})")

        # 測試 2：LLM 解析
        llm_output = json.dumps({
            "chosen": "推薦新主題",
            "reasoning": "學生掌握良好，可以拓展學習範圍",
            "confidence": 0.9,
        })
        d2 = Decision.from_llm(
            agent="teaching",
            situation="學生已學 5 詞，連續答對 3 次",
            options=["推薦新主題", "加深當前主題", "進行測驗"],
            llm_output=llm_output,
            strategy="exploration_first",
        )
        logger.log(d2)
        print(f"  決策 2: {d2.chosen} (conf={d2.confidence})")

        # 測試 3：品質 Agent 決策
        d3 = Decision(
            agent="quality",
            situation="Knowledge Agent 返回翻譯 'masalu' → '你好'",
            options=["通過", "打回重做", "降級標記"],
            chosen="打回重做",
            reasoning="'masalu' 意為 '謝謝' 非 '你好'，屬語料匹配錯誤",
            confidence=0.92,
            task="review_translation",
            strategy="mastery_first",
        )
        logger.log(d3)
        print(f"  決策 3: {d3.chosen} (conf={d3.confidence})")

        # Flush 並查詢
        logger.flush()
        stats = logger.get_stats()
        print(f"\n  統計: {json.dumps(stats, ensure_ascii=False)}")

        # 導出海報案例
        poster = logger.export_for_poster()
        print(f"\n  海報案例:\n{poster[:500]}")

    print("\n✅ Decision Log 測試通過")
