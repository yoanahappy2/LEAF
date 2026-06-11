"""
message.py — Agent 間結構化通訊協議

所有 Agent 之間的溝通通過 AgentMessage（結構化 JSON）進行，
不用自然語言，確保可靠解析和可追溯性。

訊息流向：
    User → Orchestrator → Specialist Agent → (result) → Orchestrator → User

作者: 地陪
日期: 2026-05-12
"""

import json
import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================
# 訊息類型與狀態
# ============================================

class MessageType(str, Enum):
    """Agent 間通訊的訊息類型"""
    # Orchestrator → Specialist：分派任務
    TASK_ASSIGN = "task_assign"
    # Specialist → Orchestrator：回報結果
    TASK_RESULT = "task_result"
    # Orchestrator → Specialist：要求更多資訊
    REQUEST_INFO = "request_info"
    # Specialist → Orchestrator：回報資訊
    INFO_RESPONSE = "info_response"
    # Quality Agent → 其他 Agent：品質評估結果
    QUALITY_REVIEW = "quality_review"
    # 任何 Agent → Orchestrator：回報錯誤
    ERROR = "error"
    # 廣播：系統事件通知
    BROADCAST = "broadcast"


class TaskStatus(str, Enum):
    """任務執行狀態"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"       # 等待 Quality Agent 審核
    REVISION_NEEDED = "revision_needed"  # Quality Agent 打回重做


# ============================================
# Agent 名稱常數
# ============================================

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    KNOWLEDGE = "knowledge"
    TEACHING = "teaching"
    QUALITY = "quality"


# ============================================
# AgentMessage — 核心通訊單元
# ============================================

@dataclass
class AgentMessage:
    """
    Agent 間通訊的結構化訊息

    每條訊息包含：
    - header：路由資訊（誰發的、發給誰、訊息類型）
    - payload：任務內容或結果
    - meta：追蹤用元資料

    範例（Orchestrator → Knowledge）：
        {
            "id": "msg-uuid-xxx",
            "type": "task_assign",
            "from": "orchestrator",
            "to": "knowledge",
            "payload": {
                "task": "translate",
                "params": {"text": "你好", "direction": "c2p"},
            },
            "meta": {
                "session_id": "sess-xxx",
                "parent_msg_id": null,
                "created_at": 1715481600.0,
            },
        }
    """
    # ── Header ──
    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    type: str = MessageType.TASK_ASSIGN
    from_agent: str = ""
    to_agent: str = ""

    # ── Payload ──
    payload: dict = field(default_factory=dict)
    """
    payload 格式因訊息類型而異：

    TASK_ASSIGN:
        {
            "task": "translate",           # 任務名稱
            "params": {"text": "你好"},    # 任務參數
            "context": {...},              # 上下文（可選）
        }

    TASK_RESULT:
        {
            "task": "translate",
            "status": "completed",
            "data": {"translation": "masalu"},
            "confidence": 0.95,            # 信心分數（可選）
        }

    QUALITY_REVIEW:
        {
            "reviewed_task": "translate",
            "passed": true,
            "score": 0.9,
            "feedback": "...",
            "suggestions": [...],
        }
    """

    # ── Meta ──
    meta: dict = field(default_factory=dict)
    """
    meta 欄位：
        session_id: str        — 對話 session ID
        parent_msg_id: str     — 觸發此訊息的原始訊息 ID
        created_at: float      — 建立時間戳
        priority: str          — "low" | "normal" | "high"
        retry_count: int       — 重試次數
    """

    def __post_init__(self):
        # 確保 meta 有基本欄位
        if "created_at" not in self.meta:
            self.meta["created_at"] = time.time()
        if "priority" not in self.meta:
            self.meta["priority"] = "normal"
        if "retry_count" not in self.meta:
            self.meta["retry_count"] = 0

    # ── 序列化 ──

    def to_dict(self) -> dict:
        """轉為 dict（可 JSON 序列化）"""
        return {
            "id": self.id,
            "type": self.type if isinstance(self.type, str) else self.type.value,
            "from": self.from_agent,
            "to": self.to_agent,
            "payload": self.payload,
            "meta": self.meta,
        }

    def to_json(self, indent: int = None) -> str:
        """轉為 JSON 字串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMessage":
        """從 dict 還原"""
        return cls(
            id=data.get("id", f"msg-{uuid.uuid4().hex[:12]}"),
            type=data.get("type", MessageType.TASK_ASSIGN),
            from_agent=data.get("from", ""),
            to_agent=data.get("to", ""),
            payload=data.get("payload", {}),
            meta=data.get("meta", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentMessage":
        """從 JSON 字串還原"""
        return cls.from_dict(json.loads(json_str))

    # ── 工廠方法 ──

    @classmethod
    def task_assign(cls, from_agent: str, to_agent: str,
                    task: str, params: dict = None,
                    context: dict = None,
                    session_id: str = None) -> "AgentMessage":
        """建立任務分派訊息"""
        payload = {"task": task}
        if params:
            payload["params"] = params
        if context:
            payload["context"] = context

        meta = {}
        if session_id:
            meta["session_id"] = session_id

        return cls(
            type=MessageType.TASK_ASSIGN,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            meta=meta,
        )

    @classmethod
    def task_result(cls, from_agent: str, to_agent: str,
                    task: str, status: str,
                    data: Any = None, confidence: float = None,
                    session_id: str = None,
                    parent_msg_id: str = None) -> "AgentMessage":
        """建立任務結果訊息"""
        payload = {"task": task, "status": status}
        if data is not None:
            payload["data"] = data
        if confidence is not None:
            payload["confidence"] = confidence

        meta = {}
        if session_id:
            meta["session_id"] = session_id
        if parent_msg_id:
            meta["parent_msg_id"] = parent_msg_id

        return cls(
            type=MessageType.TASK_RESULT,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            meta=meta,
        )

    @classmethod
    def error_msg(cls, from_agent: str, to_agent: str,
                  error: str, task: str = None,
                  session_id: str = None) -> "AgentMessage":
        """建立錯誤訊息"""
        payload = {"error": error}
        if task:
            payload["task"] = task

        meta = {}
        if session_id:
            meta["session_id"] = session_id

        return cls(
            type=MessageType.ERROR,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            meta=meta,
        )

    @classmethod
    def quality_review(cls, from_agent: str, to_agent: str,
                       reviewed_task: str, passed: bool,
                       score: float = None, feedback: str = None,
                       suggestions: list = None) -> "AgentMessage":
        """建立品質評估訊息"""
        payload = {
            "reviewed_task": reviewed_task,
            "passed": passed,
        }
        if score is not None:
            payload["score"] = score
        if feedback:
            payload["feedback"] = feedback
        if suggestions:
            payload["suggestions"] = suggestions

        return cls(
            type=MessageType.QUALITY_REVIEW,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
        )

    def __repr__(self) -> str:
        ptask = self.payload.get("task", "")
        return (
            f"AgentMessage({self.type} | "
            f"{self.from_agent} → {self.to_agent} | "
            f"task={ptask})"
        )


# ============================================
# MessageBus — Agent 間訊息路由
# ============================================

class MessageBus:
    """
    Agent 間的訊息匯流排

    目前是同步的 in-memory 實現。
    線 B 可替換為非同步或持久化版本。

    用法：
        bus = MessageBus()
        bus.register("orchestrator", orchestrator_agent)
        bus.register("knowledge", knowledge_agent)

        msg = AgentMessage.task_assign("orchestrator", "knowledge", "translate", ...)
        response = bus.send(msg)
    """

    def __init__(self):
        self._agents: dict[str, Any] = {}  # agent_name → agent instance
        self._history: list[AgentMessage] = []
        self._max_history = 1000

    def register(self, agent_name: str, agent_instance: Any):
        """註冊 Agent"""
        self._agents[agent_name] = agent_instance
        logger.info(f"[MessageBus] 註冊 Agent: {agent_name}")

    def unregister(self, agent_name: str):
        """取消註冊"""
        self._agents.pop(agent_name, None)

    def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        發送訊息到目標 Agent，返回回應

        Args:
            message: 要發送的訊息

        Returns:
            目標 Agent 的回應訊息，或 None（如果目標不存在）
        """
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        target = self._agents.get(message.to_agent)
        if not target:
            logger.warning(f"[MessageBus] 目標 Agent 不存在: {message.to_agent}")
            return AgentMessage.error_msg(
                from_agent="message_bus",
                to_agent=message.from_agent,
                error=f"Agent '{message.to_agent}' not registered",
            )

        logger.info(f"[MessageBus] {message.from_agent} → {message.to_agent}: {message.type}")

        try:
            response = target.handle_message(message)
            if response:
                self._history.append(response)
            return response
        except Exception as e:
            logger.error(f"[MessageBus] Agent {message.to_agent} 處理訊息失敗: {e}")
            return AgentMessage.error_msg(
                from_agent=message.to_agent,
                to_agent=message.from_agent,
                error=str(e),
                task=message.payload.get("task"),
            )

    def broadcast(self, message: AgentMessage) -> list[Optional[AgentMessage]]:
        """廣播給所有已註冊的 Agent（除了發送者）"""
        responses = []
        for name, agent in self._agents.items():
            if name != message.from_agent:
                msg_copy = AgentMessage(
                    type=message.type,
                    from_agent=message.from_agent,
                    to_agent=name,
                    payload=message.payload,
                    meta=message.meta.copy(),
                )
                try:
                    resp = self.send(msg_copy)
                    responses.append(resp)
                except Exception as e:
                    logger.error(f"[MessageBus] 廣播給 {name} 失敗: {e}")
                    responses.append(None)
        return responses

    def get_history(self, session_id: str = None,
                    agent_name: str = None,
                    limit: int = 50) -> list[AgentMessage]:
        """
        查詢訊息歷史

        Args:
            session_id: 只查特定 session
            agent_name: 只查涉及特定 Agent 的訊息
            limit: 最多返回幾條
        """
        results = self._history

        if session_id:
            results = [m for m in results if m.meta.get("session_id") == session_id]
        if agent_name:
            results = [
                m for m in results
                if m.from_agent == agent_name or m.to_agent == agent_name
            ]

        return results[-limit:]

    def clear_history(self):
        """清空歷史"""
        self._history.clear()


# ============================================
# 終端機測試
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  📨 AgentMessage 通訊協議測試")
    print("=" * 60)

    # 測試 1：建立任務分派訊息
    msg1 = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="translate",
        params={"text": "你好", "direction": "c2p"},
        session_id="test-session-001",
    )
    print(f"\n📨 訊息 1（任務分派）:")
    print(msg1.to_json(indent=2))

    # 測試 2：建立任務結果訊息
    msg2 = AgentMessage.task_result(
        from_agent="knowledge",
        to_agent="orchestrator",
        task="translate",
        status="completed",
        data={"translation": "masalu", "method": "rag"},
        confidence=0.95,
        parent_msg_id=msg1.id,
    )
    print(f"\n📨 訊息 2（任務結果）:")
    print(msg2.to_json(indent=2))

    # 測試 3：品質評估
    msg3 = AgentMessage.quality_review(
        from_agent="quality",
        to_agent="orchestrator",
        reviewed_task="translate",
        passed=True,
        score=0.92,
        feedback="翻譯準確，語料匹配度高",
    )
    print(f"\n📨 訊息 3（品質評估）:")
    print(msg3.to_json(indent=2))

    # 測試 4：JSON 往返
    json_str = msg1.to_json()
    restored = AgentMessage.from_json(json_str)
    assert restored.id == msg1.id
    assert restored.payload["task"] == "translate"
    print(f"\n✅ JSON 往返測試通過")

    # 測試 5：MessageBus
    print(f"\n--- MessageBus 測試 ---")

    class MockAgent:
        def __init__(self, name):
            self.name = name

        def handle_message(self, msg: AgentMessage) -> AgentMessage:
            print(f"  [{self.name}] 收到: {msg}")
            return AgentMessage.task_result(
                from_agent=self.name,
                to_agent=msg.from_agent,
                task=msg.payload.get("task", "unknown"),
                status="completed",
                data={"mock_response": True},
            )

    bus = MessageBus()
    bus.register("orchestrator", MockAgent("orchestrator"))
    bus.register("knowledge", MockAgent("knowledge"))

    resp = bus.send(msg1)
    print(f"\n  回應: {resp}")
    print(f"  歷史記錄: {len(bus.get_history())} 條")

    print("\n✅ 所有測試通過")
