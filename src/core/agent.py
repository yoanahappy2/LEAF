"""
agent.py — Agent 基類

所有 Agent（Orchestrator、Knowledge、Teaching、Quality）的抽象基類。
提供：
1. LLM 調用（通過 APIGuard 重試 + 模型選擇）
2. 訊息處理接口（handle_message）
3. ReAct 循環（Thought → Action → Observation）
4. System Prompt 載入
5. 工具調用框架

設計原則：
- 框架代碼不含任何語言相關邏輯
- 具體的 tools 和 prompts 從外部注入
- Agent 之間只通過 AgentMessage 溝通

作者: 地陪
日期: 2026-05-12
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Callable

from openai import OpenAI

from .message import AgentMessage, MessageType, TaskStatus
from .rate_limiter import APIGuard, get_api_guard

logger = logging.getLogger(__name__)


# ============================================
# AgentTrace — ReAct 過程追蹤
# ============================================

class AgentTrace:
    """
    記錄一次 Agent 運作的完整過程

    包含 ReAct 循環中的每一步：
    - thought：Agent 的推理過程
    - action：調用了什麼工具/發了什麼訊息
    - observation：收到什麼結果
    """

    def __init__(self, agent_name: str, task: str = ""):
        self.agent_name = agent_name
        self.task = task
        self.thoughts: list[str] = []
        self.actions: list[dict] = []
        self.observations: list[Any] = []
        self.messages_sent: list[AgentMessage] = []
        self.messages_received: list[AgentMessage] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.total_tokens = 0
        self.llm_calls = 0

    @property
    def elapsed_ms(self) -> int:
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)

    def add_thought(self, thought: str):
        self.thoughts.append(thought)

    def add_action(self, action_type: str, detail: dict):
        self.actions.append({"type": action_type, "detail": detail, "at": time.time()})

    def add_observation(self, observation: Any):
        self.observations.append(observation)

    def add_message_sent(self, msg: AgentMessage):
        self.messages_sent.append(msg)

    def add_message_received(self, msg: AgentMessage):
        self.messages_received.append(msg)

    def record_llm_call(self, tokens: int):
        self.llm_calls += 1
        self.total_tokens += tokens

    def finish(self):
        self.end_time = time.time()

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "task": self.task,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "num_observations": len(self.observations),
            "messages_sent": len(self.messages_sent),
            "messages_received": len(self.messages_received),
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "elapsed_ms": self.elapsed_ms,
        }

    def format_trace(self) -> str:
        """格式化為人類可讀的 ReAct 日誌"""
        lines = [f"📋 Trace [{self.agent_name}] task={self.task}"]
        lines.append(f"⏱ {self.elapsed_ms}ms | {self.llm_calls} LLM calls | {self.total_tokens} tokens")
        lines.append("")

        for i, action in enumerate(self.actions):
            lines.append(f"  Step {i + 1}: {action['type']}")
            lines.append(f"    Detail: {json.dumps(action['detail'], ensure_ascii=False)[:120]}")
            if i < len(self.observations):
                obs = str(self.observations[i])[:150]
                lines.append(f"    Observation: {obs}")
            lines.append("")

        if self.messages_sent:
            lines.append(f"  📤 Sent {len(self.messages_sent)} messages")
        if self.messages_received:
            lines.append(f"  📥 Received {len(self.messages_received)} messages")

        return "\n".join(lines)


# ============================================
# BaseAgent — 所有 Agent 的基類
# ============================================

class BaseAgent(ABC):
    """
    Agent 抽象基類

    子類必須實現：
    - role: Agent 角色名稱
    - handle_message(): 處理收到的 AgentMessage
    - get_system_prompt(): 返回 system prompt（或從檔案載入）

    可選覆寫：
    - get_tool_schemas(): 返回工具 schema 列表
    - get_tool_executor(): 返回工具執行函數

    用法：
        class KnowledgeAgent(BaseAgent):
            role = "knowledge"

            def get_system_prompt(self):
                return "你是知識檢索專員..."

            def handle_message(self, msg: AgentMessage) -> AgentMessage:
                task = msg.payload.get("task")
                if task == "translate":
                    result = self._translate(msg.payload["params"])
                    return AgentMessage.task_result(...)
    """

    # 子類覆寫
    role: str = "base"

    def __init__(self, client: OpenAI = None, api_guard: APIGuard = None,
                 model: str = None, config_path: Path = None):
        """
        Args:
            client: OpenAI 兼容的 LLM 客戶端（如智譜）
            api_guard: API 保護機制（如果 None，使用全域實例）
            model: 覆寫模型選擇（如果 None，根據 agent role 自動選）
            config_path: 語言配置目錄路徑（用於載入 prompt 檔案）
        """
        self.client = client
        self.api_guard = api_guard or get_api_guard()
        self._model_override = model
        self.config_path = config_path

        # 當前 trace
        self._current_trace: Optional[AgentTrace] = None

        # System prompt 快取
        self._system_prompt: Optional[str] = None

    # ── 屬性 ──

    @property
    def model(self) -> str:
        """當前使用的模型"""
        if self._model_override:
            return self._model_override
        return self.api_guard.pick_model_for_agent(self.role)

    # ── 抽象方法（子類必須實現）──

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        返回此 Agent 的 system prompt

        可以直接返回字串，或從 config_path 載入檔案。
        """
        pass

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        處理收到的 AgentMessage

        這是 Agent 的主要入口。Orchestrator 會通過 MessageBus 將
        訊息路由到這裡。

        Args:
            message: 收到的訊息

        Returns:
            回應訊息，或 None（如果不需要回應）
        """
        pass

    # ── LLM 調用 ──

    def call_llm(self, messages: list[dict], tools: list[dict] = None,
                 temperature: float = 0.4, max_tokens: int = 1000,
                 tool_choice: str = "auto", model: str = None) -> dict:
        """
        帶保護的 LLM 調用

        自動：
        - 使用 APIGuard 重試
        - 記錄 token 用量
        - 更新 AgentTrace

        Args:
            messages: OpenAI 格式的 messages 列表
            tools: 工具 schema 列表（可選）
            temperature: 溫度
            max_tokens: 最大 token 數
            tool_choice: "auto" | "none" | {"type": "function", "function": {"name": ...}}

        Returns:
            {
                "message": response.choices[0].message,  # 原始 message 物件
                "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
                "finish_reason": "stop" | "tool_calls",
            }
        """
        # Token 預算硬上限檢查
        exceeded, msg = self.api_guard.is_budget_exceeded()
        if exceeded:
            raise RuntimeError(f"🚫 {msg} — 停止調用，請到智譜平台充值或領取新資源包")
        if msg:
            logger = logging.getLogger("Agent")
            logger.warning(msg)

        kwargs = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.api_guard.call_with_retry(
            self.client.chat.completions.create,
            **kwargs,
        )

        choice = response.choices[0]
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # 記錄到 guard 和 trace
            self.api_guard.record_usage(self.model, usage["total_tokens"], self.role)
            if self._current_trace:
                self._current_trace.record_llm_call(usage["total_tokens"])

        return {
            "message": choice.message,
            "usage": usage,
            "finish_reason": choice.finish_reason,
        }

    # ── ReAct 循環 ──

    def react_loop(self, user_input: str, tools: list[dict] = None,
                   tool_executor: Callable = None,
                   max_turns: int = 3, max_tools_per_turn: int = 2,
                   chat_history: list = None,
                   extra_context: str = "") -> dict:
        """
        ReAct 循環：Thought → Action → Observation → ... → Reply

        這是 Agent 處理任務的核心循環。支援：
        1. LLM 自主決定調用工具（Action）
        2. 執行工具並返回結果（Observation）
        3. LLM 根據結果繼續推理或生成最終回覆

        Args:
            user_input: 用戶輸入（或任務描述）
            tools: 可用工具 schema 列表
            tool_executor: 工具執行函數 tool_executor(name, args) → result
            max_turns: 最大循環輪數
            max_tools_per_turn: 單輪最大工具調用數
            chat_history: 對話歷史
            extra_context: 額外注入的上下文

        Returns:
            {
                "reply": str,                    # 最終回覆
                "trace": AgentTrace,             # 完整 ReAct 過程
                "tool_calls": list[dict],        # 工具調用記錄
                "messages": list[dict],          # 完整 messages 歷史
            }
        """
        trace = AgentTrace(agent_name=self.role, task=user_input[:80])
        self._current_trace = trace

        # 建構 messages
        system_prompt = self._get_cached_prompt()
        if extra_context:
            system_prompt += f"\n\n{extra_context}"

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-20:])
        messages.append({"role": "user", "content": user_input})

        tool_call_log = []

        for turn in range(max_turns):
            trace.add_thought(f"ReAct Turn {turn + 1}")

            # 調用 LLM
            llm_result = self.call_llm(
                messages=messages,
                tools=tools,
                temperature=0.4,
            )

            message = llm_result["message"]

            # LLM 要調用工具
            if hasattr(message, 'tool_calls') and message.tool_calls:
                messages.append(message)

                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    trace.add_action("tool_call", {
                        "tool": tool_name,
                        "args": tool_args,
                    })

                    # 執行工具
                    if tool_executor:
                        try:
                            result = tool_executor(tool_name, tool_args)
                        except Exception as e:
                            result = {"error": str(e)}
                            logger.error(f"[{self.role}] 工具 {tool_name} 失敗: {e}")
                    else:
                        result = {"error": f"無工具執行器"}

                    trace.add_observation(result)
                    tool_call_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                    })

                    # 工具結果回傳 LLM
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": tc.id,
                    })

                # 安全限制
                if len(message.tool_calls) > max_tools_per_turn:
                    break

                continue

            # LLM 直接回覆（無需工具或已完成推理）
            reply = message.content or ""
            trace.add_thought("Final reply generated")
            trace.finish()

            return {
                "reply": reply,
                "trace": trace,
                "tool_calls": tool_call_log,
                "messages": messages,
            }

        # 超過最大輪數
        trace.finish()
        return {
            "reply": "",
            "trace": trace,
            "tool_calls": tool_call_log,
            "messages": messages,
            "error": "max_turns_exceeded",
        }

    # ── Prompt 管理 ──

    def _get_cached_prompt(self) -> str:
        """取得 system prompt（帶快取）"""
        if self._system_prompt is None:
            self._system_prompt = self.get_system_prompt()
        return self._system_prompt

    def load_prompt_from_file(self, filename: str) -> str:
        """
        從配置目錄載入 prompt 檔案

        Args:
            filename: 檔案名（如 "orchestrator.txt"）

        Returns:
            prompt 內容

        Raises:
            FileNotFoundError: 如果檔案不存在
        """
        if not self.config_path:
            raise FileNotFoundError("config_path 未設定")

        prompt_path = self.config_path / "prompts" / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 檔案不存在: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def invalidate_prompt_cache(self):
        """清除 prompt 快取（例如配置更新後）"""
        self._system_prompt = None

    # ── 工具方法 ──

    def _make_response(self, message: AgentMessage, task: str,
                       status: str, data: Any = None,
                       confidence: float = None) -> AgentMessage:
        """快速建立回應訊息"""
        return AgentMessage.task_result(
            from_agent=self.role,
            to_agent=message.from_agent,
            task=task,
            status=status,
            data=data,
            confidence=confidence,
            parent_msg_id=message.id,
            session_id=message.meta.get("session_id"),
        )

    def _make_error(self, message: AgentMessage, error: str) -> AgentMessage:
        """快速建立錯誤訊息"""
        return AgentMessage.error_msg(
            from_agent=self.role,
            to_agent=message.from_agent,
            error=error,
            task=message.payload.get("task"),
            session_id=message.meta.get("session_id"),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} role={self.role} model={self.model}>"


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    print("=" * 60)
    print("  🤖 BaseAgent 測試")
    print("=" * 60)

    # 測試 AgentTrace
    print("\n--- AgentTrace ---")
    trace = AgentTrace("test_agent", "測試任務")
    trace.add_thought("這是第一步")
    trace.add_action("tool_call", {"tool": "search", "args": {"query": "test"}})
    trace.add_observation({"results": ["a", "b"]})
    trace.record_llm_call(500)
    trace.finish()
    print(trace.format_trace())

    # 測試 BaseAgent 子類
    print("\n--- MinimalAgent（測試用）---")

    class MinimalAgent(BaseAgent):
        role = "test"

        def get_system_prompt(self) -> str:
            return "你是一個測試 Agent。"

        def handle_message(self, message: AgentMessage) -> AgentMessage:
            task = message.payload.get("task", "")
            return self._make_response(message, task, "completed", {"echo": task})

    agent = MinimalAgent()
    print(f"  Agent: {agent}")
    print(f"  Role: {agent.role}")
    print(f"  Model: {agent.model}")

    # 測試訊息處理
    msg = AgentMessage.task_assign("orchestrator", "test", "echo", {"text": "你好"})
    resp = agent.handle_message(msg)
    print(f"  Request: {msg}")
    print(f"  Response: {resp}")
    print(f"  Response payload: {resp.payload}")

    print("\n✅ 所有測試通過")
