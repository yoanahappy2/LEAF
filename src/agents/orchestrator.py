"""
orchestrator.py — Orchestrator Agent（總管）

系統的大腦。負責：
1. 接收用戶輸入，理解意圖
2. 自主決定分派給哪個 Agent
3. 彙整各 Agent 的結果
4. 管理對話歷史
5. 三種長程任務模式的入口

繼承 BaseAgent，同時持有其他 Agent 的引用，
通過 MessageBus 協調它們。

作者: 地陪
日期: 2026-05-12
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.agent import BaseAgent, AgentTrace
from core.message import AgentMessage, MessageType, MessageBus
from core.rate_limiter import APIGuard
from core.decision import Decision, DecisionLogger, get_decision_logger
from core.strategy import LearningStrategy, get_strategy

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent — 系統總管

    它是整個 Multi-Agent 系統的入口和協調者。

    運作模式：
    1. ReAct 模式（一般對話）：用戶 → LLM 決定 → 調工具/分派 Agent → 回覆
    2. Plan-Driven 模式：生成多步計畫 → 逐步分派 Agent 執行
    3. Self-Judge 模式：設定目標 → 循環執行 + 品質評估直到達標
    """

    role = "orchestrator"

    def __init__(self, client: OpenAI = None, api_guard: APIGuard = None,
                 model: str = None, config_path: Path = None,
                 project_root: Path = None,
                 strategy_name: str = "mastery_first",
                 decision_logger: DecisionLogger = None,
                 language: str = "paiwan"):
        super().__init__(client=client, api_guard=api_guard,
                         model=model, config_path=config_path)
        self.project_root = project_root or Path(__file__).parent.parent.parent

        # 策略系統
        self.strategy = get_strategy(strategy_name)
        self.language = language

        # Decision Log
        self.decision_logger = decision_logger or get_decision_logger(
            self.project_root / "agent_framework" / "storage" / "decisions"
        )

        # MessageBus 和 Agent 註冊
        self.bus = MessageBus()
        self._agents_registered = False

        # 對話歷史
        self._chat_history: list[dict] = []

    # ── System Prompt ──

    def get_system_prompt(self) -> str:
        lang_map = {"paiwan": "排灣族", "amis": "阿美族"}
        lang_name = lang_map.get(self.language, self.language)

        base = (
            f"你是{lang_name}語智慧教學系統的 Orchestrator Agent（總管）。\n\n"
            "你的職責：\n"
            "1. 理解用戶的意圖和需求\n"
            "2. 自主決定需要調用哪些工具或分派給哪個 Agent\n"
            "3. 根據結果決定下一步行動\n"
            "4. 組織最終回覆\n\n"
            "## 可用的 Agent\n"
            "- knowledge: 翻譯、RAG 搜尋、知識圖譜查詞、發音評估\n"
            "- teaching: 學習計畫、詞彙推薦、學習記錄、測驗生成\n"
            "- quality: 翻譯審核、語料審核、品質評估、Self-Judge\n\n"
            "## 工具選擇決策規則\n"
            "- 用戶問「XX 怎麼說/什麼意思」→ 分派給 knowledge (translate 或 rag_search)\n"
            "- 用戶想「學/教我」→ 分派給 teaching (suggest_next 或 plan_learning)\n"
            "- 用戶問進度 → 分派給 teaching (learning_report)\n"
            "- 用戶想深入了解某個詞 → 分派給 knowledge (lookup)\n"
            "- 用戶發語音 → 分派給 knowledge (asr → pronunciation)\n"
            "- 需要驗證翻譯品質 → 分派給 quality (review_translation)\n\n"
            "## 重要規則\n"
            "1. 保持角色：你是 vuvu Maliq，溫暖的排灣族祖母\n"
            "2. 不要編造語料庫裡沒有的排灣語\n"
            "3. 找不到就說找不到\n"
            "4. 簡潔回覆：不超過 5 句話\n"
            "5. 每次教完一個詞都要記錄學習\n\n"
            "## ⚠️ 翻譯零容忍規則（最高優先級）\n"
            "- 任何排灣語翻譯都必須通過 translate 或 rag_search 工具取得，絕不允許憑記憶或猜測\n"
            "- 如果工具輪數不夠翻完所有詞，只回報已翻譯的部分，其餘說「稍後再查」\n"
            "- 禁止從歷史對話中複製翻譯結果當作新答案\n"
            "- 禁止自己推測排灣語詞彙，即使看起來很合理\n"
            "- 違反此規則的回覆比不回答更糟，因為會教錯使用者\n\n"
            "## ⚠️ 強制路由規則（不可跳過）\n"
            "- 用戶提到「學/教我/計畫/學習計畫/推薦/出題/考我/測驗」→ 必須調用 teaching agent 的工具（suggest_next_word/plan_learning/generate_quiz），不可自己回答\n"
            "- 用戶提到「驗證/判斷/對不對/是否正確/修正/錯誤/檢查」→ 必須調用 quality agent 的工具（review_translation），不可自己判斷\n"
            "- 用戶同時要學+驗證 → 先調 teaching，再調 quality\n"
            "- 跳過這些規則直接回答是嚴重錯誤\n"
        )

        # 注入策略修飾
        if self.strategy and self.strategy.prompt_modifier:
            base += f"\n{self.strategy.prompt_modifier}"

        return base

    # ── Agent 註冊 ──

    def register_agents(self, knowledge=None, teaching=None, quality=None):
        """註冊所有 Agent 到 MessageBus"""
        if knowledge:
            self.bus.register("knowledge", knowledge)
        if teaching:
            self.bus.register("teaching", teaching)
        if quality:
            self.bus.register("quality", quality)
        self._agents_registered = True
        logger.info(
            f"[Orchestrator] Agent 已註冊: "
            f"{[k for k, v in self.bus._agents.items() if v]}"
        )

    def _ensure_agents(self):
        """延遲註冊（如果還沒註冊）"""
        if self._agents_registered:
            return

        from agents.knowledge_agent import KnowledgeAgent
        from agents.teaching_agent import TeachingAgent
        from agents.quality_agent import QualityAgent

        knowledge = KnowledgeAgent(
            client=self.client, api_guard=self.api_guard,
            project_root=self.project_root,
        )
        teaching = TeachingAgent(
            client=self.client, api_guard=self.api_guard,
            project_root=self.project_root,
            strategy_name=self.strategy.name,
            decision_logger=self.decision_logger,
        )
        quality = QualityAgent(
            client=self.client, api_guard=self.api_guard,
            project_root=self.project_root,
            strategy_name=self.strategy.name,
            decision_logger=self.decision_logger,
        )

        self.register_agents(knowledge=knowledge, teaching=teaching, quality=quality)

    # ── handle_message（MessageBus 調用）──

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        處理外部傳入的訊息（MainLoop 或外部調用者）

        這是 Autonomous Runner 的入口。
        """
        task = message.payload.get("task", "chat")
        params = message.payload.get("params", {})

        if task == "chat":
            reply = self.chat(params.get("user_input", ""))
            return self._make_response(
                message, task="chat", status="completed",
                data={"reply": reply},
            )
        elif task == "execute_step":
            result = self._execute_plan_step(params)
            return self._make_response(
                message, task="execute_step", status="completed",
                data=result,
            )
        else:
            return self._make_error(message, f"未知任務: {task}")

    # ── 對話入口 ──

    def chat(self, user_input: str, user_id: str = "anonymous") -> str:
        """
        主要對話入口

        ReAct 循環：
        1. LLM 理解意圖 → 決定分派給哪個 Agent
        2. 分派 → 收集結果
        3. 生成最終回覆
        """
        self._ensure_agents()

        # Bug fix: 空輸入保護（防止 1213 錯誤）
        if not user_input or not user_input.strip():
            return "請告訴我你想學什麼？"

        # 建構 tools schema（讓 LLM 可以「調用 Agent」）
        agent_tools = self._get_agent_tools()

        # 建構 messages
        system_msg = self._get_cached_prompt()
        messages = [{"role": "system", "content": system_msg}]

        # 對話歷史（只保留最近 3 輪 = 6 條，避免歷史過長導致重複行為）
        if self._chat_history:
            messages.extend(self._chat_history[-6:])
        messages.append({"role": "user", "content": user_input})

        # ── Keyword-based Pre-Routing ──
        # 在 LLM 決策之前，先根據關鍵詞強制觸發對應 Agent
        pre_route_results = self._keyword_pre_route(user_input, user_id)
        for route_result in pre_route_results:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": route_result["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": route_result["tool_name"],
                        "arguments": json.dumps(route_result["tool_args"], ensure_ascii=False),
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(route_result["result"], ensure_ascii=False),
                "tool_call_id": route_result["tool_call_id"],
            })

        # ReAct 循環
        for turn in range(5):  # 最多 5 輪：3 輪工具 + 1 輪總結提示 + 1 輪回覆
            is_final_round = (turn >= 3)  # 從第 4 輪開始不再調工具
            
            llm_result = self.call_llm(
                messages=messages,
                tools=agent_tools if not is_final_round else None,
                temperature=self.strategy.temperature if self.strategy else 0.4,
                max_tokens=1000,
            )

            msg = llm_result["message"]

            # LLM 要調用 Agent（只在非最終輪）
            if not is_final_round and hasattr(msg, 'tool_calls') and msg.tool_calls:
                messages.append(msg)

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    logger.info(f"[Orchestrator] 調用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})")

                    # 記錄決策：為什麼選這個工具
                    # 從用戶輸入 + 選擇的工具 + 被忽略的其他工具 推導理由
                    available_tools = [t["function"]["name"] for t in agent_tools]
                    # 基於工具用途生成分類理由
                    tool_reasons = {
                        "translate": "用戶明確要求翻譯",
                        "rag_search": "需要語料庫例句支持",
                        "lookup": "需要詞彙深度資訊（級詞、親屬、相關詞）",
                        "pronunciation_check": "需要評估發音準確度",
                        "suggest_next_word": "用戶請求推薦下一個學習內容",
                        "query_user_progress": "用戶詢問學習進度",
                        "record_learning": "需要記錄本次學習行為",
                        "review_translation": "需要驗證翻譯品質",
                    }
                    reasoning = tool_reasons.get(tool_name, f"根據用戶意圖 '{user_input[:40]}' 自主選擇 {tool_name}")

                    self._log_decision(
                        task=tool_name,
                        situation=f"用戶輸入: '{user_input[:80]}' | ReAct turn {turn+1} | 可用工具: {len(available_tools)}個",
                        options=available_tools[:5],  # 最多顯示 5 個候選
                        chosen=tool_name,
                        reasoning=reasoning,
                        confidence=0.8,
                    )

                    # 路由到對應的 Agent
                    result = self._dispatch_to_agent(tool_name, tool_args, user_id)

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": tc.id,
                    })

                continue

            # LLM 直接回覆
            reply = msg.content or ""
            if not reply.strip():
                # 空回覆，注入提示讓 LLM 總結
                messages.append({"role": "user", "content": "請根據以上工具調用結果，用中文生成最終回覆。"})
                continue

            # 檢查回覆是否是工具調用格式（LLM 幻覺）
            if reply.strip().startswith('<') and 'translate' in reply:
                # LLM 試圖輸出工具調用但格式不對，注入總結提示
                messages.append({"role": "user", "content": "請直接用中文回答，不要再調用工具。根據前面的翻譯結果整理回覆。"})
                continue

            # 翻譯驗證：檢查回覆中的排灣語詞彙是否都經過工具驗證
            reply = self._verify_translations(reply, messages)

            # 更新對話歷史
            self._chat_history.append({"role": "user", "content": user_input})
            self._chat_history.append({"role": "assistant", "content": reply})
            if len(self._chat_history) > 10:
                self._chat_history = self._chat_history[-10:]

            return reply

        # 所有輪次都失敗，嘗試極簡模式
        exceeded, budget_msg = self.api_guard.is_budget_exceeded()
        if exceeded:
            # Token 預算耗盡：不調 LLM，直接回傳最後已知狀態
            logger.warning("Token 預算耗盡，進入極簡模式")
            # 從 messages 中提取最後一個工具調用結果
            for msg in reversed(messages):
                if msg.get("role") == "tool":
                    try:
                        content = json.loads(msg.get("content", "{}"))
                        if content.get("translation"):
                            return f"翻譯結果: {content['translation']}"
                        if content.get("data"):
                            return str(content['data'])[:500]
                    except:
                        pass
            return "ai~~ vuvu 需要休息一下，Token 預算快用完了，稍後再試吧！"

        try:
            llm_result = self.call_llm(messages=messages, tools=None, temperature=0.3, max_tokens=800, model="glm-4.5-air")
            reply = llm_result["message"].content or ""
            if reply.strip():
                return reply
        except RuntimeError:
            # call_llm 預算檢查拋錯，回傳基本訊息
            return "系統忙碌中，請稍後再試。"
        except Exception:
            pass

        return "ai~~ vuvu 想太久了，再問一次吧！"

    # ── 翻譯驗證 ──

    def _verify_translations(self, reply: str, messages: list) -> str:
        """程序化驗證：檢查回覆中的翻譯對是否都有對應的工具調用結果"""
        import re

        # 提取這次對話中工具實際返回的翻譯對
        verified_pairs = {}  # {paiwan: chinese} 或 {chinese: paiwan}
        for msg in messages:
            # 兼容 dict 和 ChatCompletionMessage 物件
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content or ""
            elif isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                continue

            if role == "tool" and "translation" in content:
                try:
                    import json
                    content_str = content if isinstance(content, str) else str(content)
                    parsed = json.loads(content_str)
                    inp = parsed.get("input", "")
                    trans = parsed.get("translation", "")
                    if inp and trans:
                        verified_pairs[inp] = trans
                        verified_pairs[trans] = inp
                except:
                    pass

        # 如果沒有任何工具調用結果，且回覆含有翻譯內容，標記警告
        if not verified_pairs:
            return reply  # 沒有翻譯任務，不需要驗證

        # 從回覆中提取所有翻譯對
        # Pattern: **排灣語** - 中文 或 中文 → 排灣語
        reply_pairs = []
        for m in re.finditer(r'\*\*([a-zāáǎàēéěèīíǐìōóǒòūúǔù]+[.!?]?)\*\*\s*[-–→]+\s*([^\n*]+)', reply):
            reply_pairs.append((m.group(1).strip(), m.group(2).strip()))
        for m in re.finditer(r'([\u4e00-\u9fff]+)\s*→\s*([a-zāáǎàēéěèīíǐìōóǒòūúǔù]+)', reply):
            reply_pairs.append((m.group(2).strip(), m.group(1).strip()))

        # 檢查每個翻譯對是否有工具驗證
        unverified = []
        for paiwan, chinese in reply_pairs:
            is_verified = (
                paiwan in verified_pairs or
                chinese in verified_pairs or
                verified_pairs.get(paiwan, "") == chinese or
                verified_pairs.get(chinese, "") == paiwan
            )
            if not is_verified:
                unverified.append((paiwan, chinese))

        if unverified:
            # 移除未驗證的翻譯對，加警告
            import logging
            logger = logging.getLogger("Orchestrator")
            for pw, cn in unverified:
                logger.warning(f"翻譯未驗證（可能為 LLM 編造）: {cn} = {pw}")
                # 在回覆中標記
                reply = reply.replace(
                    f"**{pw}**",
                    f"**{pw}**⚠️"
                )
                reply = reply.replace(
                    f"→ {pw}",
                    f"→ {pw}⚠️"
                )

            # 在回覆末尾加註
            reply += f"\n\n⚠️ 注意：有 {len(unverified)} 個翻譯未經工具驗證（標記⚠️），可能不準確。"

        return reply

    # ── Keyword-based Pre-Routing ──

    def _keyword_pre_route(self, user_input: str, user_id: str) -> list:
        """
        在 LLM 決策之前，根據關鍵詞強制觸發對應 Agent。
        確保 teaching/quality agent 不會被 LLM 跳過。

        規則：
        - teaching 關鍵詞 → 強制調用 suggest_next_word 或 generate_quiz
        - quality 關鍵詞 → 強制調用 review_translation
        - knowledge 關鍵詞 → 不預路由（LLM 通常會自己選 translate）
        - 多個類型可以同時觸發
        """
        import uuid
        results = []
        input_lower = user_input.lower()

        teaching_kw = ["學", "教我", "計畫", "學習計畫", "推薦", "出題", "考我", "測驗", "下一階段", "下一個"]
        quality_kw = ["驗證", "判斷", "對不對", "是否正確", "修正", "檢查", "確認", "正確嗎", "對嗎"]

        triggered_teaching = any(kw in input_lower for kw in teaching_kw)
        triggered_quality = any(kw in input_lower for kw in quality_kw)

        if triggered_teaching:
            # 決定用哪個 teaching 工具
            if any(kw in input_lower for kw in ["出題", "考我", "測驗"]):
                tool_name = "generate_quiz"
                tool_args = {"topic": user_input[:100], "count": 3}
            else:
                tool_name = "suggest_next_word"
                tool_args = {"user_id": user_id, "goal": user_input[:100]}

            logger.info(f"[Pre-Route] 教學關鍵詞匹配 → {tool_name}")
            result = self._dispatch_to_agent(tool_name, tool_args, user_id)
            results.append({
                "tool_call_id": f"pre_route_{uuid.uuid4().hex[:8]}",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result,
            })

            # 如果用戶同時要求翻譯確認，也預路由 knowledge
            if any(kw in input_lower for kw in ["確認", "翻譯", "怎麼說"]):
                # 提取要翻譯的詞
                import re
                words = re.findall(r'[「\']([^」\']+)[」\']', user_input)
                if not words:
                    # 用基本詞彙列表
                    basic_words = ["母親", "父親", "孩子", "你好", "謝謝", "水", "吃"]
                    words = [w for w in basic_words if w in input_lower]
                for word in words[:2]:  # 最多 2 個
                    k_tool_name = "translate"
                    k_tool_args = {"text": word, "direction": "c2p"}
                    logger.info(f"[Pre-Route] 教學+翻譯 → translate({word})")
                    k_result = self._dispatch_to_agent(k_tool_name, k_tool_args, user_id)
                    results.append({
                        "tool_call_id": f"pre_route_{uuid.uuid4().hex[:8]}",
                        "tool_name": k_tool_name,
                        "tool_args": k_tool_args,
                        "result": k_result,
                    })

        if triggered_quality:
            # 從用戶輸入中提取待驗證的翻譯
            tool_name = "review_translation"
            # 嘗試提取翻譯對
            import re
            pairs = re.findall(r'[「"\'](.+?)[」"\']\s*[→>到是]\s*[「"\'](.+?)[」"\']', input_lower)
            if not pairs:
                # 嘗試另一個 pattern：中文 → 排灣語
                pairs = re.findall(r'(母親|父親|孩子|你好|謝謝|水|吃|太陽|月亮|手|眼睛|火|星星|房子|道路|人|朋友|山|名字|狗)\s*[→>到是]\s*(\\w+)', input_lower)

            if pairs:
                for original, translation in pairs:
                    tool_args = {"original": original, "translation": translation, "direction": "c2p"}
                    logger.info(f"[Pre-Route] 品質關鍵詞匹配 → review_translation({original} → {translation})")
                    result = self._dispatch_to_agent(tool_name, tool_args, user_id)
                    results.append({
                        "tool_call_id": f"pre_route_{uuid.uuid4().hex[:8]}",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "result": result,
                    })
            else:
                # 沒有提取到具體翻譯對，用整個輸入做審核
                tool_args = {"original": user_input[:200], "translation": "", "direction": "auto"}
                logger.info("[Pre-Route] 品質關鍵詞匹配 → review_translation(整句)")
                result = self._dispatch_to_agent(tool_name, tool_args, user_id)
                results.append({
                    "tool_call_id": f"pre_route_{uuid.uuid4().hex[:8]}",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result,
                })

        return results

    # ── Agent 分派 ──

    def _dispatch_to_agent(self, tool_name: str, args: dict, user_id: str) -> dict:
        """根據工具名稱分派到對應的 Agent"""

        # 工具 → Agent 映射
        agent_task_map = {
            # Knowledge Agent
            "translate": ("knowledge", "translate"),
            "rag_search": ("knowledge", "rag_search"),
            "lookup": ("knowledge", "lookup"),
            "pronunciation_check": ("knowledge", "pronunciation"),
            "asr_recognize": ("knowledge", "asr"),
            # Teaching Agent
            "suggest_next_word": ("teaching", "suggest_next"),
            "query_user_progress": ("teaching", "learning_report"),
            "record_learning": ("teaching", "record_learning"),
            "plan_learning": ("teaching", "plan_learning"),
            "generate_quiz": ("teaching", "generate_quiz"),
            # Quality Agent
            "review_translation": ("quality", "review_translation"),
            "self_judge": ("quality", "self_judge"),
        }

        mapping = agent_task_map.get(tool_name)
        if not mapping:
            return {"error": f"未知工具: {tool_name}"}

        agent_name, task = mapping

        # 注入 user_id
        if tool_name in ("suggest_next_word", "query_user_progress", "record_learning"):
            args.setdefault("user_id", user_id)

        # 通過 MessageBus 發送
        msg = AgentMessage.task_assign(
            from_agent="orchestrator",
            to_agent=agent_name,
            task=task,
            params=args,
        )

        response = self.bus.send(msg)
        if response and response.payload.get("status") == "completed":
            return response.payload.get("data", {})
        elif response:
            return {"error": response.payload.get("error", "Agent 返回錯誤")}
        else:
            return {"error": f"Agent {agent_name} 無回應"}

    def _execute_plan_step(self, step: dict) -> dict:
        """
        執行 Plan 中的單一步驟（Plan-Driven 模式用）

        Args:
            step: {"name": "...", "description": "...", "agent": "knowledge", "task_type": "translate", ...}

        Returns:
            {"reply": ..., "tool_calls": [...]}
        """
        agent_name = step.get("agent", "knowledge")
        task_type = step.get("task_type", "rag_search")
        description = step.get("description", step.get("name", ""))

        logger.info(f"[Orchestrator] 執行 Plan 步驟: {description} → {agent_name}.{task_type}")

        # 用 description 作為用戶輸入
        reply = self.chat(description)

        return {"reply": reply, "step": step}

    # ── Tools Schema ──

    def _log_decision(self, task: str, situation: str,
                      options: list, chosen: str,
                      reasoning: str, confidence: float):
        """記錄一次 Agent 決策"""
        decision = Decision(
            agent=self.role,
            situation=situation,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            confidence=confidence,
            task=task,
            strategy=self.strategy.name,
            language=self.language,
        )
        self.decision_logger.log(decision)

    def _get_agent_tools(self) -> list[dict]:
        """返回 LLM 可調用的工具 schema（對應各 Agent 的能力）"""
        return [
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
            {
                "type": "function",
                "function": {
                    "name": "pronunciation_check",
                    "description": "評估排灣語發音準確度",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recognized": {"type": "string", "description": "用戶說的"},
                            "target": {"type": "string", "description": "正確答案"},
                        },
                        "required": ["recognized", "target"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "suggest_next_word",
                    "description": "根據學習歷史推薦下一個學習詞彙",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_user_progress",
                    "description": "查詢用戶學習進度",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_learning",
                    "description": "記錄學習行為（每次教完詞都要調用）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "word": {"type": "string", "description": "學習的排灣語詞"},
                            "result": {"type": "string", "enum": ["learned", "correct", "wrong"]},
                        },
                        "required": ["word", "result"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "review_translation",
                    "description": "審核翻譯品質（交叉驗證）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "translation": {"type": "string"},
                            "direction": {"type": "string", "default": "auto"},
                        },
                        "required": ["original", "translation"],
                    },
                },
            },
        ]

    def clear_history(self):
        """清空對話歷史"""
        self._chat_history.clear()


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    print("=" * 60)
    print("  🏔️ OrchestratorAgent 測試")
    print("=" * 60)

    client = OpenAI(
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    orch = OrchestratorAgent(client=client)
    orch._ensure_agents()

    # 測試 1：翻譯
    print("\n--- 測試 1：翻譯 ---")
    reply = orch.chat("謝謝的排灣語怎麼說？")
    print(f"  👵 vuvu: {reply}")

    # 測試 2：學習
    print("\n--- 測試 2：學習推薦 ---")
    reply = orch.chat("教我一個新的排灣語詞")
    print(f"  👵 vuvu: {reply}")

    # 測試 3：閒聊
    print("\n--- 測試 3：一般對話 ---")
    reply = orch.chat("你好")
    print(f"  👵 vuvu: {reply}")

    print("\n✅ OrchestratorAgent 測試完成")
