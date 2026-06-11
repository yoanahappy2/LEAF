"""
teaching_agent.py — 教學策略專員

負責所有「教學相關」的任務：
- 學習路徑規劃（Plan-Driven 的 plan 生成）
- 詞彙推薦（下一個該學什麼）
- 學習記錄（追蹤進度）
- 測驗生成（出題）
- 學習報告

繼承 BaseAgent，接收 Orchestrator 分派的教學任務。

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
from core.message import AgentMessage, MessageType
from core.rate_limiter import APIGuard
from core.decision import Decision, DecisionLogger, get_decision_logger
from core.strategy import LearningStrategy, get_strategy

logger = logging.getLogger(__name__)


class TeachingAgent(BaseAgent):
    """
    教學策略專員

    支援的任務：
    - "plan_learning"    生成學習計畫（Plan-Driven）
    - "suggest_next"     推薦下一個學習詞彙
    - "record_learning"  記錄學習行為
    - "generate_quiz"    生成測驗題
    - "learning_report"  學習報告

    教學策略：
    1. 初學者路徑：問候 → 數字 → 身體 → 自然 → 家族
    2. 個人化：根據已學詞彙、薄弱詞彙推薦
    3. 測驗驅動：學完主題後測驗，答錯的加入複習
    """

    role = "teaching"

    def __init__(self, client: OpenAI = None, api_guard: APIGuard = None,
                 model: str = None, config_path: Path = None,
                 project_root: Path = None,
                 strategy_name: str = "mastery_first",
                 decision_logger: DecisionLogger = None):
        super().__init__(client=client, api_guard=api_guard,
                         model=model, config_path=config_path)
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.strategy = get_strategy(strategy_name)
        self.decision_logger = decision_logger or get_decision_logger(
            self.project_root / "agent_framework" / "storage" / "decisions"
        )
        self._user_profile = None
        self._corpus = None
        self._services_loaded = False

    def get_system_prompt(self) -> str:
        base = (
            "你是排灣族語教學策略專員（Teaching Agent）。\n\n"
            "職責：\n"
            "1. 根據用戶程度設計個人化學習路徑\n"
            "2. 規劃多步學習計畫\n"
            "3. 生成測驗題目\n"
            "4. 追蹤學習進度\n\n"
            "學習路徑設計原則：\n"
            "- 初學者：問候 → 數字 → 身體 → 自然 → 家族\n"
            "- 每個主題 3-5 個詞彙\n"
            "- 學完一個主題後做測驗\n"
            "- 答錯的詞記錄下來，之後複習\n\n"
            "輸出格式：結構化 JSON\n"
        )
        if self.strategy and self.strategy.prompt_modifier:
            base += f"\n{self.strategy.prompt_modifier}"
        return base

    def _ensure_services(self):
        if self._services_loaded:
            return

        logger.info("[TeachingAgent] 載入教學服務...")

        # 用戶檔案
        try:
            sys.path.insert(0, str(self.project_root))
            from feishu_bot.user_profile import UserProfile
            self._user_profile = UserProfile()
            logger.info("  ✅ 用戶檔案已載入")
        except Exception as e:
            logger.warning(f"  ⚠️ 用戶檔案載入失敗: {e}")

        # 語料庫
        try:
            corpus_path = self.project_root / "data" / "merged_corpus.json"
            if not corpus_path.exists():
                corpus_path = self.project_root / "data" / "expanded_corpus.json"
            if corpus_path.exists():
                with open(corpus_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self._corpus = raw.get("entries", raw) if isinstance(raw, dict) else raw
                logger.info(f"  ✅ 語料庫已載入: {len(self._corpus)} 筆")
            else:
                self._corpus = []
        except Exception as e:
            logger.warning(f"  ⚠️ 語料庫載入失敗: {e}")
            self._corpus = []

        self._services_loaded = True

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self._ensure_services()

        task = message.payload.get("task", "")
        params = message.payload.get("params", {})
        logger.info(f"[TeachingAgent] 收到任務: {task}")

        handlers = {
            "plan_learning": self._handle_plan_learning,
            "suggest_next": self._handle_suggest_next,
            "record_learning": self._handle_record_learning,
            "generate_quiz": self._handle_generate_quiz,
            "learning_report": self._handle_learning_report,
        }

        handler = handlers.get(task)
        if not handler:
            return self._make_error(message, f"未知任務類型: {task}")

        try:
            return handler(message, params)
        except Exception as e:
            logger.error(f"[TeachingAgent] 任務 {task} 失敗: {e}", exc_info=True)
            return self._make_error(message, str(e))

    # ── 學習計畫生成（Plan-Driven 核心）──

    def _handle_plan_learning(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """
        根據目標生成多步學習計畫

        這是 Plan-Driven 長程任務的入口。
        返回一個可由 Orchestrator 逐步執行的 plan。
        """
        goal = params.get("goal", "學排灣語問候")
        user_id = params.get("user_id", "anonymous")
        level = params.get("level", "beginner")  # beginner | intermediate

        if not self.client:
            return self._make_error(msg, "LLM 不可用，無法生成計畫")

        # 查用戶進度（如果可用）
        user_context = ""
        if self._user_profile:
            try:
                learned = self._user_profile.get_learned_words(user_id)
                weak = self._user_profile.get_weak_words(user_id)
                user_context = f"\n已學詞彙: {learned[-20:] if learned else '無'}\n薄弱詞彙: {weak[:10] if weak else '無'}"
            except Exception:
                pass

        prompt = (
            f"你是排灣族語教學系統的學習規劃 Agent。\n\n"
            f"用戶 ID: {user_id}\n"
            f"程度: {level}\n"
            f"學習目標: {goal}"
            f"{user_context}\n\n"
            f"請生成一個 3-5 步的學習計畫。每步包含：\n"
            f'- step: 步驟編號\n'
            f'- name: 步驟名稱\n'
            f'- description: 這一步要做什麼（詳細描述，Agent 可直接執行）\n'
            f'- task_type: "translate" | "rag_search" | "quiz" | "practice"\n'
            f'- agent: 負責的 agent（通常是 "knowledge" 或 "teaching"）\n\n'
            f"輸出 JSON 陣列，不要其他文字。"
        )

        llm_result = self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            tools=None,
        )

        content = llm_result["message"].content.strip()

        # 解析 JSON
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            plan_steps = json.loads(content.strip())
        except json.JSONDecodeError:
            # Fallback：硬編碼簡單計畫
            plan_steps = [
                {"step": 1, "name": "認識問候語", "description": f"教三個排灣語問候詞：{goal}", "task_type": "rag_search", "agent": "knowledge"},
                {"step": 2, "name": "練習發音", "description": "讓用戶練習剛學的問候詞", "task_type": "practice", "agent": "teaching"},
                {"step": 3, "name": "測驗", "description": "測驗剛學的問候詞", "task_type": "quiz", "agent": "teaching"},
            ]

        return self._make_response(
            msg,
            task="plan_learning",
            status="completed",
            data={
                "goal": goal,
                "level": level,
                "steps": plan_steps,
                "total_steps": len(plan_steps),
            },
        )

    # ── 詞彙推薦 ──

    def _handle_suggest_next(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """推薦下一個學習詞彙（根據策略走不同路徑）"""
        user_id = params.get("user_id", "anonymous")

        # === 策略分叉：不同策略走不同推薦邏輯 ===

        # Exam Driven: 優先高頻詞
        if self.strategy.prioritize_high_freq and self._corpus:
            # 按語料出現頻率排序（這裡用簡單的詞長 + 重複度估算）
            freq_map = {}
            for entry in (self._corpus or []):
                if isinstance(entry, dict):
                    pw = entry.get("paiwan", "")
                    cn = entry.get("chinese", "")
                    if pw:
                        freq_map[pw] = freq_map.get(pw, 0) + 1

            # 取最高頻的詞
            sorted_words = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
            if sorted_words:
                top_word = sorted_words[0][0]
                # 找對應中文
                for entry in (self._corpus or []):
                    if isinstance(entry, dict) and entry.get("paiwan") == top_word:
                        suggestion = {"paiwan": top_word, "chinese": entry.get("chinese", "")}
                        break
                else:
                    suggestion = {"paiwan": top_word, "chinese": ""}

                self.decision_logger.log(Decision(
                    agent="teaching",
                    situation=f"用戶 {user_id} 請求推薦 | Exam Driven 策略",
                    options=["高頻詞優先", "個人化推薦", "隨機探索"],
                    chosen="高頻詞優先",
                    reasoning=f"Exam Driven 策略：優先最高頻詞 '{top_word}'（出現 {sorted_words[0][1]} 次）",
                    confidence=0.9,
                    task="suggest_next",
                    strategy=self.strategy.name,
                ))
                return self._make_response(
                    msg, task="suggest_next", status="completed",
                    data={"suggestion": suggestion, "source": "high_freq", "strategy": "exam_driven"},
                )

        # Mastery First: 個人化 + 複習優先
        if self.strategy.retry_on_error and self._user_profile:
            try:
                weak = self._user_profile.get_weak_words(user_id) if hasattr(self._user_profile, 'get_weak_words') else []
                if weak:
                    # 有薄弱詞，優先複習
                    word = weak[0] if isinstance(weak[0], str) else weak[0].get("word", str(weak[0]))
                    suggestion = {"paiwan": word, "chinese": "", "review": True}
                    self.decision_logger.log(Decision(
                        agent="teaching",
                        situation=f"用戶 {user_id} 請求推薦 | 有 {len(weak)} 個薄弱詞",
                        options=["複習薄弱詞", "推新詞", "測驗"],
                        chosen="複習薄弱詞",
                        reasoning=f"Mastery First 策略：優先複習薄弱詞 '{word}'（共 {len(weak)} 個待複習）",
                        confidence=0.9,
                        task="suggest_next",
                        strategy=self.strategy.name,
                    ))
                    return self._make_response(
                        msg, task="suggest_next", status="completed",
                        data={"suggestion": suggestion, "source": "weak_review", "strategy": "mastery_first"},
                    )
            except Exception:
                pass

        # Exploration First: 隨機推新主題
        if not self.strategy.retry_on_error and self._corpus:
            import random
            # 嘗試選一個不同主題的詞
            entry = random.choice(self._corpus)
            if isinstance(entry, dict):
                suggestion = {"paiwan": entry.get("paiwan", ""), "chinese": entry.get("chinese", "")}
            else:
                suggestion = {"raw": str(entry)[:100]}

            self.decision_logger.log(Decision(
                agent="teaching",
                situation=f"用戶 {user_id} 請求推薦 | Exploration First 策略",
                options=["隨機新主題", "複習舊詞", "高頻詞"],
                chosen="隨機新主題",
                reasoning=f"Exploration First 策略：隨機推薦 '{suggestion.get('chinese', '')}' 擴展學習面",
                confidence=0.7,
                task="suggest_next",
                strategy=self.strategy.name,
            ))
            return self._make_response(
                msg, task="suggest_next", status="completed",
                data={"suggestion": suggestion, "source": "random_explore", "strategy": "exploration_first"},
            )

        # Fallback: 個人化推薦（如有用戶檔案）
        if self._user_profile:
            try:
                suggestion = self._user_profile.suggest_next(user_id, self._corpus or [])
                if suggestion:
                    self.decision_logger.log(Decision(
                        agent="teaching",
                        situation=f"用戶 {user_id} 請求推薦 | Fallback",
                        options=["個人化推薦", "語料庫隨機"],
                        chosen="個人化推薦",
                        reasoning=f"無特定策略觸發，使用用戶檔案推薦",
                        confidence=0.75,
                        task="suggest_next",
                        strategy=self.strategy.name,
                    ))
                    return self._make_response(
                        msg, task="suggest_next", status="completed",
                        data={"suggestion": suggestion, "source": "user_profile"},
                    )
            except Exception:
                pass

        # 最終 Fallback：語料庫隨機
        if self._corpus:
            import random
            entry = random.choice(self._corpus)
            if isinstance(entry, dict):
                suggestion = {"paiwan": entry.get("paiwan", ""), "chinese": entry.get("chinese", "")}
            else:
                suggestion = {"raw": str(entry)[:100]}

            return self._make_response(
                msg, task="suggest_next", status="completed",
                data={"suggestion": suggestion, "source": "random_corpus"},
            )

        return self._make_error(msg, "無法推薦：用戶檔案和語料庫均不可用")

    # ── 學習記錄 ──

    def _handle_record_learning(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """記錄學習行為"""
        user_id = params.get("user_id", "anonymous")
        word = params.get("word", "")
        result = params.get("result", "learned")  # learned | correct | wrong

        if not word:
            return self._make_error(msg, "記錄學習需要 word 參數")

        if self._user_profile:
            try:
                if result == "learned":
                    self._user_profile.record_learn(user_id, word)
                elif result == "correct":
                    self._user_profile.record_correct(user_id, word)
                elif result == "wrong":
                    self._user_profile.record_wrong(user_id, word)
                else:
                    return self._make_error(msg, f"未知的學習結果: {result}")

                return self._make_response(
                    msg, task="record_learning", status="completed",
                    data={"user_id": user_id, "word": word, "result": result},
                )
            except Exception as e:
                return self._make_error(msg, f"記錄失敗: {e}")

        return self._make_response(
            msg, task="record_learning", status="completed",
            data={"user_id": user_id, "word": word, "result": result, "note": "未持久化（用戶檔案不可用）"},
        )

    # ── 測驗生成 ──

    def _handle_generate_quiz(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """生成測驗題"""
        topic = params.get("topic", "")
        words = params.get("words", [])
        quiz_type = params.get("type", "mixed")  # mixed | c2p | p2c

        if not self.client:
            return self._make_error(msg, "LLM 不可用，無法生成測驗")

        prompt = (
            f"生成排灣語測驗題。\n\n"
            f"主題: {topic or '綜合'}\n"
            f"指定詞彙: {words if words else '由你從常見詞彙中選擇'}\n"
            f"測驗類型: {quiz_type}\n\n"
            f"生成 3 題，每題包含：\n"
            f"- question: 題目\n"
            f"- answer: 正確答案\n"
            f"- type: 'c2p'（中→排灣）或 'p2c'（排灣→中）\n\n"
            f"輸出 JSON 陣列。"
        )

        llm_result = self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
            tools=None,
        )

        content = llm_result["message"].content.strip()
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            quiz = json.loads(content.strip())
        except json.JSONDecodeError:
            quiz = [{"question": "「謝謝」的排灣語怎麼說？", "answer": "masalu", "type": "c2p"}]

        return self._make_response(
            msg, task="generate_quiz", status="completed",
            data={"quiz": quiz, "count": len(quiz), "topic": topic},
        )

    # ── 學習報告 ──

    def _handle_learning_report(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """生成學習報告"""
        user_id = params.get("user_id", "anonymous")

        if self._user_profile:
            try:
                stats = self._user_profile.get_stats(user_id)
                weak = self._user_profile.get_weak_words(user_id)
                learned = self._user_profile.get_learned_words(user_id)

                return self._make_response(
                    msg, task="learning_report", status="completed",
                    data={
                        "user_id": user_id,
                        "stats": stats,
                        "weak_words": weak[:10],
                        "learned_count": len(learned),
                        "recent_learned": learned[-10:] if learned else [],
                    },
                )
            except Exception as e:
                return self._make_error(msg, f"報告生成失敗: {e}")

        return self._make_response(
            msg, task="learning_report", status="completed",
            data={"user_id": user_id, "note": "用戶檔案不可用，無報告數據"},
        )


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    print("=" * 60)
    print("  📚 TeachingAgent 測試")
    print("=" * 60)

    client = OpenAI(
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    agent = TeachingAgent(client=client)

    # 測試 1：生成學習計畫
    print("\n--- 測試 1：Plan-Driven 學習計畫 ---")
    msg1 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="teaching",
        task="plan_learning",
        params={"goal": "學排灣語問候", "level": "beginner", "user_id": "test"},
    )
    resp1 = agent.handle_message(msg1)
    plan_data = resp1.payload.get("data", {})
    print(f"  目標: {plan_data.get('goal', '')}")
    print(f"  步驟數: {plan_data.get('total_steps', 0)}")
    for step in plan_data.get("steps", []):
        print(f"    Step {step.get('step', '?')}: {step.get('name', '')} → {step.get('agent', '')}.{step.get('task_type', '')}")

    # 測試 2：推薦下一個詞
    print("\n--- 測試 2：詞彙推薦 ---")
    msg2 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="teaching",
        task="suggest_next",
        params={"user_id": "test"},
    )
    resp2 = agent.handle_message(msg2)
    print(f"  推薦: {resp2.payload.get('data', {})}")

    # 測試 3：記錄學習
    print("\n--- 測試 3：記錄學習 ---")
    msg3 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="teaching",
        task="record_learning",
        params={"user_id": "test", "word": "masalu", "result": "correct"},
    )
    resp3 = agent.handle_message(msg3)
    print(f"  結果: {resp3.payload.get('status')}")

    # 測試 4：生成測驗
    print("\n--- 測試 4：測驗生成 ---")
    msg4 = AgentMessage.task_assign(
        from_agent="orchestrator", to_agent="teaching",
        task="generate_quiz",
        params={"topic": "問候", "words": ["masalu", "djavadjvai"]},
    )
    resp4 = agent.handle_message(msg4)
    quiz_data = resp4.payload.get("data", {})
    print(f"  題數: {quiz_data.get('count', 0)}")
    for q in quiz_data.get("quiz", []):
        print(f"    Q: {q.get('question', '')} A: {q.get('answer', '')}")

    print("\n✅ TeachingAgent 所有測試完成")
