"""
loop.py — 自主運行主控制器（Autonomous Runner）

讓 Multi-Agent 系統能夠自主運行數小時，無需人工干預。

核心循環：
1. 健康檢查
2. 檢查目標是否達成
3. 獲取下一個任務（Plan 中的步驟 或 自主生成）
4. 分派給 Agent 執行
5. Self-Judge 評估結果
6. 保存 checkpoint
7. 自適應等待

支援三種長程任務模式：
- Plan-Driven：按預生成的 Plan 逐步執行
- Self-Judge：循環執行直到目標達成
- ReAct：每步由 LLM 自主決定

作者: 地陪
日期: 2026-05-12
"""

import json
import time
import signal
import logging
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.agent import BaseAgent
from core.message import AgentMessage, MessageType
from core.rate_limiter import APIGuard, get_api_guard
from core.plan import Goal, Plan, PlanStep
from core.state import SystemState, StateManager

logger = logging.getLogger(__name__)


# ============================================
# 運行配置
# ============================================

@dataclass
class LoopConfig:
    """主循環配置"""
    # Loop 控制
    base_wait: float = 2.0              # 基礎等待秒數
    idle_wait: float = 30.0             # 無任務時等待
    degraded_wait: float = 60.0         # 降級模式等待
    checkpoint_interval: float = 300.0  # checkpoint 間隔秒數

    # 安全限制
    max_consecutive_failures: int = 5
    max_loop_count: int = 1000          # 最大循環次數
    requests_per_minute_limit: int = 50

    # Self-Judge
    judge_interval: int = 5             # 每 N 步 judge 一次
    judge_min_confidence: float = 0.3   # 低於此信心分數觸發調整


# ============================================
# MainLoop — 核心
# ============================================

class MainLoop:
    """
    自主運行主控制器

    用法：
        from agent_framework.core.loop import MainLoop, Goal, LoopConfig

        goal = Goal(
            name="完善排灣語語料庫",
            description="提升翻譯品質和語料覆蓋率",
            success_criteria={
                "corpus_coverage": {"min": 0.8, "current": 0.0},
            },
            max_duration_hours=20,
        )

        loop = MainLoop(goal=goal)
        loop.run()  # 啟動自主運行

    也可以從 checkpoint 恢復：
        loop = MainLoop(goal=goal)
        loop.run(resume=True)
    """

    def __init__(self, goal: Goal = None, config: LoopConfig = None,
                 client: OpenAI = None, project_root: Path = None):
        self.goal = goal
        self.config = config or LoopConfig()
        self.project_root = project_root or Path(__file__).parent.parent.parent

        # 外部依賴
        self.client = client
        self.api_guard = get_api_guard()

        # 組件（延遲初始化）
        self.state_manager = StateManager(
            storage_path=self.project_root / "agent_framework" / "storage" / "state"
        )
        self.orchestrator = None  # 延遲建立

        # 當前狀態
        self.state: Optional[SystemState] = None
        self._stop_requested = False

    # ── 初始化 ──

    def _initialize_client(self):
        """初始化 LLM 客戶端"""
        if self.client:
            return

        import os
        from dotenv import load_dotenv
        load_dotenv(self.project_root / ".env")

        self.client = OpenAI(
            api_key=os.environ.get("ZHIPUAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )

    def _initialize_orchestrator(self):
        """初始化 Orchestrator 和所有 Agent"""
        if self.orchestrator:
            return

        from agents.orchestrator import OrchestratorAgent

        self._initialize_client()
        self.orchestrator = OrchestratorAgent(
            client=self.client,
            api_guard=self.api_guard,
            project_root=self.project_root,
        )
        self.orchestrator._ensure_agents()
        logger.info("[MainLoop] Orchestrator 和所有 Agent 已初始化")

    # ── 啟動入口 ──

    def run(self, resume: bool = False):
        """
        啟動自主運行

        Args:
            resume: 是否從 checkpoint 恢復
        """
        self._initialize_orchestrator()

        # 1. 初始化或恢復狀態
        if resume:
            self.state = self.state_manager.load_latest()
            if not self.state:
                logger.warning("無可恢復的 checkpoint，建立新狀態")
                self.state = self._create_initial_state()
            else:
                logger.info(f"從 checkpoint 恢復: session={self.state.session_id}, loop={self.state.loop_count}")
                # 從 checkpoint 恢復 goal
                if not self.goal and self.state.goal:
                    try:
                        self.goal = Goal.from_dict(self.state.goal)
                        logger.info(f"恢復目標: {self.goal.name}")
                    except Exception as e:
                        logger.warning(f"恢復目標失敗: {e}")
        else:
            self.state = self._create_initial_state()
            logger.info(f"新任務啟動: session={self.state.session_id}")

        # 2. 生成 Plan（Plan-Driven 模式）
        if self.goal and not self.state.plan:
            self._generate_plan()

        # 3. 註冊信號處理（優雅退出）
        self._setup_signal_handlers()

        # 4. 主循環
        logger.info("=" * 60)
        logger.info(f"  🚀 MainLoop 啟動")
        logger.info(f"  目標: {self.goal.name if self.goal else '無'}")
        logger.info(f"  最大時長: {self.goal.max_duration_hours if self.goal else '∞'}h")
        logger.info("=" * 60)

        try:
            while not self._should_stop():
                self._loop_iteration()

        except KeyboardInterrupt:
            logger.info("收到中斷信號（Ctrl+C）")
        except Exception as e:
            logger.error(f"主循環異常: {e}", exc_info=True)
            self.state.last_error = str(e)
            self.state.health_status = "critical"
        finally:
            self._graceful_shutdown()

    def _create_initial_state(self) -> SystemState:
        goal_dict = self.goal.to_dict() if self.goal else {}
        return self.state_manager.create_initial_state(goal=goal_dict)

    # ── 主循環迭代 ──

    def _loop_iteration(self):
        """單次循環迭代"""
        self.state.loop_count += 1
        iteration_start = time.time()

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Loop #{self.state.loop_count} | "
                     f"elapsed={self.state.elapsed_hours():.1f}h | "
                     f"tokens={self.state.tokens_used_total:,}")
        logger.info(f"{'─' * 50}")

        # 1. 健康檢查
        health = self._health_check()
        if health == "critical":
            logger.error("❌ 健康檢查失敗，停止運行")
            self.state.should_stop = True
            self.state.stop_reason = "health_check_failed"
            return
        elif health == "degraded":
            logger.warning("⚠️ 降級模式，延長等待")
            time.sleep(self.config.degraded_wait)

        # 2. Token 預算檢查（優雅降級：先存檔再停）
        exceeded, budget_msg = self.api_guard.is_budget_exceeded()
        if exceeded:
            logger.error(f"💰 {budget_msg}")
            self.state.should_stop = True
            self.state.stop_reason = "token_budget_exceeded"
            self.state_manager.save(self.state)
            logger.info("📦 Checkpoint 已保存，Runner 因預算耗盡停止")
            return
        elif budget_msg:
            logger.warning(f"💰 {budget_msg}")

        # 3. 檢查目標是否達成
        if self.goal and self.goal.is_achieved(self.state.metrics):
            logger.info("🎉 目標已達成！")
            self.state.should_stop = True
            self.state.stop_reason = "goal_achieved"
            return

        # 3. 檢查終止條件
        if self.goal:
            should_term, reason = self.goal.should_terminate(
                self.state.metrics,
                self.state.elapsed_hours(),
                self.state.tokens_used_total,
            )
            if should_term:
                logger.info(f"終止條件觸發: {reason}")
                self.state.should_stop = True
                self.state.stop_reason = reason
                return

        # 4. 獲取並執行下一個任務
        task_result = self._get_and_execute_next_task()

        if task_result:
            self.state.iteration_results.append({
                "loop": self.state.loop_count,
                "timestamp": time.time(),
                "result": task_result,
            })

            # 5. Self-Judge（每 N 步）
            if self.state.loop_count % self.config.judge_interval == 0:
                self._self_judge(task_result)

            # 更新 token 用量
            usage = self.api_guard.get_usage_summary()
            self.state.tokens_used_total = usage.get("total_tokens", self.state.tokens_used_total)
            self.state.llm_calls_total = usage.get("total_requests", self.state.llm_calls_total)

        else:
            # 沒有任務
            logger.info("無待執行任務，等待...")
            time.sleep(self.config.idle_wait)

        # 6. 保存 checkpoint（每次 Loop 都保存，防止 SIGKILL 丟失數據）
        self.state_manager.save(self.state)
        self.state_manager.cleanup(keep_n=10)
        self.state.last_checkpoint_at = time.time()

        # 7. 自適應等待
        iteration_ms = (time.time() - iteration_start) * 1000
        wait = self._adaptive_wait(iteration_ms)
        if wait > 0:
            time.sleep(wait)

    # ── 任務執行 ──

    def _get_and_execute_next_task(self) -> Optional[dict]:
        """獲取下一個任務並執行"""
        # 嘗試從 Plan 獲取（Plan-Driven）
        if self.state.plan and self.state.plan.get("steps"):
            plan = Plan.from_dict(self.state.plan)
            next_step = plan.get_next_step()

            if next_step:
                return self._execute_plan_step(plan, next_step)

            # Plan 完成
            if plan.is_complete():
                logger.info("📋 Plan 所有步驟已完成")
                progress = plan.get_progress()
                logger.info(f"   進度: {progress['done']}/{progress['total']} 完成, {progress['failed']} 失敗")
                self.state.plan = {}  # 清空 Plan
                return None

        # 沒有 Plan：ReAct 模式（讓 LLM 自主決定下一步）
        if self.goal:
            return self._react_step()

        return None

    def _execute_plan_step(self, plan: Plan, step: dict) -> dict:
        """執行 Plan 中的單一步驟"""
        step_id = step.get("id", "")
        step_name = step.get("name", "")
        logger.info(f"📋 執行步驟: [{step_id}] {step_name}")

        # 標記為進行中
        plan.mark_step(step_id, "in_progress")
        self.state.plan = plan.to_dict()

        try:
            # 通過 Orchestrator 執行
            msg = AgentMessage.task_assign(
                from_agent="main_loop",
                to_agent="orchestrator",
                task="execute_step",
                params=step,
            )
            response = self.orchestrator.handle_message(msg)

            if response and response.payload.get("status") == "completed":
                result_data = response.payload.get("data", {})
                plan.mark_step(step_id, "done", result=result_data)
                self.state.consecutive_failures = 0

                logger.info(f"   ✅ 完成: {step_name}")
                return {"step": step_name, "status": "done", "data": result_data}
            else:
                error = response.payload.get("error", "未知錯誤") if response else "無回應"
                plan.mark_step(step_id, "failed", error=error)
                self.state.consecutive_failures += 1

                logger.warning(f"   ❌ 失敗: {step_name} — {error}")
                return {"step": step_name, "status": "failed", "error": error}

        except Exception as e:
            plan.mark_step(step_id, "failed", error=str(e))
            self.state.consecutive_failures += 1
            logger.error(f"   ❌ 異常: {step_name} — {e}")
            return {"step": step_name, "status": "error", "error": str(e)}

        finally:
            self.state.plan = plan.to_dict()

    # 具體學習任務列表（避免模糊指令導致死循環）
    _LEARNING_TASKS = [
        "教我排灣語的謝謝怎麼說，並解釋用法",
        "masalu 是什麼意思？什麼時候用？",
        "教我排灣語的你好，還有其他打招呼的方式嗎？",
        "djavadjavai 是什麼意思？",
        "我想學排灣語的水這個詞，排灣語怎麼說？",
        "教我排灣語的早安怎麼說",
        "我想知道排灣語的朋友怎麼說，還有相關的詞嗎？",
        "請推薦我下一個應該學的排灣語詞彙",
        "查一下排灣語的對不起怎麼說",
        "我想練習排灣語的數字一到五",
        "排灣語的你好嗎怎麼說？na tarivak sun 是什麼意思？",
        "教我一個排灣語的日常用語，要有例句",
        "用排灣語說：我叫___（自我介紹）",
        "查詢我目前的學習進度",
        "再見用排灣語怎麼說？",
    ]

    def _react_step(self) -> dict:
        """ReAct 模式：讓 LLM 自主決定下一步做什麼"""
        goal_desc = self.goal.description if self.goal else "持續改進"
        progress = self.goal.get_progress(self.state.metrics) if self.goal else {}

        # 隨機挑 3 個任務當參考
        import random
        sample_tasks = random.sample(self._LEARNING_TASKS, min(3, len(self._LEARNING_TASKS)))
        task_examples = "\n".join(f"  - {t}" for t in sample_tasks)

        prompt = (
            f"你是排灣語學習系統的自主決策模組。\n\n"
            f"目標: {goal_desc}\n"
            f"當前進度: {json.dumps(progress, ensure_ascii=False)}\n"
            f"已運行: {self.state.elapsed_hours():.1f} 小時\n"
            f"Loop 次數: {self.state.loop_count}\n\n"
            f"請決定下一步。你必須輸出一個【具體的、包含明確詞彙的】學習請求。\n\n"
            f"參考方向（選一個或自己生成，但必須具體）：\n"
            f"{task_examples}\n\n"
            f"規則：\n"
            f"- 必須包含具體的中文詞或排灣語詞，不要說「一個詞」或「一個用語」\n"
            f"- 只輸出一句話，不要解釋"
        )

        try:
            llm_result = self.orchestrator.call_llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=100,
                tools=None,
                model="glm-4.5-air",
            )
            action = llm_result["message"].content.strip()
            logger.info(f"🔄 ReAct 決策: {action}")

            # 驗證：決策必須包含至少一個具體詞彙（長度 >= 5）
            if not action or len(action) < 5:
                logger.warning("ReAct 決策為空或太短，使用 fallback")
                action = random.choice(self._LEARNING_TASKS)
                logger.info(f"🔄 Fallback 決策: {action}")

            # 執行
            reply = self.orchestrator.chat(action)

            return {"mode": "react", "action": action, "reply": reply[:800]}

        except Exception as e:
            logger.error(f"ReAct 步驟失敗: {e}")
            return {"mode": "react", "status": "error", "error": str(e)}

    # ── Plan 生成 ──

    def _generate_plan(self):
        """通過 Teaching Agent 生成執行計畫"""
        logger.info("[MainLoop] 生成執行計畫...")

        msg = AgentMessage.task_assign(
            from_agent="main_loop",
            to_agent="teaching",
            task="plan_learning",
            params={
                "goal": self.goal.description,
                "level": "beginner",
            },
        )

        # 直接調用 teaching agent
        if "teaching" in self.orchestrator.bus._agents:
            response = self.orchestrator.bus.send(msg)
            if response and response.payload.get("status") == "completed":
                plan_data = response.payload.get("data", {})
                plan = Plan.from_llm_output(
                    plan_data.get("steps", []),
                    goal_name=self.goal.name,
                )
                self.state.plan = plan.to_dict()
                logger.info(f"📋 計畫已生成: {len(plan.steps)} 步")
            else:
                logger.warning("Plan 生成失敗，將使用 ReAct 模式")
        else:
            logger.warning("Teaching Agent 未註冊，將使用 ReAct 模式")

    # ── Self-Judge ──

    def _self_judge(self, last_result: dict):
        """調用 Quality Agent 進行 Self-Judge"""
        logger.info("🔍 執行 Self-Judge...")

        if "quality" not in self.orchestrator.bus._agents:
            return

        # 收集當前指標
        metrics = self.state.metrics.copy()
        # 從最近的結果中提取指標（如果有翻譯相關的）
        if last_result.get("data", {}).get("score") is not None:
            metrics["last_task_score"] = last_result["data"]["score"]

        target = {}
        if self.goal:
            for key, crit in self.goal.success_criteria.items():
                if isinstance(crit, dict):
                    target[key] = crit.get("min", 0)

        msg = AgentMessage.task_assign(
            from_agent="main_loop",
            to_agent="quality",
            task="self_judge",
            params={
                "goal": self.goal.description if self.goal else "",
                "metrics": metrics,
                "target_metrics": target,
                "iteration": self.state.loop_count,
                "max_iterations": self.config.max_loop_count,
            },
        )

        response = self.orchestrator.bus.send(msg)
        if response and response.payload.get("status") == "completed":
            judgment = response.payload.get("data", {})
            achieved = judgment.get("achieved", False)
            confidence = judgment.get("confidence", 0)
            next_action = judgment.get("next_action", "continue")

            logger.info(f"   達成: {achieved} | 信心: {confidence} | 下一步: {next_action}")

            if achieved:
                self.state.should_stop = True
                self.state.stop_reason = "self_judge_achieved"
            elif next_action == "stop":
                self.state.should_stop = True
                self.state.stop_reason = f"self_judge_stop: {judgment.get('reason', '')}"
            elif next_action == "adjust" and confidence < self.config.judge_min_confidence:
                logger.warning(f"   ⚠️ 信心過低 ({confidence})，切換到 ReAct 模式")
                self.state.plan = {}  # 清空 Plan，切回 ReAct

    # ── 健康檢查 ──

    def _health_check(self) -> str:
        """簡單健康檢查，返回 healthy/degraded/critical"""
        # 連續失敗檢查
        if self.state.consecutive_failures >= self.config.max_consecutive_failures:
            logger.error(f"連續失敗 {self.state.consecutive_failures} 次")
            return "critical"
        elif self.state.consecutive_failures >= self.config.max_consecutive_failures // 2:
            return "degraded"

        # API 限速檢查
        usage = self.api_guard.get_usage_summary(last_n_minutes=1)
        if usage.get("total_requests", 0) >= self.config.requests_per_minute_limit:
            logger.warning("API 接近限速")
            return "degraded"

        return "healthy"

    # ── 自適應等待 ──

    def _adaptive_wait(self, iteration_ms: float) -> float:
        """根據系統狀態動態調整等待時間"""
        base = self.config.base_wait

        # API 用量高 → 延長等待
        usage = self.api_guard.get_usage_summary(last_n_minutes=1)
        if usage.get("total_requests", 0) >= self.config.requests_per_minute_limit * 0.8:
            return base * 5

        # 上次迭代很慢 → 延長等待
        if iteration_ms > 10000:
            return base * 3

        # 連續失敗 → 延長等待
        if self.state.consecutive_failures > 0:
            return base * (2 ** self.state.consecutive_failures)

        return base

    # ── 停止判斷 ──

    def _should_stop(self) -> bool:
        if self._stop_requested:
            return True
        if self.state and self.state.should_stop:
            return True
        # 外部停止信號
        stop_file = self.project_root / "agent_framework" / "storage" / "STOP.signal"
        if stop_file.exists():
            logger.info("檢測到外部停止信號 (STOP.signal)")
            return True
        return False

    # ── 優雅退出 ──

    def _setup_signal_handlers(self):
        """註冊信號處理器"""
        def handle_signal(signum, frame):
            logger.info(f"收到信號 {signum}，準備優雅退出...")
            self._stop_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _graceful_shutdown(self):
        """優雅關閉：保存狀態、輸出報告"""
        logger.info("\n" + "=" * 60)
        logger.info("  🛑 MainLoop 停止")
        logger.info("=" * 60)

        if self.state:
            # 保存最終狀態
            self.state_manager.save(self.state)

            # 輸出報告
            elapsed = self.state.elapsed_hours()
            logger.info(f"  運行時間: {elapsed:.2f} 小時")
            logger.info(f"  循環次數: {self.state.loop_count}")
            logger.info(f"  LLM 調用: {self.state.llm_calls_total}")
            logger.info(f"  Token 用量: {self.state.tokens_used_total:,}")
            logger.info(f"  停止原因: {self.state.stop_reason or '用戶中斷'}")
            logger.info(f"  健康狀態: {self.state.health_status}")

            if self.goal:
                progress = self.goal.get_progress(self.state.metrics)
                logger.info(f"  目標進度: {json.dumps(progress, ensure_ascii=False, indent=2)}")

            if self.state.plan:
                plan = Plan.from_dict(self.state.plan)
                progress = plan.get_progress()
                logger.info(f"  Plan 進度: {progress['done']}/{progress['total']} 完成")

        logger.info("  狀態已保存，可使用 --resume 恢復運行")


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  🔄 MainLoop 測試（3 次迭代後停止）")
    print("=" * 60)

    goal = Goal(
        name="測試運行",
        description="測試自主運行循環",
        success_criteria={},
        max_duration_hours=0.01,  # 不到 1 分鐘就會超時停止
    )

    config = LoopConfig(
        base_wait=1.0,
        checkpoint_interval=60,
        max_loop_count=3,
    )

    loop = MainLoop(goal=goal, config=config)
    loop.run()
