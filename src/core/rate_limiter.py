"""
rate_limiter.py — API 調用保護機制

功能：
1. 指數退避重試（Exponential Backoff）
2. 模型選擇策略（根據任務複雜度自動選模型）
3. Token 用量追蹤
4. 請求速率限制（防止 burst）

線 A（期末展示）：簡單版，保證穩定。
線 B（暑假）：可加令牌桶、持久化計量。

作者: 地陪
日期: 2026-05-12
"""

import time
import logging
import threading
from typing import Callable, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================
# 模型配置
# ============================================

# 預設模型常數（具體值從 paiwan_config/config.yaml 讀取，這裡是 fallback）
# ⚠️ 2026-05-13: 全面切換到 glm-4.5-air（使用免費 1200 萬 token 資源包）
# 所有任務統一用 glm-4.5-air，避免消耗付費的 glm-4-plus 額度
# 如需更強推理能力，可手動覆寫為 glm-4.7 或 glm-5
DEFAULT_MODELS = {
    "fast": "glm-4.5-air",    # 快速回應、簡單決策
    "standard": "glm-4.5-air",  # 標準品質、翻譯、教學
    "complex": "glm-4.5-air",   # 複雜推理、規劃、品質評估
}

# Token 預算硬上限（1200 萬資源包，保留 10% 安全餘量）
TOKEN_BUDGET_HARD_LIMIT = 10_800_000  # 1200 萬 × 90%


# ============================================
# APIGuard — 核心 API 保護
# ============================================

class APIGuard:
    """
    API 調用保護機制

    用法：
        guard = APIGuard(max_retries=3, base_delay=2.0)

        # 帶重試的調用
        result = guard.call_with_retry(llm_client.chat.completions.create, **kwargs)

        # 根據任務複雜度選模型
        model = guard.pick_model("complex")

        # 記錄 token 用量
        guard.record_usage(model="glm-4.5-air", tokens=500)
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 2.0,
                 models: dict = None):
        """
        Args:
            max_retries: 最大重試次數
            base_delay: 首次重試等待秒數（之後指數增長）
            models: 模型名稱映射 {"fast": ..., "standard": ..., "complex": ...}
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.models = models or DEFAULT_MODELS

        # Token 用量追蹤（線程安全）
        self._usage_lock = threading.Lock()
        self._usage: list[dict] = []
        self._max_usage_records = 500
        self._total_tokens_cumulative = 0  # 累計總量（不受列表截斷影響）

        # 速率限制
        self._rate_lock = threading.Lock()
        self._request_timestamps: list[float] = []
        self._max_requests_per_minute = 60  # 智譜 API 限制

    # ── 重試機制 ──

    def call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        帶指數退避的重試調用

        Args:
            func: 要調用的函數（通常是 LLM API 調用）
            *args, **kwargs: 傳給 func 的參數

        Returns:
            func 的返回值

        Raises:
            最後一次的異常（如果所有重試都失敗）
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # 速率限制檢查
                self._wait_for_rate_limit()

                result = func(*args, **kwargs)

                # 成功：記錄 token 用量
                if hasattr(result, 'usage') and result.usage:
                    self.record_usage(
                        model=kwargs.get('model', 'unknown'),
                        tokens=result.usage.total_tokens,
                    )

                return result

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # 不可重試的錯誤類型
                non_retryable = ["invalid api key", "authentication", "permission denied"]
                if any(nr in error_str for nr in non_retryable):
                    logger.error(f"[APIGuard] 不可重試的錯誤: {e}")
                    raise

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)  # 2s → 4s → 8s
                    # 加入隨機抖動避免雷群效應
                    jitter = delay * 0.1 * (hash(str(time.time())) % 10) / 10
                    actual_delay = delay + jitter

                    logger.warning(
                        f"[APIGuard] 調用失敗（第 {attempt + 1}/{self.max_retries} 次），"
                        f"{actual_delay:.1f}s 後重試: {e}"
                    )
                    time.sleep(actual_delay)
                else:
                    logger.error(
                        f"[APIGuard] 調用失敗，已重試 {self.max_retries} 次: {e}"
                    )

        raise last_error

    # ── 模型選擇 ──

    def pick_model(self, complexity: str = "standard") -> str:
        """
        根據任務複雜度選擇模型

        Args:
            complexity: "fast" | "standard" | "complex"
                - fast: 簡單判斷、格式解析、self-judge
                - standard: 一般對話、翻譯、教學
                - complex: 複雜推理、規劃、品質評估

        Returns:
            模型名稱字串
        """
        return self.models.get(complexity, self.models["standard"])

    def pick_model_for_agent(self, agent_role: str) -> str:
        """
        根據 Agent 角色選擇預設模型

        不同 Agent 有不同的模型需求：
        - orchestrator: complex（需要理解意圖、規劃）
        - knowledge: standard（翻譯、檢索）
        - teaching: standard（教學策略）
        - quality: complex（品質評估需要推理）
        """
        agent_model_map = {
            "orchestrator": "complex",
            "knowledge": "standard",
            "teaching": "standard",
            "quality": "complex",
        }
        complexity = agent_model_map.get(agent_role, "standard")
        return self.pick_model(complexity)

    # ── 用量追蹤 ──

    def record_usage(self, model: str, tokens: int, agent: str = None):
        """記錄 token 用量"""
        with self._usage_lock:
            self._total_tokens_cumulative += tokens
            self._usage.append({
                "timestamp": time.time(),
                "model": model,
                "tokens": tokens,
                "agent": agent or "unknown",
            })
            # 限制記錄數量（但累計總量不受影響）
            if len(self._usage) > self._max_usage_records:
                self._usage = self._usage[-self._max_usage_records:]

    def get_total_tokens_used(self) -> int:
        """查詢累計消耗的 token 總數（不受列表截斷影響）"""
        return self._total_tokens_cumulative

    def is_budget_exceeded(self) -> tuple:
        """檢查是否超過 token 預算硬上限
        Returns: (exceeded: bool, message: str)
        """
        total = self.get_total_tokens_used()
        if total >= TOKEN_BUDGET_HARD_LIMIT:
            return True, f"Token 預算已耗盡 ({total:,}/{TOKEN_BUDGET_HARD_LIMIT:,})，請充值後繼續"
        elif total >= TOKEN_BUDGET_HARD_LIMIT * 0.9:
            return False, f"⚠️ Token 預算警告: 已用 {total:,}/{TOKEN_BUDGET_HARD_LIMIT:,} ({total*100//TOKEN_BUDGET_HARD_LIMIT}%)"
        return False, ""

    def get_usage_summary(self, last_n_minutes: int = None) -> dict:
        """
        獲取用量統計

        Args:
            last_n_minutes: 只統計最近 N 分鐘，None 表示全部

        Returns:
            {
                "total_tokens": int,
                "total_requests": int,
                "by_model": {"glm-4.5-air": {"tokens": ..., "requests": ...}},
                "by_agent": {"orchestrator": {"tokens": ..., "requests": ...}},
            }
        """
        with self._usage_lock:
            records = self._usage

        if last_n_minutes:
            cutoff = time.time() - last_n_minutes * 60
            records = [r for r in records if r["timestamp"] >= cutoff]

        if not records:
            return {"total_tokens": 0, "total_requests": 0, "by_model": {}, "by_agent": {}}

        by_model: dict[str, dict] = {}
        by_agent: dict[str, dict] = {}
        total_tokens = 0

        for r in records:
            total_tokens += r["tokens"]

            # 按模型
            m = r["model"]
            if m not in by_model:
                by_model[m] = {"tokens": 0, "requests": 0}
            by_model[m]["tokens"] += r["tokens"]
            by_model[m]["requests"] += 1

            # 按 Agent
            a = r["agent"]
            if a not in by_agent:
                by_agent[a] = {"tokens": 0, "requests": 0}
            by_agent[a]["tokens"] += r["tokens"]
            by_agent[a]["requests"] += 1

        return {
            "total_tokens": total_tokens,
            "total_requests": len(records),
            "by_model": by_model,
            "by_agent": by_agent,
        }

    # ── 速率限制 ──

    def _wait_for_rate_limit(self):
        """確保不超過每分鐘請求數限制"""
        with self._rate_lock:
            now = time.time()
            cutoff = now - 60  # 1 分鐘窗口

            # 清理過期時間戳
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts >= cutoff
            ]

            if len(self._request_timestamps) >= self._max_requests_per_minute:
                # 計算需要等待多久
                oldest = self._request_timestamps[0]
                wait_time = oldest + 60 - now + 0.1  # 多等 0.1s 保險
                if wait_time > 0:
                    logger.warning(f"[APIGuard] 速率限制，等待 {wait_time:.1f}s")
                    time.sleep(wait_time)
                    now = time.time()

            self._request_timestamps.append(now)

    # ── 重設 ──

    def reset(self):
        """重設所有計量"""
        with self._usage_lock:
            self._usage.clear()
        with self._rate_lock:
            self._request_timestamps.clear()


# ============================================
# 全域實例（懶載入）
# ============================================

_global_guard: Optional[APIGuard] = None


def get_api_guard(models: dict = None) -> APIGuard:
    """獲取全域 APIGuard 實例"""
    global _global_guard
    if _global_guard is None:
        _global_guard = APIGuard(models=models)
    return _global_guard


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  🛡️ APIGuard 測試")
    print("=" * 60)

    guard = APIGuard(max_retries=3, base_delay=0.1)  # 測試用短延遲

    # 測試 1：模型選擇
    print("\n--- 模型選擇 ---")
    for complexity in ["fast", "standard", "complex"]:
        model = guard.pick_model(complexity)
        print(f"  {complexity} → {model}")

    for role in ["orchestrator", "knowledge", "teaching", "quality"]:
        model = guard.pick_model_for_agent(role)
        print(f"  {role} → {model}")

    # 測試 2：重試機制
    print("\n--- 重試機制 ---")
    call_count = [0]

    def mock_api_call():
        """模擬前兩次失敗、第三次成功的 API 調用"""
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"模擬網路錯誤（第 {call_count[0]} 次）")
        return {"status": "ok", "data": "success"}

    result = guard.call_with_retry(mock_api_call)
    print(f"  結果: {result}")
    print(f"  總調用次數: {call_count[0]}")

    # 測試 3：用量追蹤
    print("\n--- 用量追蹤 ---")
    guard.record_usage("glm-4.5-air", 500, "orchestrator")
    guard.record_usage("glm-4.5-air", 1200, "knowledge")
    guard.record_usage("glm-4.5-air", 300, "orchestrator")

    summary = guard.get_usage_summary()
    print(f"  總 token: {summary['total_tokens']}")
    print(f"  總請求數: {summary['total_requests']}")
    print(f"  按模型: {summary['by_model']}")
    print(f"  按 Agent: {summary['by_agent']}")

    # 測試 4：重試耗盡
    print("\n--- 重試耗盡 ---")
    try:
        guard.call_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("永久失敗")))
    except RuntimeError as e:
        print(f"  ✅ 正確拋出異常: {e}")

    print("\n✅ 所有測試通過")
