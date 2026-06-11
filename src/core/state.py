"""
state.py — 系統狀態持久化

保存和恢復 Autonomous Runner 的運行狀態，
讓系統可以從中斷點繼續運行。

存儲格式：JSON 文件
存儲位置：agent_framework/storage/state/

作者: 地陪
日期: 2026-05-12
"""

import json
import time
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """
    系統運行狀態（用於持久化和恢復）

    包含：
    - 當前 Goal 和 Plan
    - 運行指標（token 用量、LLM 調用次數）
    - 健康狀態
    - Loop 控制標記
    """
    # 基本資訊
    session_id: str = ""
    goal: dict = field(default_factory=dict)       # Goal 的 dict 表示
    plan: dict = field(default_factory=dict)       # Plan 的 dict 表示
    started_at: float = 0
    last_checkpoint_at: float = 0

    # 運行指標
    metrics: dict = field(default_factory=dict)
    llm_calls_total: int = 0
    tokens_used_total: int = 0

    # 健康狀態
    health_status: str = "healthy"
    consecutive_failures: int = 0
    last_error: str = ""

    # Loop 控制
    loop_count: int = 0
    should_stop: bool = False
    stop_reason: str = ""
    iteration_results: list = field(default_factory=list)

    def elapsed_hours(self) -> float:
        if not self.started_at:
            return 0
        return (time.time() - self.started_at) / 3600

    def to_dict(self) -> dict:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, list):
                # 每個元素獨立序列化，避免嵌套 dataclass 遞迴
                safe_items = []
                for item in v:
                    try:
                        safe_items.append(json.loads(json.dumps(item, default=str, ensure_ascii=False)))
                    except (TypeError, ValueError, RecursionError):
                        safe_items.append(str(item)[:500])
                d[k] = safe_items
            elif isinstance(v, dict):
                try:
                    d[k] = json.loads(json.dumps(v, default=str, ensure_ascii=False))
                except (TypeError, ValueError, RecursionError):
                    d[k] = {}
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SystemState":
        fields = {}
        for k, v in data.items():
            if k not in cls.__dataclass_fields__:
                continue
            # 類型修正：確保 list/dict 不會是 str
            if k == "iteration_results" and isinstance(v, str):
                v = []  # 反序列化失敗就重置
            fields[k] = v
        return cls(**fields)


class StateManager:
    """
    狀態持久化管理器

    功能：
    1. 保存狀態到 JSON 文件（checkpoint）
    2. 從最新 checkpoint 恢復
    3. 列出歷史 checkpoint
    4. 清理舊 checkpoint
    """

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path(__file__).parent / "storage" / "state"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, state: SystemState) -> str:
        """保存狀態，返回 checkpoint_id"""
        state.last_checkpoint_at = time.time()
        checkpoint_id = f"ckpt_{state.session_id}_{int(time.time())}"
        path = self.storage_path / f"{checkpoint_id}.json"

        data = {
            "id": checkpoint_id,
            "saved_at": time.time(),
            "state": state.to_dict(),
        }

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

        # 更新 latest 指標
        latest = self.storage_path / "latest.json"
        latest.write_text(json.dumps({"checkpoint_id": checkpoint_id}))

        logger.info(f"[StateManager] 已保存 checkpoint: {checkpoint_id}")
        return checkpoint_id

    def load_latest(self) -> Optional[SystemState]:
        """載入最新的 checkpoint"""
        latest_path = self.storage_path / "latest.json"
        if not latest_path.exists():
            return None

        try:
            latest = json.loads(latest_path.read_text())
            ckpt_id = latest["checkpoint_id"]
            ckpt_path = self.storage_path / f"{ckpt_id}.json"
            data = json.loads(ckpt_path.read_text())
            return SystemState.from_dict(data["state"])
        except Exception as e:
            logger.error(f"[StateManager] 載入 checkpoint 失敗: {e}")
            return None

    def list_checkpoints(self) -> list:
        """列出所有 checkpoint"""
        ckpts = []
        for f in sorted(self.storage_path.glob("ckpt_*.json")):
            try:
                data = json.loads(f.read_text())
                ckpts.append({
                    "id": data["id"],
                    "saved_at": data["saved_at"],
                    "session_id": data["state"].get("session_id", ""),
                    "loop_count": data["state"].get("loop_count", 0),
                })
            except Exception:
                pass
        return ckpts

    def cleanup(self, keep_n: int = 10):
        """只保留最近 N 個 checkpoint"""
        ckpts = sorted(self.storage_path.glob("ckpt_*.json"))
        if len(ckpts) > keep_n:
            for old in ckpts[:-keep_n]:
                old.unlink(missing_ok=True)
            logger.info(f"[StateManager] 清理了 {len(ckpts) - keep_n} 個舊 checkpoint")

    def create_initial_state(self, goal: dict, plan: dict = None) -> SystemState:
        """建立初始狀態"""
        return SystemState(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            goal=goal,
            plan=plan or {},
            started_at=time.time(),
            last_checkpoint_at=time.time(),
            metrics={},
        )


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  💾 StateManager 測試")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        sm = StateManager(storage_path=Path(tmp))

        # 建立初始狀態
        state = sm.create_initial_state(
            goal={"name": "測試目標", "success_criteria": {}},
            plan={"steps": [{"id": "s1", "status": "pending"}]},
        )
        print(f"\n  Session: {state.session_id}")

        # 保存
        ckpt_id = sm.save(state)
        print(f"  Checkpoint: {ckpt_id}")

        # 更新狀態
        state.loop_count = 5
        state.tokens_used_total = 1000
        sm.save(state)

        # 載入
        restored = sm.load_latest()
        assert restored is not None
        assert restored.loop_count == 5
        assert restored.tokens_used_total == 1000
        print(f"  載入: loop={restored.loop_count}, tokens={restored.tokens_used_total}")

        # 列出
        ckpts = sm.list_checkpoints()
        print(f"  Checkpoint 數: {len(ckpts)}")

        print("\n✅ StateManager 測試通過")
