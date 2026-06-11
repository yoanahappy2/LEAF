"""
strategy.py — 教學策略系統（LearningStrategy）

三種可實驗的教學策略，控制 Agent 的決策風格。
不是「人格」，而是「教學方法論」。

Mastery First:  學會再往下，答錯就複習
Exploration First: 先大量接觸，不要求立即記住
Exam Driven: 專攻高頻詞，測驗驅動

每個策略影響：
1. Orchestrator 的 system prompt（決策偏好）
2. Teaching Agent 的推薦邏輯
3. Quality Agent 的通過閾值
4. temperature 和其他 LLM 參數

作者: yu
日期: 2026-06-04
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================
# 策略定義
# ============================================

@dataclass
class LearningStrategy:
    """
    教學策略配置

    影響所有 Agent 的行為：
    - prompt_modifier: 注入到 system prompt 的策略描述
    - error_threshold: 連錯 N 次觸發策略調整
    - new_word_threshold: 連對 N 次才推新詞
    - quality_pass_threshold: Quality Agent 通過閾值
    - temperature: LLM 溫度
    - review_interval: 每 N 個詞複習一次
    - max_new_per_session: 單次最多推幾個新詞
    """
    name: str
    display_name: str
    description: str

    # 策略參數
    prompt_modifier: str = ""          # 注入 system prompt
    error_threshold: int = 2           # 連錯 N 次 → 調整
    new_word_threshold: int = 2        # 連對 N 次 → 推新詞
    quality_pass_threshold: float = 0.7  # QA 通過閾值
    temperature: float = 0.4           # LLM 溫度
    review_interval: int = 3           # 每 N 個詞複習
    max_new_per_session: int = 5       # 單次最多新詞
    prioritize_high_freq: bool = False  # 優先高頻詞
    allow_skip: bool = True            # 允許跳過
    retry_on_error: bool = True        # 答錯重試

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "error_threshold": self.error_threshold,
            "new_word_threshold": self.new_word_threshold,
            "quality_pass_threshold": self.quality_pass_threshold,
            "temperature": self.temperature,
            "review_interval": self.review_interval,
            "max_new_per_session": self.max_new_per_session,
            "prioritize_high_freq": self.prioritize_high_freq,
            "allow_skip": self.allow_skip,
            "retry_on_error": self.retry_on_error,
        }


# ============================================
# 三種預定義策略
# ============================================

MASTERY_FIRST = LearningStrategy(
    name="mastery_first",
    display_name="Mastery First（精熟優先）",
    description="學會再往下，確保每個詞都掌握",
    prompt_modifier=(
        "## 教學策略：精熟優先（Mastery First）\n\n"
        "核心原則：確保學生完全掌握每個詞彙後才推進新內容。\n\n"
        "決策規則：\n"
        "- 學生答對 2 次以上 → 推薦下一個新詞\n"
        "- 學生答錯 → 立即回到該詞複習，不推新詞\n"
        "- 連錯 2 次 → 降級到更簡單的詞或提供額外例句\n"
        "- 每 3 個詞插入一次複習測驗\n"
        "- 單次最多教 5 個新詞\n"
        "- 優先選擇與已學詞相關的新詞（建立聯想記憶）\n\n"
        "推薦邏輯：優先複習薄弱詞 > 推薦相關新詞 > 推薦全新主題\n"
    ),
    error_threshold=2,
    new_word_threshold=2,
    quality_pass_threshold=0.8,
    temperature=0.3,
    review_interval=3,
    max_new_per_session=5,
    prioritize_high_freq=False,
    allow_skip=False,
    retry_on_error=True,
)

EXPLORATION_FIRST = LearningStrategy(
    name="exploration_first",
    display_name="Exploration First（探索優先）",
    description="先大量接觸，培養語感，不要求立即記住",
    prompt_modifier=(
        "## 教學策略：探索優先（Exploration First）\n\n"
        "核心原則：讓學生大量接觸新詞彙和例句，培養語感，不要求立即記住。\n\n"
        "決策規則：\n"
        "- 即使學生答錯，也繼續推新詞（錯誤是學習的一部分）\n"
        "- 每次對話至少介紹 2 個新詞\n"
        "- 單次最多教 10 個新詞\n"
        "- 答錯時給予鼓勵和解釋，但不強制複習\n"
        "- 穿插不同主題的詞彙（增加語言接觸面的廣度）\n"
        "- 偶爾回頭提及之前學過的詞（間隔重現，但不中斷探索節奏）\n\n"
        "推薦邏輯：推薦有趣的新主題 > 跨主題探索 > 複習（僅在自然提及時）\n"
    ),
    error_threshold=5,
    new_word_threshold=1,
    quality_pass_threshold=0.6,
    temperature=0.6,
    review_interval=8,
    max_new_per_session=10,
    prioritize_high_freq=False,
    allow_skip=True,
    retry_on_error=False,
)

EXAM_DRIVEN = LearningStrategy(
    name="exam_driven",
    display_name="Exam Driven（測驗驅動）",
    description="專攻高頻詞，測驗驅動學習",
    prompt_modifier=(
        "## 教學策略：測驗驅動（Exam Driven）\n\n"
        "核心原則：以測驗為主要學習手段，專攻高頻詞彙，追求最高效率。\n\n"
        "決策規則：\n"
        "- 優先教授語料庫中頻率最高的詞彙\n"
        "- 每教 2 個詞立即出測驗題\n"
        "- 測驗題目類型多樣化（中→排灣、排灣→中、聽力選擇）\n"
        "- 答錯的詞自動加入高頻複習佇列\n"
        "- 不浪費時間在低頻詞上\n"
        "- 追蹤正確率，動態調整難度\n\n"
        "推薦邏輯：最高頻未學詞 > 答錯待複習詞 > 測驗 > 下一批高頻詞\n"
    ),
    error_threshold=1,
    new_word_threshold=1,
    quality_pass_threshold=0.9,
    temperature=0.2,
    review_interval=2,
    max_new_per_session=8,
    prioritize_high_freq=True,
    allow_skip=False,
    retry_on_error=True,
)


# ============================================
# 策略管理器
# ============================================

STRATEGIES = {
    "mastery_first": MASTERY_FIRST,
    "exploration_first": EXPLORATION_FIRST,
    "exam_driven": EXAM_DRIVEN,
}


def get_strategy(name: str) -> LearningStrategy:
    """取得策略"""
    if name not in STRATEGIES:
        raise ValueError(f"未知策略: {name}，可用: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def list_strategies() -> list[dict]:
    """列出所有策略"""
    return [
        {
            "name": s.name,
            "display_name": s.display_name,
            "description": s.description,
            "error_threshold": s.error_threshold,
            "new_word_threshold": s.new_word_threshold,
            "max_new_per_session": s.max_new_per_session,
        }
        for s in STRATEGIES.values()
    ]


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  📊 LearningStrategy 測試")
    print("=" * 60)

    # 列出策略
    print("\n--- 可用策略 ---")
    for s in list_strategies():
        print(f"  {s['name']}: {s['description']}")

    # 測試各策略參數
    print("\n--- 策略參數對比 ---")
    for name, s in STRATEGIES.items():
        print(f"\n  [{s.display_name}]")
        print(f"    error_threshold: {s.error_threshold}")
        print(f"    new_word_threshold: {s.new_word_threshold}")
        print(f"    quality_pass_threshold: {s.quality_pass_threshold}")
        print(f"    temperature: {s.temperature}")
        print(f"    max_new_per_session: {s.max_new_per_session}")
        print(f"    prompt_modifier 長度: {len(s.prompt_modifier)} chars")

    # 測試取得策略
    print("\n--- get_strategy ---")
    s = get_strategy("mastery_first")
    print(f"  取得: {s.display_name}")

    try:
        get_strategy("nonexistent")
    except ValueError as e:
        print(f"  錯誤處理: {e}")

    print("\n✅ LearningStrategy 測試通過")
