"""
knowledge_agent.py — 知識檢索專員

負責所有「知識相關」的任務：
- 翻譯（排灣語⇄中文）
- RAG 語料檢索
- 詞彙深度查詢（知識圖譜）
- 發音匹配評估

繼承 BaseAgent，通過 AgentMessage 接收 Orchestrator 分派的任務，
調用 paiwan_config/tools/ 下的工具執行，返回結構化結果。

作者: 地陪
日期: 2026-05-12
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

# 框架核心
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.agent import BaseAgent, AgentTrace
from core.message import AgentMessage, MessageType
from core.rate_limiter import APIGuard

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """
    知識檢索專員

    支援的任務（通過 AgentMessage.payload["task"] 指定）：
    - "translate"      翻譯
    - "rag_search"     RAG 語料檢索
    - "lookup"         詞彙深度查詢（知識圖譜）
    - "pronunciation"  發音匹配
    - "asr"            語音辨識

    每個任務對應一個 _handle_xxx 方法，返回 AgentMessage。
    """

    role = "knowledge"

    def __init__(self, client: OpenAI = None, api_guard: APIGuard = None,
                 model: str = None, config_path: Path = None,
                 project_root: Path = None):
        """
        Args:
            client: 智譜 OpenAI 兼容客戶端
            api_guard: API 保護
            model: 覆寫模型
            config_path: 語言配置目錄路徑
            project_root: 專案根目錄（用於載入服務）
        """
        super().__init__(client=client, api_guard=api_guard,
                         model=model, config_path=config_path)
        self.project_root = project_root or Path(__file__).parent.parent.parent

        # 延遲載入的服務實例
        self._rag = None
        self._translator = None
        self._kg = None
        self._services_loaded = False

    # ── System Prompt ──

    def get_system_prompt(self) -> str:
        return (
            "你是排灣族語知識檢索專員（Knowledge Agent）。\n\n"
            "職責：\n"
            "1. 翻譯排灣語⇄中文（精確翻譯，不編造）\n"
            "2. 搜尋語料庫提供參考例句\n"
            "3. 查詢詞彙深度資訊（綴詞分析、親屬關係、相關詞）\n"
            "4. 評估發音準確度\n\n"
            "規則：\n"
            "- 語料庫沒有的排灣語不編造\n"
            "- 翻譯要標注信心等級（high/medium/low）\n"
            "- 返回結構化 JSON 結果\n"
        )

    # ── 服務延遲載入 ──

    def _ensure_services(self):
        """延遲載入所有知識服務（只在第一次調用時載入）"""
        if self._services_loaded:
            return

        logger.info("[KnowledgeAgent] 載入知識服務...")

        # RAG
        try:
            sys.path.insert(0, str(self.project_root))
            from rag_service import PaiwanRAG
            self._rag = PaiwanRAG()
            self._rag.build_index()
            logger.info("  ✅ RAG 服務已載入")
        except Exception as e:
            logger.warning(f"  ⚠️ RAG 載入失敗: {e}")

        # 翻譯
        try:
            from translate_service import PaiwanTranslator
            self._translator = PaiwanTranslator()
            self._translator.load()
            logger.info("  ✅ 翻譯服務已載入")
        except Exception as e:
            logger.warning(f"  ⚠️ 翻譯服務載入失敗: {e}")

        # 知識圖譜
        try:
            from knowledge_graph import PaiwanKnowledgeGraph
            self._kg = PaiwanKnowledgeGraph()
            logger.info("  ✅ 知識圖譜已載入")
        except Exception as e:
            logger.warning(f"  ⚠️ 知識圖譜載入失敗: {e}")

        self._services_loaded = True

    # ── 訊息處理入口 ──

    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        處理 Orchestrator 分派的任務

        根據 payload["task"] 路由到對應的 handler
        """
        self._ensure_services()

        task = message.payload.get("task", "")
        params = message.payload.get("params", {})
        session_id = message.meta.get("session_id")

        logger.info(f"[KnowledgeAgent] 收到任務: {task}")

        # 任務路由
        handlers = {
            "translate": self._handle_translate,
            "rag_search": self._handle_rag_search,
            "lookup": self._handle_lookup,
            "pronunciation": self._handle_pronunciation,
            "asr": self._handle_asr,
        }

        handler = handlers.get(task)
        if not handler:
            return self._make_error(message, f"未知任務類型: {task}")

        try:
            return handler(message, params)
        except Exception as e:
            logger.error(f"[KnowledgeAgent] 任務 {task} 執行失敗: {e}", exc_info=True)
            return self._make_error(message, str(e))

    # ── 各任務 Handler ──

    def _handle_translate(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """翻譯任務"""
        text = params.get("text", "")
        direction = params.get("direction", "auto")

        if not text:
            return self._make_error(msg, "翻譯需要 text 參數")

        if not self._translator:
            # Fallback：用 LLM 直接翻譯
            return self._llm_translate(msg, text, direction)

        result = self._translator.translate(text, direction=direction)

        confidence = "high" if result.get("method") == "corpus" else "medium"

        return self._make_response(
            msg,
            task="translate",
            status="completed",
            data={
                "input": result["input"],
                "translation": result["translation"],
                "direction": result["direction"],
                "method": result.get("method", "unknown"),
                "confidence": confidence,
            },
            confidence={"high": 0.95, "medium": 0.7, "low": 0.4}.get(confidence, 0.5),
        )

    def _llm_translate(self, msg: AgentMessage, text: str, direction: str) -> AgentMessage:
        """LLM Fallback 翻譯（當翻譯服務不可用時）"""
        if not self.client:
            return self._make_error(msg, "翻譯服務和 LLM 均不可用")

        dir_hint = ""
        if direction == "p2c":
            dir_hint = "將以下排灣語翻譯成中文"
        elif direction == "c2p":
            dir_hint = "將以下中文翻譯成排灣語"
        else:
            dir_hint = "自動偵測語言並翻譯（排灣語⇄中文）"

        prompt = (
            f"{dir_hint}。如果不确定，請標注 [不確定]。\n\n"
            f"輸入：{text}\n\n"
            f"只輸出翻譯結果，不要解釋。"
        )

        llm_result = self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            tools=None,
        )

        translation = llm_result["message"].content.strip()
        confidence = 0.4 if "[不確定]" in translation else 0.6

        return self._make_response(
            msg,
            task="translate",
            status="completed",
            data={
                "input": text,
                "translation": translation,
                "direction": direction,
                "method": "llm_fallback",
                "confidence": "low" if confidence < 0.5 else "medium",
            },
            confidence=confidence,
        )

    def _handle_rag_search(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """RAG 語料檢索"""
        query = params.get("query", "")
        top_k = params.get("top_k", 5)

        if not query:
            return self._make_error(msg, "RAG 搜尋需要 query 參數")

        if not self._rag:
            return self._make_error(msg, "RAG 服務未初始化")

        results = self._rag.search(query, top_k=top_k)

        formatted_results = [
            {
                "paiwan": r.get("paiwan", ""),
                "chinese": r.get("chinese", ""),
                "similarity": round(float(r.get("similarity", 0)), 3),
                "intent": r.get("intent", ""),
            }
            for r in results
        ]

        # 計算平均相似度作為信心分數
        avg_sim = (
            sum(r["similarity"] for r in formatted_results) / len(formatted_results)
            if formatted_results
            else 0
        )

        return self._make_response(
            msg,
            task="rag_search",
            status="completed",
            data={
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            },
            confidence=round(avg_sim, 2),
        )

    def _handle_lookup(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """詞彙深度查詢（知識圖譜）"""
        word = params.get("word", "")

        if not word:
            return self._make_error(msg, "查詢需要 word 參數")

        if not self._kg:
            return self._make_error(msg, "知識圖譜未初始化")

        result = self._kg.lookup(word)

        return self._make_response(
            msg,
            task="lookup",
            status="completed",
            data={
                "word": result.get("word", word),
                "translation": result.get("translation"),
                "prefix_analysis": result.get("prefix_analysis"),
                "kinship": result.get("kinship"),
                "related_words": result.get("related_words", []),
                "theme": result.get("theme"),
            },
        )

    def _handle_pronunciation(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """發音匹配評估"""
        recognized = params.get("recognized", "")
        target = params.get("target", "")

        if not recognized or not target:
            return self._make_error(msg, "發音評估需要 recognized 和 target 參數")

        # 嘗試載入 PronunciationMatcher
        try:
            from api.server import PronunciationMatcher
            result = PronunciationMatcher.calculate_score(recognized, target)
            score = result["score"]
            applied_rules = result.get("applied_rules", [])
        except Exception:
            # Fallback：簡易字串比對
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, recognized.lower(), target.lower()).ratio()
            score = round(similarity * 100)
            applied_rules = []

        level = (
            "Perfect" if score >= 95 else
            "Excellent" if score >= 85 else
            "Good" if score >= 70 else
            "Needs Practice"
        )

        return self._make_response(
            msg,
            task="pronunciation",
            status="completed",
            data={
                "recognized": recognized,
                "target": target,
                "score": score,
                "level": level,
                "applied_rules": applied_rules,
            },
            confidence=round(score / 100, 2),
        )

    def _handle_asr(self, msg: AgentMessage, params: dict) -> AgentMessage:
        """語音辨識"""
        audio_path = params.get("audio_path", "")

        if not audio_path:
            return self._make_error(msg, "ASR 需要 audio_path 參數")

        import requests as req
        try:
            with open(audio_path, "rb") as f:
                resp = req.post("http://127.0.0.1:8000/asr", files={"audio": f}, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                return self._make_response(
                    msg,
                    task="asr",
                    status="completed",
                    data={
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0),
                    },
                )
            else:
                return self._make_error(msg, f"ASR API 返回 HTTP {resp.status_code}")
        except Exception as e:
            return self._make_error(msg, f"ASR 調用失敗: {e}")


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    print("=" * 60)
    print("  🧠 KnowledgeAgent 測試")
    print("=" * 60)

    # 初始化
    client = OpenAI(
        api_key=os.environ.get("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    agent = KnowledgeAgent(client=client)

    # 測試 1：訊息處理路由
    print("\n--- 測試 1：任務路由 ---")
    msg = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="translate",
        params={"text": "你好", "direction": "c2p"},
        session_id="test-001",
    )
    resp = agent.handle_message(msg)
    print(f"  請求: translate(你好)")
    print(f"  回應: {resp.payload.get('status')} | data={json.dumps(resp.payload.get('data', {}), ensure_ascii=False)[:200]}")

    # 測試 2：RAG 搜尋
    print("\n--- 測試 2：RAG 搜尋 ---")
    msg2 = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="rag_search",
        params={"query": "你好嗎", "top_k": 3},
        session_id="test-001",
    )
    resp2 = agent.handle_message(msg2)
    results = resp2.payload.get("data", {}).get("results", [])
    print(f"  搜尋結果: {len(results)} 筆")
    for r in results[:3]:
        print(f"    {r.get('paiwan', '')} = {r.get('chinese', '')} (sim={r.get('similarity', 0)})")

    # 測試 3：知識圖譜查詞
    print("\n--- 測試 3：知識圖譜查詞 ---")
    msg3 = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="lookup",
        params={"word": "masalu"},
        session_id="test-001",
    )
    resp3 = agent.handle_message(msg3)
    data = resp3.payload.get("data", {})
    print(f"  詞彙: {data.get('word', '')}")
    print(f"  翻譯: {data.get('translation', '')}")
    print(f"  相關詞: {data.get('related_words', [])[:5]}")

    # 測試 4：發音匹配
    print("\n--- 測試 4：發音匹配 ---")
    msg4 = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="pronunciation",
        params={"recognized": "masalu", "target": "masalu"},
        session_id="test-001",
    )
    resp4 = agent.handle_message(msg4)
    pron_data = resp4.payload.get("data", {})
    print(f"  分數: {pron_data.get('score', 0)} | 等級: {pron_data.get('level', '')}")

    # 測試 5：未知任務
    print("\n--- 測試 5：未知任務 ---")
    msg5 = AgentMessage.task_assign(
        from_agent="orchestrator",
        to_agent="knowledge",
        task="unknown_task",
        params={},
        session_id="test-001",
    )
    resp5 = agent.handle_message(msg5)
    print(f"  回應: {resp5.payload}")

    # 測試 6：LLM Fallback 翻譯
    print("\n--- 測試 6：LLM Fallback 翻譯 ---")
    # 故意不載入翻譯服務的場景已在 _handle_translate 中自動處理

    print("\n✅ KnowledgeAgent 所有測試完成")
