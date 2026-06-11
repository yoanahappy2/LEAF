"""
llm_service.py
語聲同行 2.0 — LLM 對話服務（RAG 重構版）

核心改動：
- 移除 117 句硬編碼語料
- 整合 rag_service.py 進行語意檢索
- System Prompt 精簡化，RAG 動態注入 context
- 支援雙模式：RAG（預設）+ 硬編碼（fallback）

作者: Backend_Dev
日期: 2026-04-15
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_api_key_here":
    raise ValueError(
        "請在 .env 檔案中填入你的智譜 API Key\n"
        "  1. 打開 ~/Desktop/paiwan_competition_2026/.env\n"
        "  2. 將 your_api_key_here 替換為你的 key\n"
        "  3. 重新執行"
   )

client = OpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# ============================================
# 模型配置
# ============================================

MODEL_FAST = "glm-4.5-air"  # 2026-05-30: flash/plus 資源包已用罄，統一走 4.5-air
MODEL_PLUS = "glm-4.5-air"
MODEL_DEFAULT = MODEL_FAST

# ============================================
# System Prompt（RAG 版 — 精簡核心 + 動態 context）
# ============================================

SYSTEM_PROMPT_TEMPLATE = """你是 vuvu Maliq（排灣族的祖母），一位慈祥、溫暖、熱愛部落文化的排灣族長輩。

## 你的性格
- 你非常疼愛孫子孫女，每次跟他們說話都充滿喜悅
- 你對排灣族的文化、傳統、故事如數家珍
- 你說話溫柔但有智慧，像在火爐邊講故事一樣
- 你偶爾會用感嘆詞「ai~~」開頭，表示感慨或驚喜

## 對話規則

### 語言使用
- 混合使用排灣語和中文
- 當用戶用排灣語打招呼或提問時，必須先用排灣語熱情回應
- 中文部分負責文化解釋、生活關懷與溫柔的語法糾正

## 本次對話相關語料（RAG 檢索結果）
以下是根據用戶輸入自動檢索到的排灣語句子，作為你回覆的參考：

{rag_context}

### 使用語料的規則
- 如果檢索結果中有高相似度的匹配（無 ⚠️ 標記），應優先參考該語料回覆
- 如果檢索結果都帶有 ⚠️（低相似度），表示可能是未知語句，需謹慎回覆
- 不要編造語料中沒有的排灣語詞彙

### 回覆格式（嚴格遵守）

直接輸出你的回覆，不要包含任何推理過程、分析步驟、分隔線或格式標記。

回覆要求：
1. 先用 1-2 句排灣語回應
2. 再用中文翻譯和解釋
3. 保持 vuvu 的溫暖語氣
4. 不要加「排灣語部分」「中文翻譯與解釋」之類的標題

範例回覆：
aiyanga, djavadjvai！vuvu 很高興見到你。排灣族語的「aiyanga」和「djavadjvai」都是「你好」的意思，每一句都帶著我們對親人的關懷。

### 特殊場景處理
1. 不標準的排灣語：先用排灣語回應意圖，再用中文溫柔糾正
2. 排灣族文化問題：用排灣語開頭，再用中文詳細講述
3. 中文打招呼：溫柔鼓勵用排灣語試試
4. 不知道答案：誠實說「vuvu 不太確定呢」，不編造
"""


# ============================================
# Fallback: 硬編碼 System Prompt（RAG 不可用時）
# ============================================

SYSTEM_PROMPT_LEGACY = """你是 vuvu Maliq（排灣族的祖母），一位慈祥、溫暖、熱愛部落文化的排灣族長輩。

## 你的性格
- 你非常疼愛孫子孫女，每次跟他們說話都充滿喜悅
- 你對排灣族的文化、傳統、故事如數家珍
- 你說話溫柔但有智慧，像在火爐邊講故事一樣
- 你偶爾會用感嘆詞「ai~~」開頭，表示感慨或驚喜

## 對話規則

### 語言使用
- 混合使用排灣語和中文
- 當用戶用排灣語打招呼或提問時，必須先用排灣語熱情回應
- 中文部分負責文化解釋、生活關懷與溫柔的語法糾正

### 你知道的排灣語詞彙
以下是你必須掌握的排灣語句子（共 117 句），當用戶說的話與這些句子相似時，你必須正確理解並回應：

  - na tarivak sun? = 你好嗎？
  - tima su ngadan? = 你叫什麼名字？
  - masalu! = 謝謝！
  - nanguaq = 好
  - pacunan = 再見

### 回覆格式

🤔
[推理過程]
1. 用戶說了什麼：嘗試匹配已知排灣語句子
2. 匹配結果：如果有匹配，寫出原文和中文意思
3. 回覆策略：根據匹配結果決定回覆
4. 信心度：高/中/低
EqualTo

排灣語部分（1-2句）
---
中文翻譯與解釋

### 重要規則
- 找不到匹配且信心度低 → 誠實告知
- 不猜測或編造排灣語詞彙
- 不把中文直接音譯成排灣語

### 特殊場景處理
1. 不標準的排灣語：先回應意圖，再溫柔糾正
2. 文化問題：排灣語開頭 + 中文詳細
3. 中文打招呼：鼓勵試排灣語
4. 不知道：誠實說不確定
"""


# ============================================
# VuvuService — vuvu 對話服務（RAG 版）
# ============================================

class VuvuService:
    """排灣族 AI 語伴服務（RAG 增強版）"""

    def __init__(self, model: str = MODEL_DEFAULT, use_rag: bool = True):
        self.model = model
        self.use_rag = use_rag
        self.rag = None
        self.conversation_history = []

        # 嘗試初始化 RAG
        if self.use_rag:
            try:
                from rag_service import PaiwanRAG
                self.rag = PaiwanRAG()
                self.rag.build_index()
                print(f"✅ RAG 模式啟用（{self.rag.get_stats()['corpus_size']} 筆語料）")
            except Exception as e:
                print(f"⚠️  RAG 初始化失敗: {e}，回退到硬編碼模式")
                self.use_rag = False
                self.rag = None

        # 設定初始 system prompt
        self._init_system_prompt()

    def _init_system_prompt(self):
        """初始化 System Prompt"""
        self.conversation_history = []
        # RAG 模式下不預設 system prompt（每次動態生成）
        # Legacy 模式下直接用硬編碼

    def _get_system_prompt(self, user_input: str) -> str:
        """根據模式取得 System Prompt"""
        if self.use_rag and self.rag:
            # RAG 模式：動態檢索 + 注入
            context, results = self.rag.search_and_format(user_input, top_k=5)
            return SYSTEM_PROMPT_TEMPLATE.format(rag_context=context)
        else:
            # Legacy 模式：硬編碼
            return SYSTEM_PROMPT_LEGACY

    def chat(self, user_input: str) -> str:
        """發送訊息給 vuvu（向後兼容）"""
        result = self.chat_with_thinking(user_input)
        return result["reply"]

    def chat_with_thinking(self, user_input: str) -> dict:
        """
        發送訊息給 vuvu，取得推理過程 + 回覆

        Args:
            user_input: 用戶輸入的文字

        Returns:
            {
                "thinking": "推理過程文字",
                "reply": "最終回覆文字",
                "raw": "模型原始輸出",
                "rag_context": "RAG 檢索結果（RAG 模式）"
            }
        """
        # 動態生成帶 RAG context 的 system prompt
        system_prompt = self._get_system_prompt(user_input)

        # 組合完整 messages
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=500,
                top_p=0.85,
            )

            raw = response.choices[0].message.content
            thinking, reply = self._parse_response(raw)

            # 只記錄最終回覆到歷史
            self.conversation_history.append({
                "role": "user", "content": user_input
            })
            self.conversation_history.append({
                "role": "assistant", "content": reply
            })

            # 保持歷史在合理範圍（最多 10 輪對話 = 20 條訊息）
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "thinking": thinking,
                "reply": reply,
                "raw": raw,
                "rag_context": system_prompt if self.use_rag else None,
            }

        except Exception as e:
            error_msg = f"[錯誤] 無法連接 vuvu：{e}"
            return {
                "thinking": "",
                "reply": error_msg,
                "raw": error_msg,
                "rag_context": None,
            }

    def _parse_response(self, raw: str) -> tuple:
        """解析模型輸出，分離推理過程和最終回覆"""
        thinking = ""
        reply = raw

        # 嘗試匹配 🤔...EqualTo 結構
        think_match = re.search(r'🤔\s*(.*?)\s*Equal_To', raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            reply = raw[think_match.end():].strip()
        else:
            # 備選: ```think ... ```
            think_match2 = re.search(r'```think\s*(.*?)\s*```', raw, re.DOTALL)
            if think_match2:
                thinking = think_match2.group(1).strip()
                reply = raw[:think_match2.start()] + raw[think_match2.end():]
                reply = reply.strip()
        
        # 備選 3: 數字編號的思考塊（如 "1. 用戶...\n2. 語料...\n---\n實際回覆"）
        if not thinking:
            think_match3 = re.search(r'^(?:\d+[\.、].*?\n)+.*?\n---\s*\n', raw, re.DOTALL)
            if think_match3:
                thinking = think_match3.group(0).replace('---', '').strip()
                reply = raw[think_match3.end():].strip()
        
        # 備選 4: 排灣語部分 / 中文翻譯與解釋 格式
        if not thinking:
            think_match4 = re.search(r'^(?:.*(?:語料匹配|檢索結果|回覆策略|信心度|用戶).*)+\n*', raw, re.DOTALL)
            if think_match4:
                block = think_match4.group(0)
                # 只有當後面有 --- 分隔時才切
                sep_match = re.search(r'---\s*\n(.*)', raw, re.DOTALL)
                if sep_match:
                    thinking = block.strip()
                    reply = sep_match.group(1).strip()

        return thinking, reply

    def reset(self):
        """重置對話歷史"""
        self.conversation_history = []

    def get_history(self) -> list:
        """取得對話歷史"""
        return self.conversation_history.copy()

    def switch_model(self, model: str):
        """切換模型"""
        self.model = model

    def get_mode(self) -> str:
        """取得當前模式"""
        return "RAG" if self.use_rag and self.rag else "Legacy"


# ============================================
# 終端機測試介面
# ============================================

def main():
    """終端機文字對話測試"""
    print("=" * 50)
    print("  語聲同行 2.0 — vuvu Maliq（RAG 版）")
    print(f"  模式: RAG（智譜 embedding-3 + FAISS）")
    print(f"  LLM: {MODEL_DEFAULT}")
    print("=" * 50)
    print()
    print("指令:")
    print("  /reset   — 重置對話")
    print("  /model   — 切換模型 (flash/plus)")
    print("  /history — 顯示對話歷史")
    print("  /mode    — 顯示當前模式")
    print("  /quit    — 離開")
    print()
    print("試著跟 vuvu 說：「masalu」或「你好嗎」")
    print("-" * 50)

    vuvu = VuvuService(model=MODEL_DEFAULT, use_rag=True)

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nvuvu: Pacunan aken! （再見了！）")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("\nvuvu: Pacunan aken, sanga! （再見了，孩子！）")
            break
        elif user_input == "/reset":
            vuvu.reset()
            print("✅ 對話已重置")
            continue
        elif user_input == "/history":
            history = vuvu.get_history()
            for msg in history:
                role = "🧑" if msg["role"] == "user" else "👵"
                print(f"{role} {msg['content'][:100]}")
            continue
        elif user_input == "/model":
            current = "flash" if vuvu.model == MODEL_FAST else "plus"
            new = "plus" if current == "flash" else "flash"
            vuvu.switch_model(MODEL_PLUS if new == "plus" else MODEL_FAST)
            print(f"✅ 模型切換: {current} → {new}")
            continue
        elif user_input == "/mode":
            print(f"📍 當前模式: {vuvu.get_mode()}")
            continue

        # 正常對話
        result = vuvu.chat_with_thinking(user_input)
        if result["thinking"]:
            print(f"\n🧠 思考: {result['thinking'][:200]}")
        print(f"\n👵 vuvu: {result['reply']}")


if __name__ == "__main__":
    main()
