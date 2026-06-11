"""
translate_service.py
語聲同行 2.0 — 排灣語↔中文 雙向翻譯服務

基於 RAG 檢索 + LLM 生成，支援：
- 排灣語 → 中文
- 中文 → 排灣語
- 混合 RAG（關鍵詞 + 向量）

作者: 地陪
日期: 2026-04-26
"""

import os
import re
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPUAI_API_KEY_EMBEDDING = os.getenv("ZHIPUAI_API_KEY_EMBEDDING") or ZHIPUAI_API_KEY

client = OpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

MODEL_FAST = "glm-4.5-air"  # 2026-05-30: 切到 4.5-air（flash 資源包已用罄）
EMBEDDING_MODEL = "embedding-3"
EMBEDDING_DIM = 2048

BASE_DIR = Path(__file__).parent

# ============================================
# 翻譯 System Prompt
# ============================================

TRANSLATE_SYSTEM_PROMPT = """你是一個專業的**東排灣語**-中文翻譯助手。

## 任務
根據提供的參考語料，將輸入翻譯為目標語言。

## 參考語料（RAG 檢索結果）
{rag_context}

## 常用詞彙表（翻譯時優先查此表）
{vocab_table}

## ⚠️ 排灣語硬性語法規範（必須遵守）

### 1. 語序：VSO（動詞-主詞-賓語）
排灣語的基本語序是 VSO，不是中文的 SVO。
- 中文「我 吃 飯」→ 排灣語「吃 我 飯」
- 中文「他 想 家人」→ 排灣語「想 他 家人」

### 2. 格位標記（Case Markers）
賓語前面必須加格位標記：
- **tua**：一般名詞賓語（事物、概念）
  例：sengelit aken tua taqumaqanan（我想念家人）
- **tjai / tjanu**：人名或人稱賓語
  例：sengelit aken tjai vuvu（我想念祖母）
- **ti**：專有名詞前（人名）
  例：ti savan a ku ngadan（我的名字是 savan）
- **a**：不定名詞標記（在數量詞或描述後）
  例：izua a drangi（有一個朋友）

### 3. 代名詞綁定
人稱代名詞是獨立詞，緊隨動詞（不與動詞融合）：
- **aken** = 我（主格）
- **sun** = 你（主格）
- **timadju / tiamadju** = 他/她
- **itjen / itjenan** = 我們（含/不含聽者）
- **mun** = 你們
- **tiamadju** = 他們

代名詞位置：動詞 + 代名詞 + 賓語
例：qemavilj aken tua kakanen（我買食物）

### 4. 所有格標記
- **ku** = 我的（ku ngadan = 我的名字）
- **su** = 你的（su ngadan = 你的名字）
- **nia** = 他/她的
- **ta / ita** = 我們的
- **u** = 所有格標記（放在名詞前）
  例：tua u taqumaqanan = 的家人

### 5. 疑問句
疑問詞放在句首或動詞前：
- **inuan** = 哪裡
- **anema** = 什麼
- **pida** = 多少
- **tima** = 誰
- **nungida** = 什麼時候
- **namakuda** = 如何

## 📘 語法規則（摘自《排灣語語法概論》張秀絹著，原住民族委員會出版）

### 焦點系統（極度重要！）
排灣語用焦點系統標記句子的核心角色，不是用詞序：
- 主事焦點 <em>：凸顯「做動作的人」→ qemauqaung ca 'edrian.（這小孩在哭）
- 受事焦點 -in/-en：凸顯「被動作影響的對象」→ vinuljuq ta qaciljay a qezung.（窗戶被石頭丟了）
- 處所焦點 -an：凸顯「動作發生的地點」→ cavuan ta ciqav a cekui.（桌子是用來包魚的地方）
- 受惠/工具焦點 si-：凸顯「受益者/工具」→ sivuljuq ta qezung a qaciljay.（石頭被用來丟窗戶）

### 時貌語氣
- 完成貌：<in> 或 na- → na'emananga ta vaqu a 'edrian.（小孩已經吃了小米）
- 進行貌：動詞部分重疊 → 'ema'ana'en ta vurati.（我正在吃地瓜）
- 未來貌：uri → uri vai'esun a semataihuku?（你要去台北嗎？）
- 否定：inia（不）、ne'a（沒有）、maya（別）

### 存在句
izua + 名詞組 → izua gimeng imaza.（這裡有學校）
否定：ne'a → ne'a nu gimeng imaza.（這裡沒有學校）

### 祈使句
- 肯定：動詞原型 → vai'u!（你走！）、ekelju!（跑！）
- 否定：maya + 動詞 → maya qemaung!（別哭！）
- 規勸：-av → tja'anav ca vasa!（咱們吃芋頭吧！）

### 連動結構
V1 sa V2（先做V1然後做V2）
'ema'an ta vutjulj sa cavucavu ta ciqav.（吃了肉然後包魚）

## 組合翻譯範例（Few-shot）

範例 1：中文「我很想念我的家人」→ 排灣語
拆解：想念(V) + 我(S) + 家人(O)
→ sengelit（想念）+ aken（我）+ tua（格位）+ u（所有格）+ taqumaqanan（家人）
→ **sengelit aken tua u taqumaqanan.**

範例 2：中文「他今天去山上工作」→ 排灣語
拆解：工作(V) + 他(S) + 今天 + 山上
→ masengeseng（工作）+ tiamadju（他）+ nusauni（今天）+ i gadu（在山上）
→ **masengeseng tiamadju nusauni i gadu.**

範例 3：中文「你叫什麼名字」→ 排灣語
拆解：名字 + 你 + 什麼
→ **tima su ngadan?**

範例 4：中文「我想喝水」→ 排灣語
拆解：想喝(V) + 我(S) + 水(O)
→ tjengelay aken a kium tu zaljum.

範例 5：中文「我在吃地瓜」→ 排灣語
拆解：吃(V-進行貌) + 我(S) + 地瓜(O)
→ 'ema'ana'en ta vurati.

範例 6：中文「別哭」→ 排灣語
→ maya qemaung!

範例 7：中文「你的衣服漂亮」→ 排灣語
→ bulabulay a su 'itung.

範例 8：中文「他用石頭丟窗戶」→ 排灣語
拆解：丟(V-受事焦點) + 石頭 + 窗戶
→ vinuljuq ta qaciljay a qezung niamadju nimadju.

範例 9：中文「你想去台北嗎」→ 排灣語
→ uri vaik sun a semataihuku?

## 翻譯規則（優先級從高到低）
1. 參考語料中有完全匹配 → 直接使用該翻譯
2. **近似語料替換**：如果找到結構相似但部分詞不同的語料，保留語料中的排灣語結構，只替換不同的詞。例如：語料「媽媽說山上有很多蚊子 → aya ti kina liaw cacalag i vavuwa」，如果你要翻譯「媽媽說山上有很多動物」，就把 cacalag（蚊子）替換為正確的詞（如 qemuziquzip 動物），保留其餘結構
3. 有部分匹配 → 根據語料中的詞彙，嚴格按照上述 VSO 語法和格位標記規則組合
4. 組合時必須包含格位標記（tua/tjai/tjanu），不得省略
5. 不要編造語料中沒有的排灣語詞彙
6. 如果無法確定翻譯，在前面加 [不確定]

## 輸出格式（嚴格遵守）
只輸出翻譯結果本身，不要任何解釋或多餘文字。

範例：
輸入「masalu」→ 輸出「謝謝！」
輸入「你好嗎」→ 輸出「na tarivak sun?」
輸入「我很想念我的家人」→ 輸出「sengelit aken tua u taqumaqanan."
"""


# ============================================
# 翻譯服務
# ============================================

class PaiwanTranslator:
    """排灣語↔中文 雙向翻譯服務"""

    def __init__(self):
        self.index = None
        self.corpus = []
        self.dictionary = None  # 精確詞典
        self.keyword_map_paiwan = {}  # 排灣語關鍵詞索引
        self.keyword_map_chinese = {}  # 中文關鍵詞索引
        self._loaded = False

    def load(self):
        """載入擴充語料和 FAISS 索引"""
        if self._loaded:
            return

        # 優先載入合併語料（含 klokah 東排灣）
        merged_file = BASE_DIR / "data" / "merged_corpus.json"
        fallback_file = BASE_DIR / "data" / "expanded_corpus.json"

        if merged_file.exists():
            corpus_file = merged_file
        elif fallback_file.exists():
            corpus_file = fallback_file
        else:
            raise FileNotFoundError("語料檔案不存在")

        with open(corpus_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict) and "entries" in raw:
                self.corpus = raw["entries"]
            elif isinstance(raw, list):
                self.corpus = raw
            else:
                raise ValueError(f"未知的語料格式: {type(raw)}")

        # ============================================
        # 語料大清洗（P0 修復）
        # ============================================
        
        # 1. 基本過濾
        self.corpus = [e for e in self.corpus if e.get('chinese', '').strip() 
                       and e.get('paiwan', '').strip()
                       and e['chinese'].strip().lower() != 'undefined'
                       and e['paiwan'].strip().lower() != 'undefined']
        
        # 2. 移除 B:/A: 對話前綴
        for e in self.corpus:
            c = e.get('chinese', '').strip()
            c = re.sub(r'^[A-Za-z]:\s*', '', c)
            c = c.rstrip('。！？：；,. ')
            e['chinese'] = c
        
        # 3. 過濾掉空中文（清洗後可能變空）
        self.corpus = [e for e in self.corpus if e.get('chinese', '').strip()]
        
        # 4. 排灣語單字元黑名單（語法虛詞，不應作為翻譯結果）
        PAIWAN_CHAR_BLACKLIST = {'a', 'u', 'i', 'na', 'ta', 'ku', 'su', 'ti', 'nu', 'sa', 'ka', 'la', 'ma', 'pa', 'qa', 'se', 'n', 'm', 'k', 'l', 'p', 's', 't'}
        
        # 5. 過濾 custom 來源中的垃圾數據
        before_count = len(self.corpus)
        cleaned = []
        for e in self.corpus:
            source = e.get('source', '')
            paiwan = e.get('paiwan', '').strip()
            chinese = e.get('chinese', '').strip()
            
            # custom 來源：嚴格過濾
            if source == 'custom':
                # 排灣語太短（單字元或常見虛詞）
                if paiwan.lower().rstrip('?.!') in PAIWAN_CHAR_BLACKLIST:
                    continue
                # 排灣語是語法說明（含中文括號）
                if any(marker in paiwan for marker in ['(', '（', '標記', '虛詞', '助詞']):
                    continue
                # 中文是語法說明
                if any(marker in chinese for marker in ['標記', '虛詞', '助詞', '焦點', '連繫', '斜格', '主格']):
                    continue
            
            # 通用過濾：亂碼檢測
            if any(bad in chinese for bad in ['å', '©', 'º', 'ç']):
                continue
            
            cleaned.append(e)
        
        self.corpus = cleaned
        removed = before_count - len(self.corpus)
        print(f"  語料清洗：移除 {removed} 筆（{before_count} → {len(self.corpus)}）")

        # 載入核心詞彙表（最高優先級，直接作為精確匹配的首選）
        self._core_vocab = {}  # {paiwan_lower: chinese}
        self._core_vocab_reverse = {}  # {chinese: paiwan} 反向映射
        core_vocab_file = BASE_DIR / "data" / "core_vocab.tsv"
        if core_vocab_file.exists():
            with open(core_vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        p, c = parts[0].strip(), parts[1].strip()
                        if p and c:
                            self._core_vocab[p.lower()] = c
                            # 反向映射：如果中文還沒有對應，或新詞更標準，就覆蓋
                            self._core_vocab_reverse[c] = p
            if self._core_vocab:
                print(f"  核心詞彙表：{len(self._core_vocab)} 筆（含 {len(self._core_vocab_reverse)} 反向映射）")

        # 載入精確詞典（短詞優先匹配）
        dict_file = BASE_DIR / "data" / "paiwan_dictionary.json"
        if dict_file.exists():
            with open(dict_file, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
        else:
            self.dictionary = None

        # 載入 FAISS 向量索引（優先本地 384維 → fallback 智譜 2048維）
        local_index = BASE_DIR / "data" / "local_faiss.index"
        merged_index = BASE_DIR / "data" / "merged_faiss.index"
        fallback_index = BASE_DIR / "data" / "expanded_faiss.index"

        if local_index.exists():
            self.index = faiss.read_index(str(local_index))
            print(f"  向量索引：本地 all-MiniLM-L6-v2（{self.index.d} 維）")
        elif merged_index.exists():
            self.index = faiss.read_index(str(merged_index))
            print(f"  向量索引：智譜 embedding-3（{self.index.d} 維）")
        elif fallback_index.exists():
            self.index = faiss.read_index(str(fallback_index))
            print(f"  向量索引：智譜 embedding-3 舊版（{self.index.d} 維）")
        else:
            print("⚠️ 向量索引不存在，只使用關鍵詞檢索")
            self.index = None

        # 建立關鍵詞索引
        self._build_keyword_maps()

        # 來源可信度權重（用於 exact match 排序）
        self._source_priority = {
            'klokah_d23': 10,     # 官方語料，最可信
            'conversation': 9,
            'dialogue': 9,
            'speech': 8,
            'readingtext': 8,
            'culture': 8,
            'essay': 7,
            'read': 7,
            'phrases': 7,
            'picture': 7,
            'song': 6,
            'vocabulary': 6,
            'nine': 6,
            'alphabet': 6,
            'ebook情境式族語教材': 8,
            'ebook情境式族語教材_對話': 8,
            '語法概論例句': 7,
            'core_vocab_fix': 9,
            'csv': 6,
            'custom': 3,          # 用戶自建，最低優先
            'corpus': 5,
        }
        
        # 為每個 corpus 條目添加優先級
        for e in self.corpus:
            e['_priority'] = self._source_priority.get(e.get('source', ''), 5)
        
        self._loaded = True
        print(f"✅ 翻譯服務載入完成：{len(self.corpus)} 筆語料（已清洗+加權）")

    def _build_keyword_maps(self):
        """建立雙向關鍵詞索引 + 精確匹配索引（含來源優先級）"""
        self.keyword_map_paiwan = {}
        self.keyword_map_chinese = {}
        # 精確匹配索引：記錄 text + priority + source
        self._exact_paiwan = {}
        self._exact_chinese = {}

        for item in self.corpus:
            paiwan = item.get("paiwan", "").strip()
            chinese = item.get("chinese", "").strip()
            priority = item.get('_priority', 5)
            source = item.get('source', '')

            # 精確匹配索引 — 排灣語→中文
            p_key = paiwan.lower().rstrip("?.!:")
            if p_key:
                if p_key not in self._exact_paiwan:
                    self._exact_paiwan[p_key] = []
                for c_part in re.split(r'[；;/]', chinese):
                    c_part = c_part.strip(' 。！？：，,')
                    if c_part:
                        existing = [x for x in self._exact_paiwan[p_key] if x['text'] == c_part]
                        if not existing:
                            self._exact_paiwan[p_key].append({'text': c_part, 'priority': priority, 'source': source})
                        elif existing[0]['priority'] < priority:
                            existing[0]['priority'] = priority
                            existing[0]['source'] = source

            # 精確匹配索引 — 中文→排灣語
            if chinese:
                c_key = chinese.rstrip("?.！，,、：；")
                if c_key:
                    if c_key not in self._exact_chinese:
                        self._exact_chinese[c_key] = []
                    if paiwan:
                        existing = [x for x in self._exact_chinese[c_key] if x['text'] == paiwan]
                        if not existing:
                            self._exact_chinese[c_key].append({'text': paiwan, 'priority': priority, 'source': source})
                        elif existing[0]['priority'] < priority:
                            existing[0]['priority'] = priority
                            existing[0]['source'] = source

            # 關鍵詞索引（用於 RAG 模糊檢索）
            if paiwan:
                for word in paiwan.lower().split():
                    word = word.strip("?.!,")
                    if word and len(word) > 1:
                        if word not in self.keyword_map_paiwan:
                            self.keyword_map_paiwan[word] = []
                        self.keyword_map_paiwan[word].append(item)

            if chinese:
                for i in range(len(chinese)):
                    for length in [2, 3, 4]:
                        if i + length <= len(chinese):
                            phrase = chinese[i:i+length]
                            if phrase not in self.keyword_map_chinese:
                                self.keyword_map_chinese[phrase] = []
                            self.keyword_map_chinese[phrase].append(item)

        print(f"  精確匹配索引：{len(self._exact_paiwan)} 排灣語詞、{len(self._exact_chinese)} 中文詞")

        # 建立變體索引
        self._build_variant_index()

    def _is_paiwan(self, text: str) -> bool:
        """判斷輸入是否為排灣語"""
        # 排灣語特徵：拉丁字母 + 常見排灣語詞綴
        paiwan_markers = [
            "na ", "su ", "aken", "sun?", "mun", "tima", "pida",
            "anema", "inuan", "ini", "maya", "masalu", "tarivak",
            "nanguaq", "pacunan", "tjengelay", "vuvu", "kama",
            "cavilj", "ngadan", "umaq", "zua", "tiyamadju",
            "tjaljaljak", "vavayan", "uqaljay", "siruvetjek",
            "taqumaqanan", "qadupu", "milingan", "kakedrian",
        ]
        text_lower = text.lower().strip()
        # 檢查是否有拉丁字母（非中文）
        has_latin = any(c.isalpha() and ord(c) < 128 for c in text_lower)
        if not has_latin:
            return False
        # 如果包含中文字，不是排灣語
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        if has_chinese:
            return False
        return True

    def _keyword_search(self, query: str, direction: str, top_k: int = 5) -> list:
        """關鍵詞檢索"""
        results = []
        seen_ids = set()

        if direction == "p2c":
            # 排灣語 → 中文：在排灣語關鍵詞中搜索
            query_words = query.lower().split()
            for word in query_words:
                word = word.strip("?.!,")
                matches = self.keyword_map_paiwan.get(word, [])
                for m in matches:
                    mid = id(m)  # 用物件 id 去重
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        results.append(m)
        else:
            # 中文 → 排灣語：在中文關鍵詞中搜索
            for i in range(len(query)):
                for length in [4, 3, 2]:
                    if i + length <= len(query):
                        phrase = query[i:i+length]
                        matches = self.keyword_map_chinese.get(phrase, [])
                        for m in matches:
                            mid = id(m)
                            if mid not in seen_ids:
                                seen_ids.add(mid)
                                results.append(m)

        return results[:top_k]

    # 本地 embedding 模型（延遲載入）
    _local_embedder = None

    @classmethod
    def _get_local_embedder(cls):
        """取得本地 sentence-transformers 模型（延遲載入，只載入一次）"""
        if cls._local_embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("  🔄 載入本地 embedding 模型 all-MiniLM-L6-v2...")
                cls._local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("  ✅ 本地 embedding 模型載入完成")
            except ImportError:
                print("  ⚠️ sentence-transformers 未安裝，本地 embedding 不可用")
                return None
        return cls._local_embedder

    def _vector_search(self, query: str, top_k: int = 5) -> list:
        """向量語意檢索（優先本地 embedding，fallback 智譜 API）"""
        if self.index is None:
            return []

        query_vec = None

        # 方案 1：本地 embedding（免費、離線）
        embedder = self._get_local_embedder()
        if embedder is not None:
            try:
                vec = embedder.encode(query, normalize_embeddings=True)
                # FAISS 索引是 2048 維（智譜），本地模型是 384 維
                # 維度不匹配 → 需要用本地模型重建索引
                if self.index.d == vec.shape[0]:
                    query_vec = np.array([vec], dtype=np.float32)
                else:
                    # 維度不匹配，嘗試用智譜 API
                    pass
            except Exception as e:
                print(f"⚠️ 本地 embedding 失敗: {e}")

        # 方案 2：智譜 API embedding（fallback）
        if query_vec is None:
            try:
                embed_client = OpenAI(
                    api_key=ZHIPUAI_API_KEY_EMBEDDING,
                    base_url="https://open.bigmodel.cn/api/paas/v4"
                )
                response = embed_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=query,
                )
                query_vec = np.array([response.data[0].embedding], dtype=np.float32)
            except Exception as e:
                print(f"⚠️ 智譜 embedding 也失敗: {e}")
                return []

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.corpus):
                item = self.corpus[idx].copy()
                item["distance"] = float(dist)
                results.append(item)

        return results

    def _hybrid_search(self, query: str, direction: str, top_k: int = 5) -> list:
        """混合檢索（關鍵詞 + 向量）"""
        # 關鍵詞結果
        kw_results = self._keyword_search(query, direction, top_k=top_k)

        # 向量結果（多拉幾條再跟關鍵詞合併篩選）
        vec_results = self._vector_search(query, top_k=top_k * 3)

        # 合併去重（關鍵詞優先）
        seen = set()
        merged = []

        for item in kw_results:
            key = f"{item.get('paiwan', '')}|{item.get('chinese', '')}"
            if key not in seen:
                seen.add(key)
                item["source"] = "keyword"
                merged.append(item)

        for item in vec_results:
            key = f"{item.get('paiwan', '')}|{item.get('chinese', '')}"
            if key not in seen:
                seen.add(key)
                item["source"] = "vector"
                merged.append(item)

        return merged[:top_k]

    def _format_rag_context(self, results: list) -> str:
        """格式化 RAG 結果給 LLM"""
        if not results:
            return "（未找到相關語料）"

        lines = []
        for i, r in enumerate(results, 1):
            src = r.get("source", "")
            src_tag = f" [{src}]" if src else ""
            line = f"{i}. 排灣語: {r.get('paiwan', '')} ｜中文: {r.get('chinese', '')}{src_tag}"
            lines.append(line)
        return "\n".join(lines)

    def _roundtrip_verify(self, original_chinese: str, generated_paiwan: str, threshold: float = 0.5) -> tuple:
        """往返驗證：將生成的排灣語翻回中文，跟原始中文比對語義一致性。

        Returns:
            (verified: bool, score: float, back_translation: str)
        """
        paiwan_clean = generated_paiwan.strip()
        if not paiwan_clean or len(paiwan_clean) < 3:
            return False, 0.0, ""

        # Step 1: 將排灣語翻回中文（_quick_p2c 已包含精確匹配 + LLM fallback）
        back_tr = self._quick_p2c(paiwan_clean)

        if not back_tr:
            return False, None, ""

        # Step 2: 計算語義相似度
        score = self._semantic_similarity(original_chinese, back_tr)

        # Step 3: 判定
        verified = score >= threshold
        return verified, score, back_tr

    def _quick_p2c(self, paiwan_text: str) -> str:
        """快速排灣語→中文：優先用精確匹配，找不到就用 LLM"""
        # 清理
        clean = re.sub(r'^[\s?.!,，。！、：；]+|[\s?.!,，。！、：；]+$', '', paiwan_text.strip().lower())
        if not clean:
            return ""

        # 1. 精確匹配
        matches = self._exact_paiwan.get(clean, [])
        if matches:
            first = matches[0]
            # 兼容新舊格式
            if isinstance(first, dict):
                return first.get('text', '')
            return first

        # 2. 如果精確匹配找不到，直接用 LLM 翻譯（句子級別更準確）
        try:
            back_response = client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {"role": "system", "content": "你是排灣語-中文翻譯助手。直接將排灣語翻譯為中文，只輸出翻譯結果，不要加任何解釋。"},
                    {"role": "user", "content": f"翻譯為中文：{paiwan_text}"},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            back_tr = back_response.choices[0].message.content.strip()
            return back_tr
        except Exception:
            return ""

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """計算兩段中文的語義相似度（關鍵詞重疊 + 字元重疊）

        綜合評分：
        1. 關鍵詞重疊率（分詞後的交集）
        2. 字元級 n-gram 重疊
        3. 長度比例懲罰（避免短句跟長句誤匹配）
        """
        if not text1 or not text2:
            return 0.0

        # 清理
        t1 = re.sub(r'[^\u4e00-\u9fff\w]', '', text1)
        t2 = re.sub(r'[^\u4e00-\u9fff\w]', '', text2)

        if not t1 or not t2:
            return 0.0

        # 1. 字元級 unigram 重疊
        set1 = set(t1)
        set2 = set(t2)
        char_jaccard = len(set1 & set2) / max(len(set1 | set2), 1)

        # 2. bigram 重疊
        bigrams1 = set(t1[i:i+2] for i in range(len(t1)-1))
        bigrams2 = set(t2[i:i+2] for i in range(len(t2)-1))
        if bigrams1 and bigrams2:
            bigram_jaccard = len(bigrams1 & bigrams2) / len(bigrams1 | bigrams2)
        else:
            bigram_jaccard = 0.0

        # 3. 關鍵詞匹配（2-4字的滑動窗口）
        kw_hits = 0
        kw_total = 0
        for length in [3, 2, 4]:
            for i in range(len(t1) - length + 1):
                phrase = t1[i:i+length]
                if phrase in t2:
                    kw_hits += 1
                kw_total += 1
        kw_overlap = kw_hits / max(kw_total, 1)

        # 4. 長度比例懲罰
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2), 1)

        # 綜合評分
        score = (
            char_jaccard * 0.2 +
            bigram_jaccard * 0.3 +
            kw_overlap * 0.3 +
            len_ratio * 0.2
        )

        return round(score, 3)

    def _build_vocab_table(self, text: str, direction: str) -> str:
        """根據輸入文字，從詞彙表提取相關詞彙注入 prompt"""
        if not hasattr(self, '_prompt_vocab'):
            vocab_file = BASE_DIR / "data" / "prompt_vocab.json"
            if vocab_file.exists():
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    self._prompt_vocab = json.load(f)
            else:
                self._prompt_vocab = {}

        if direction == "c2p" and self._prompt_vocab:
            # 從中文輸入中提取匹配的詞彙
            matched = []
            for cn, pw in self._prompt_vocab.items():
                if cn in text:
                    matched.append(f"  {cn} → {pw}")
            if matched:
                return "\n".join(matched[:30])  # 最多 30 個詞
        return "（無特別詞彙）"

    def translate(self, text: str, direction: str = "auto", top_k: int = 8) -> dict:
        """
        翻譯入口

        Args:
            text: 要翻譯的文字
            direction: "p2c"(排灣→中), "c2p"(中→排灣), "auto"(自動偵測)
            top_k: RAG 檢索數量

        Returns:
            {
                "input": 原文,
                "direction": 方向,
                "translation": 翻譯結果,
                "rag_context": RAG 檢索結果,
                "rag_results": 原始檢索結果,
                "method": "exact" | "rag_llm"
            }
        """
        if not self._loaded:
            self.load()

        # 自動偵測方向
        if direction == "auto":
            direction = "p2c" if self._is_paiwan(text) else "c2p"

        # 混合 RAG 檢索
        rag_results = self._hybrid_search(text, direction, top_k=top_k)

        # 檢查是否有精確匹配
        exact = self._check_exact_match(text, direction, rag_results)
        if exact:
            return {
                "input": text,
                "direction": direction,
                "translation": exact,
                "rag_context": self._format_rag_context(rag_results),
                "rag_results": rag_results,
                "method": "exact",
            }

        # RAG + LLM 翻譯
        context_str = self._format_rag_context(rag_results)
        vocab_str = self._build_vocab_table(text, direction)
        system_prompt = TRANSLATE_SYSTEM_PROMPT.format(rag_context=context_str, vocab_table=vocab_str)

        # 方向提示
        if direction == "p2c":
            user_msg = f"請將以下排灣語翻譯為中文：{text}"
        else:
            user_msg = f"請將以下中文翻譯為排灣語：{text}"

        try:
            response = client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            translation = response.choices[0].message.content.strip()

            # 清理 LLM 輸出的多餘解釋（只保留第一行或第一句）
            # 如果輸出包含換行，取第一行
            if '\n' in translation:
                first_line = translation.split('\n')[0].strip()
                # 如果第一行太短（可能是標記），取第二行
                if len(first_line) < 5 and len(translation) > 10:
                    lines = [l.strip() for l in translation.split('\n') if l.strip()]
                    for l in lines:
                        if len(l) >= 5:
                            first_line = l
                            break
                translation = first_line
            # 如果輸出包含「這個翻譯」或「翻譯為」等解釋，提取排灣語部分
            explanation_markers = ['這個翻譯', '翻譯為', '這句話', '以下是', '在排灣語中', '根據語法']
            for marker in explanation_markers:
                if marker in translation:
                    # 嘗試提取第一句排灣語
                    parts = re.split(r'[。！？\n]', translation)
                    for p in parts:
                        p = p.strip()
                        if p and re.match(r'^[a-z\']', p) and len(p) > 3:
                            translation = p.rstrip('.')
                            break
                    break

        except Exception as e:
            return {
                "input": text,
                "direction": direction,
                "translation": f"[翻譯失敗: {e}]",
                "rag_context": context_str,
                "rag_results": rag_results,
                "method": "error",
            }

        # ============================================
        # 往返驗證（Round-trip Verification）
        # 僅對 c2p（中文→排灣語）方向執行
        # ============================================
        verified = False
        verification_score = None
        back_translation = None

        if direction == "c2p" and translation and not translation.startswith('['):
            verified, verification_score, back_translation = self._roundtrip_verify(
                original_chinese=text,
                generated_paiwan=translation,
                threshold=0.35,
            )

            if not verified:
                # 驗證不通過 → 嘗試重試一次（帶驗證提示）
                retry_prompt = (
                    f"你剛才把「{text}」翻譯為「{translation}」，"
                    f"但翻回中文後變成「{back_translation}」，跟原文意思不一致。\n"
                    f"請重新翻譯「{text}」，嚴格按照語法規則和參考語料。"
                    f"如果不確定，請輸出 [不確定] 加上你知道的排灣語關鍵詞。"
                )
                try:
                    retry_response = client.chat.completions.create(
                        model=MODEL_FAST,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": translation},
                            {"role": "user", "content": retry_prompt},
                        ],
                        temperature=0.1,  # 重試用更低溫度
                        max_tokens=200,
                    )
                    retry_translation = retry_response.choices[0].message.content.strip()

                    if retry_translation and not retry_translation.startswith('['):
                        retry_verified, retry_score, retry_back = self._roundtrip_verify(
                            original_chinese=text,
                            generated_paiwan=retry_translation,
                            threshold=0.5,
                        )
                        if retry_verified:
                            translation = retry_translation
                            verified = True
                            verification_score = retry_score
                            back_translation = retry_back
                        elif retry_score is not None and (verification_score is None or retry_score > verification_score):
                            # 重試比第一次好，但還是不過關 → 降級
                            translation = retry_translation
                            verification_score = retry_score
                            back_translation = retry_back
                except Exception:
                    pass  # 重試失敗就用原本的翻譯

        return {
            "input": text,
            "direction": direction,
            "translation": translation,
            "rag_context": context_str,
            "rag_results": rag_results,
            "method": "rag_llm",
            "verified": verified,
            "verification_score": verification_score,
            "back_translation": back_translation,
        }

    def _check_exact_match(self, text: str, direction: str, results: list = None) -> str | None:
        """在全語料庫中做精確匹配（O(1) hash 查找）— 按來源優先級排序"""
        text_clean = re.sub(r'^[\s?.!,，。！、：；]+|[\s?.!,，。！、：；]+$', '', text.strip().lower())
        if not text_clean:
            return None

        # 1. 核心詞彙表（最高優先級，雙向）
        if direction == "p2c" and text_clean in self._core_vocab:
            return self._core_vocab[text_clean]
        if direction == "c2p" and text_clean in self._core_vocab_reverse:
            return self._core_vocab_reverse[text_clean]

        # 2. 全語料精確匹配（新的 dict-of-dicts 結構）
        if direction == "p2c":
            matches = self._exact_paiwan.get(text_clean, [])
        else:
            matches = self._exact_chinese.get(text_clean, [])

        if not matches:
            return None

        return self._pick_best_translation(matches)

    def _pick_best_translation(self, translations: list) -> str:
        """從多個翻譯中選最佳的一個
        
        P0 優化：優先按來源可信度排序，再按頻次
        translations 格式：[{'text': str, 'priority': int, 'source': str}, ...]
        """
        if not translations:
            return ""
        
        # 向下兼容：如果還是舊格式（純字串），走舊邏輯
        if isinstance(translations[0], str):
            from collections import Counter
            counter = Counter(t.strip(' 。！？：；,:') for t in translations if t.strip())
            if counter:
                return counter.most_common(1)[0][0]
            return translations[0]
        
        # 新格式：按優先級分組排序
        # Group by text, keeping max priority
        text_best = {}  # {text: max_priority}
        for item in translations:
            t = item['text'].strip(' 。！？：；,:')
            if not t:
                continue
            pri = item.get('priority', 5)
            if t not in text_best or text_best[t] < pri:
                text_best[t] = pri
        
        if not text_best:
            return translations[0].get('text', '')
        
        # Sort by: priority desc, then frequency (count occurrences in original list)
        from collections import Counter
        text_freq = Counter(item['text'].strip(' 。！？：；,:') for item in translations if item.get('text', '').strip())
        
        sorted_items = sorted(text_best.items(), key=lambda x: (-x[1], -text_freq.get(x[0], 1)))
        
        return sorted_items[0][0]

    def _build_variant_index(self):
        """建立變體拼寫索引：如果用戶輸入 sa，也能匹配 saka 的結果"""
        # 收集所有含 'saka' 的 key，建立 'sa' 的 alias
        variant_pairs = {}  # {變體key: [原key1, 原key2]}
        
        for key in list(self._exact_paiwan.keys()):
            # saka → sa 變體
            if 'saka ' in key:
                alt = key.replace('saka ', 'sa ')
                if alt not in self._exact_paiwan:
                    variant_pairs.setdefault(alt, []).append(key)
            # taka → ta 變體  
            if 'taka ' in key:
                alt = key.replace('taka ', 'ta ')
                if alt not in self._exact_paiwan:
                    variant_pairs.setdefault(alt, []).append(key)
        
        # 把變體指向原 key 的結果
        for variant, originals in variant_pairs.items():
            merged = []
            for orig in originals:
                merged.extend(self._exact_paiwan[orig])
            self._exact_paiwan[variant] = merged
        
        if variant_pairs:
            print(f"  變體索引：{len(variant_pairs)} 個拼寫變體")

    def _merge_translations(self, translations: list) -> str:
        """合併多個翻譯：去重、清理、排序後用「/」連接
        
        例如 ['媽媽', '媽', '母親', '阿姨'] → '媽媽 / 母親 / 阿姨'
        例如 ['天', '太陽', '太陽天', '一天'] → '太陽 / 天'
        """
        if not translations:
            return ""
        if len(translations) == 1:
            return translations[0].strip("。！：； ")

        # Step 1: 清理每個翻譯
        cleaned = []
        for t in translations:
            t = t.strip("。！：；,， ").strip()
            if not t or len(t) == 0:
                continue
            cleaned.append(t)

        # Step 2: 精確去重（保留順序）
        seen = set()
        unique = []
        for t in cleaned:
            t_lower = t.lower().strip()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)

        # Step 3: 只做精確去重，不做子串去重
        # （中文子串關係不代表冗餘，「媽」和「媽媽」是不同意義）
        final = unique[:]

        # Step 4: 排序 — 優先核心意義
        # 策略：純漢字且 >=2 字的排最前，然後純數字，然後其他
        def _sort_key(x):
            is_pure_chinese = all('\u4e00' <= c <= '\u9fff' for c in x)
            is_pure_number = all(c.isdigit() for c in x)
            if is_pure_chinese and len(x) >= 2:
                return (0, len(x), x)  # 最優先：2字以上的純中文
            elif is_pure_chinese:
                return (1, len(x), x)  # 其次：1字中文
            elif is_pure_number:
                return (2, len(x), x)  # 再次：純數字
            else:
                return (3, len(x), x)  # 最後：混合

        final.sort(key=_sort_key)

        # Step 5: 限制數量（多義詞最多顯示前 8 個）
        if len(final) > 8:
            final = final[:8]

        return " / ".join(final)

    def batch_translate(self, texts: list, direction: str = "auto") -> list:
        """批量翻譯"""
        results = []
        for i, text in enumerate(texts):
            print(f"  [{i+1}/{len(texts)}] {text[:30]}...")
            result = self.translate(text, direction)
            results.append(result)
        return results


# ============================================
# 終端機測試
# ============================================

def main():
    print("=" * 50)
    print("  語聲同行 2.0 — 雙向翻譯測試")
    print("=" * 50)

    translator = PaiwanTranslator()
    translator.load()

    test_cases = [
        ("masalu", "auto"),
        ("na tarivak sun?", "auto"),
        ("tima su ngadan?", "auto"),
        ("你好嗎", "auto"),
        ("謝謝", "auto"),
        ("你叫什麼名字", "auto"),
        ("你幾歲", "auto"),
        ("pida anga su cavilj?", "auto"),
        ("再見", "auto"),
        ("pacunan", "auto"),
    ]

    for text, direction in test_cases:
        result = translator.translate(text, direction)
        method_tag = "🎯" if result["method"] == "exact" else "🤖"
        dir_tag = "排→中" if result["direction"] == "p2c" else "中→排"
        print(f"\n{method_tag} [{dir_tag}] {result['input']}")
        print(f"   → {result['translation']}")
        print(f"   方法: {result['method']}")

    print("\n" + "=" * 50)
    print("測試完成")


if __name__ == "__main__":
    main()
