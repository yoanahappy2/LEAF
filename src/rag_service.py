"""
rag_service.py
語聲同行 2.0 — 排灣語 RAG 檢索服務

將 117 句排灣語語料轉為向量索引，實現語意檢索。
Embedding: 智譜 embedding-3（OpenAI 兼容格式）
向量庫: FAISS (IndexFlatL2)

作者: Backend_Dev
日期: 2026-04-15
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ============================================
# 配置
# ============================================

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
EMBEDDING_MODEL = "embedding-3"
EMBEDDING_DIM = 2048  # 智譜 embedding-3 維度
TOP_K = 5             # 預設檢索數量

# 路徑
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CORPUS_FILE = BASE_DIR / "paiwan_phrases.txt"
INTENT_FILE = BASE_DIR / "api" / "data" / "intent_training.jsonl"

# 索引檔案
INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "corpus_metadata.json"
EMBEDDINGS_FILE = DATA_DIR / "paiwan_embeddings.npy"

# 合併版索引（優先使用）
MERGED_INDEX = DATA_DIR / "merged_faiss.index"
MERGED_CORPUS = DATA_DIR / "merged_corpus.json"

# ============================================
# Embedding 客戶端
# ============================================

embed_client = OpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


def get_embedding(text: str) -> list:
    """取得單筆文字的 embedding 向量"""
    response = embed_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list, batch_size: int = 16) -> list:
    """批次取得 embedding（智譜 API 限制每次最多 16 筆）"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embed_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {len(all_embeddings)}/{len(texts)} sentences")
    return all_embeddings


# ============================================
# 語料解析
# ============================================

def parse_phrases() -> list:
    """從 paiwan_phrases.txt 解析語料"""
    corpus = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 格式: "排灣語 = 中文翻譯"
            if "=" in line:
                parts = line.split("=", 1)
                paiwan = parts[0].strip()
                chinese = parts[1].strip()
                corpus.append({
                    "id": line_num - 1,
                    "paiwan": paiwan,
                    "chinese": chinese,
                    "category": _classify_intent(paiwan, chinese),
                    "source": "paiwan_phrases.txt",
                })
    return corpus


def load_intent_data() -> dict:
    """載入 intent_training.jsonl 的 intent 資訊"""
    intent_map = {}
    if not INTENT_FILE.exists():
        return intent_map
    with open(INTENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get("normalized_text", data.get("text", ""))
            if text:
                intent_map[text] = {
                    "intent": data.get("intent", "unknown"),
                    "negation_type": data.get("negation_type"),
                }
    return intent_map


def _classify_intent(paiwan: str, chinese: str) -> str:
    """簡易意圖分類（基於關鍵詞）"""
    p, c = paiwan.lower(), chinese
    if any(w in p for w in ["tarivak", "masalu", "nanguaq", "pacunan"]):
        return "greeting"
    if "tima" in p:
        return "identity_query"
    if any(w in p for w in ["mapida", "pida"]):
        return "quantity_query"
    if any(w in p for w in ["anema", "ainu", "inuan", "inuan"]):
        return "query"
    if "ini" in p:
        return "negation"
    if "maya" in p:
        return "prohibition"
    if any(w in p for w in ["tjengelay", "padjalim", "rutjemalupung"]):
        return "question"
    if any(w in c for w in ["故事", "傳說", "部落"]):
        return "culture"
    return "general"


# ============================================
# RAG 服務核心
# ============================================

class PaiwanRAG:
    """排灣語 RAG 檢索服務"""

    def __init__(self):
        self.index = None
        self.metadata = []
        self._loaded = False

    # ── 建構索引 ──

    def build_index(self, force_rebuild: bool = False):
        """
        建構或載入 FAISS 索引
        
        Args:
            force_rebuild: 是否強制重建（忽略快取）
        """
        DATA_DIR.mkdir(exist_ok=True)

        # 嘗試載入快取
        if not force_rebuild and self._load_cached():
            print(f"✅ 載入快取索引：{len(self.metadata)} 筆語料")
            self._loaded = True
            return

        # 解析語料
        print("📖 解析語料...")
        corpus = parse_phrases()
        intent_map = load_intent_data()

        # 合併 intent 資訊
        for item in corpus:
            intent_info = intent_map.get(item["paiwan"])
            if intent_info:
                item["intent"] = intent_info["intent"]
                item["negation_type"] = intent_info.get("negation_type")
            else:
                item["intent"] = item.get("category", "general")

        # 生成 embedding（排灣語 + 中文 拼接）
        print(f"🔢 生成 Embedding（{len(corpus)} 筆）...")
        texts = [f"{item['paiwan']} | {item['chinese']}" for item in corpus]
        embeddings = get_embeddings_batch(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # 建構 FAISS 索引
        print("🏗️  建構 FAISS 索引...")
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embeddings_np)
        self.metadata = corpus

        # 儲存快取
        self._save_cache(embeddings_np)
        self._loaded = True
        print(f"✅ 索引建構完成：{len(corpus)} 筆語料，維度 {EMBEDDING_DIM}")

    # ── 檢索 ──

    def search(self, query: str, top_k: int = TOP_K) -> list:
        """
        混合檢索：關鍵詞匹配 + 向量語意
        
        Args:
            query: 用戶輸入（排灣語或中文）
            top_k: 返回前 K 筆結果
            
        Returns:
            [{"paiwan": ..., "chinese": ..., "score": ..., "intent": ...}, ...]
        """
        if not self._loaded:
            self.build_index()

        seen_indices = set()
        results = []

        # 1. 關鍵詞精確匹配（優先級最高）
        query_lower = query.lower().strip()
        for i, item in enumerate(self.metadata):
            p = item.get("paiwan", "").lower()
            c = item.get("chinese", "").lower()
            if query_lower == p or query_lower == c:
                r = item.copy()
                r["distance"] = 0.0
                r["similarity"] = 1.0
                r["match_type"] = "exact"
                results.append(r)
                seen_indices.add(i)

        # 2. 關鍵詞包含匹配
        for i, item in enumerate(self.metadata):
            if i in seen_indices:
                continue
            p = item.get("paiwan", "")
            c = item.get("chinese", "")
            if query_lower in p.lower() or query_lower in c.lower():
                r = item.copy()
                r["distance"] = 0.1
                r["similarity"] = 0.9
                r["match_type"] = "keyword"
                results.append(r)
                seen_indices.add(i)
            if len(results) >= top_k * 2:
                break

        # 3. 向量語意搜尋（補充）
        query_embedding = self._get_query_embedding(query)
        if query_embedding is not None and len(results) < top_k:
            distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.metadata)))
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata) or idx in seen_indices:
                    continue
                item = self.metadata[idx].copy()
                item["distance"] = float(dist)
                item["similarity"] = max(0, 1 - dist / 10)
                item["match_type"] = "vector"
                results.append(item)
                seen_indices.add(idx)

        # 排序：exact > keyword > vector，然後按 similarity 降序
        priority = {"exact": 0, "keyword": 1, "vector": 2}
        results.sort(key=lambda x: (priority.get(x.get("match_type", "vector"), 2), -x.get("similarity", 0)))

        return results[:top_k]

    def format_context(self, results: list) -> str:
        """
        將檢索結果格式化為 LLM 可用的 context 字串
        """
        if not results:
            return "（未找到相關語料）"

        lines = []
        for i, r in enumerate(results, 1):
            sim = r.get("similarity", 0)
            line = f"{i}. 排灣語: {r['paiwan']} ｜中文: {r['chinese']}"
            if r.get("intent"):
                line += f" ｜意圖: {r['intent']}"
            if sim < 0.3:
                line += " ⚠️（低相似度）"
            lines.append(line)

        return "\n".join(lines)

    def search_and_format(self, query: str, top_k: int = TOP_K) -> tuple:
        """
        搜尋 + 格式化（一次搞定）
        
        Returns:
            (formatted_context, raw_results)
        """
        results = self.search(query, top_k)
        context = self.format_context(results)
        return context, results

    # ── 快取管理 ──

    def _load_cached(self) -> bool:
        """嘗試載入快取的索引（優先本地 384 維，zero API cost）"""
        # 最高優先：本地 RAG 索引（384 維，不花 token）
        local_rag_index = DATA_DIR / "rag_local.index"
        local_rag_meta = DATA_DIR / "rag_local_metadata.json"
        if local_rag_index.exists() and local_rag_meta.exists():
            try:
                self.index = faiss.read_index(str(local_rag_index))
                with open(local_rag_meta, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"✅ 載入本地 RAG 索引：{len(self.metadata)} 筆語料（384 維，zero API cost）")
                return True
            except Exception as e:
                print(f"⚠️ 本地 RAG 索引載入失敗: {e}")

        # Fallback: 合併版（2048 維，需智譜 API）
        if MERGED_INDEX.exists() and MERGED_CORPUS.exists():
            try:
                self.index = faiss.read_index(str(MERGED_INDEX))
                with open(MERGED_CORPUS, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self.metadata = raw.get("entries", raw) if isinstance(raw, dict) else raw
                print(f"✅ 載入合併索引：{len(self.metadata)} 筆語料（2048 維）")
                return True
            except Exception as e:
                print(f"⚠️ 合併索引載入失敗: {e}")

        # 最後 fallback: 舊版
        if not all(f.exists() for f in [INDEX_FILE, METADATA_FILE, EMBEDDINGS_FILE]):
            return False

        try:
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        except Exception as e:
            print(f"⚠️  快取載入失敗: {e}")
            return False

    def _get_query_embedding(self, query: str):
        """取得 query 的 embedding（優先本地，fallback 智譜 API）"""
        # 方案 1：本地 embedding
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_local_model') or self._local_model is None:
                self._local_model = SentenceTransformer('all-MiniLM-L6-v2')
            vec = self._local_model.encode(query, normalize_embeddings=True)
            if self.index is not None and self.index.d == vec.shape[0]:
                return np.array([vec], dtype=np.float32)
        except Exception as e:
            print(f"⚠️ 本地 embedding 失敗: {e}")

        # 方案 2：智譜 API（fallback）
        try:
            embedding = get_embedding(query)
            return np.array([embedding], dtype=np.float32)
        except Exception as e:
            print(f"⚠️ 智譜 embedding 也失敗: {e}")
            return None

    def _save_cache(self, embeddings_np: np.ndarray):
        """儲存索引和 metadata 到快取"""
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        np.save(str(EMBEDDINGS_FILE), embeddings_np)

    # ── 工具 ──

    def get_stats(self) -> dict:
        """取得索引統計"""
        return {
            "corpus_size": len(self.metadata),
            "embedding_dim": EMBEDDING_DIM,
            "model": EMBEDDING_MODEL,
            "index_type": "IndexFlatL2",
            "loaded": self._loaded,
        }


# ============================================
# 終端機測試
# ============================================

def main():
    """測試 RAG 檢索"""
    print("=" * 50)
    print("  語聲同行 — RAG 檢索測試")
    print("=" * 50)

    rag = PaiwanRAG()
    rag.build_index()

    print(f"\n📊 索引統計: {rag.get_stats()}")

    test_queries = [
        "masalu",           # 排灣語：謝謝
        "你好嗎",            # 中文
        "你叫什麼名字",       # 中文
        "pida anga su cavilj?",  # 排灣語：你幾歲
        "天氣",              # 無關查詢
    ]

    for q in test_queries:
        print(f"\n{'─' * 40}")
        print(f"🔍 查詢: {q}")
        context, results = rag.search_and_format(q, top_k=3)
        print(f"📄 結果:\n{context}")

    print(f"\n{'=' * 50}")
    print("測試完成")


if __name__ == "__main__":
    main()
