#!/usr/bin/env python3
"""
build_local_rag_index.py — 用本地 embedding (all-MiniLM-L6-v2) 重建 PaiwanRAG 索引

解決問題：智譜 embedding-3 API 額度用罄 (429)，PaiwanRAG.search() 完全不能用
方案：用本地 384 維模型重建索引，之後 zero API cost

作者: 地陪
日期: 2026-05-27
"""

import sys
import json
import numpy as np
import faiss
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def build():
    print("=" * 50)
    print("  🔨 重建本地 RAG 索引 (all-MiniLM-L6-v2, 384 維)")
    print("=" * 50)

    # 1. 載入 sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers 未安裝")
        print("   執行: pip3 install sentence-transformers")
        sys.exit(1)

    print("  🔄 載入 all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  ✅ 模型載入完成，維度: {model.get_sentence_embedding_dimension()}")

    # 2. 載入語料庫
    merged = DATA_DIR / "merged_corpus.json"
    if not merged.exists():
        # 嘗試其他位置
        for name in ["corpus_merged.json", "expanded_corpus.json"]:
            alt = DATA_DIR / name
            if alt.exists():
                merged = alt
                break

    if not merged.exists():
        print(f"❌ 找不到語料庫檔案")
        sys.exit(1)

    with open(merged, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'entries' in raw:
        corpus = raw['entries']
    elif isinstance(raw, list):
        corpus = raw
    else:
        print(f"❌ 未知的語料格式: {type(raw)}")
        sys.exit(1)

    print(f"  📖 語料庫: {len(corpus)} 筆")

    # 3. 生成 embedding
    texts = []
    for item in corpus:
        paiwan = item.get("paiwan", item.get("排灣語", ""))
        chinese = item.get("chinese", item.get("中文", ""))
        texts.append(f"{paiwan} | {chinese}")

    print(f"  🔢 生成 embedding ({len(texts)} 筆)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"  ✅ Embedding 完成: {embeddings_np.shape}")

    # 4. 建 FAISS 索引
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    print(f"  🏗️  FAISS 索引: {index.ntotal} 筆, {dim} 維")

    # 5. 儲存
    index_path = DATA_DIR / "rag_local.index"
    faiss.write_index(index, str(index_path))
    print(f"  💾 索引已儲存: {index_path}")

    # 儲存 metadata（確保格式統一）
    meta_path = DATA_DIR / "rag_local_metadata.json"
    normalized_corpus = []
    for item in corpus:
        normalized_corpus.append({
            "paiwan": item.get("paiwan", item.get("排灣語", "")),
            "chinese": item.get("chinese", item.get("中文", "")),
            "intent": item.get("intent", item.get("category", "general")),
            "category": item.get("category", ""),
        })
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_corpus, f, ensure_ascii=False, indent=2)
    print(f"  💾 Metadata: {meta_path}")

    # 6. 驗證
    print("\n  🧪 驗證搜尋...")
    test_query = "你好"
    qvec = model.encode([test_query], normalize_embeddings=True)
    D, I = index.search(np.array(qvec, dtype=np.float32), 5)
    for i, (d, idx) in enumerate(zip(D[0], I[0])):
        item = normalized_corpus[idx]
        print(f"    {i+1}. [{d:.3f}] {item['paiwan']} = {item['chinese']}")

    print("\n  ✅ 本地 RAG 索引重建完成！")
    print(f"  維度: {dim} (本地, zero API cost)")
    print(f"  語料: {len(normalized_corpus)} 筆")
    return True

if __name__ == "__main__":
    build()
