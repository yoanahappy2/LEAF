"""
benchmark/build_vector_index.py — 建立向量索引

使用智譜 embedding-3 模型將語料向量化，存為 FAISS 索引
"""

import json
import os
import time
import numpy as np

from openai import OpenAI

ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')
if not ZHIPU_API_KEY:
    env_path = os.path.expanduser('~/Desktop/paiwan_competition_2026/.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('ZHIPUAI_API_KEY'):
                    ZHIPU_API_KEY = line.split('=', 1)[1].strip()

client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


def get_embeddings_batch(texts: list, batch_size: int = 20) -> list:
    """批量獲取 embedding"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="embedding-3",
                input=batch,
            )
            # 按 index 排序確保順序正確
            sorted_data = sorted(response.data, key=lambda x: x.index)
            for item in sorted_data:
                all_embeddings.append(item.embedding)
            
            print(f"  Embedded {i+1}-{min(i+batch_size, len(texts))} / {len(texts)}")
            time.sleep(0.5)  # 避免 rate limit
            
        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            # 用零向量填充
            for _ in batch:
                all_embeddings.append([0.0] * 2048)
    
    return all_embeddings


def build_index(corpus_path: str, output_dir: str):
    """建立向量索引"""
    
    with open(corpus_path) as f:
        data = json.load(f)
    
    entries = data['entries']
    print(f"Building index for {len(entries)} entries...")
    
    # 構建文本：排灣語 + 中文
    texts = [f"{e['paiwan']} {e['chinese']}" for e in entries]
    
    # 獲取 embeddings
    print("Getting embeddings...")
    embeddings = get_embeddings_batch(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # 歸一化（用於 cosine similarity）
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_np = embeddings_np / norms
    
    # 建構 FAISS 索引
    import faiss
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity (after normalization)
    index.add(embeddings_np)
    
    # 儲存
    os.makedirs(output_dir, exist_ok=True)
    
    # FAISS 索引
    faiss.write_index(index, os.path.join(output_dir, 'expanded_faiss.index'))
    
    # Embeddings numpy
    np.save(os.path.join(output_dir, 'expanded_embeddings.npy'), embeddings_np)
    
    # 索引元數據
    metadata = {
        'total': len(entries),
        'dimension': dimension,
        'model': 'embedding-3',
        'created': '2026-04-25',
        'source': corpus_path,
    }
    with open(os.path.join(output_dir, 'expanded_index_meta.json'), 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Index saved to {output_dir}/")
    print(f"   FAISS index: expanded_faiss.index")
    print(f"   Embeddings:  expanded_embeddings.npy ({embeddings_np.shape})")
    print(f"   Metadata:    expanded_index_meta.json")
    
    return index, embeddings_np


if __name__ == '__main__':
    corpus_path = '/Users/sbb-mei/Desktop/paiwan_competition_2026/data/expanded_corpus.json'
    output_dir = '/Users/sbb-mei/Desktop/paiwan_competition_2026/data/'
    
    build_index(corpus_path, output_dir)
