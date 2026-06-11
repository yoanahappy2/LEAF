"""
benchmark/run_rag_vector_benchmark.py — 向量檢索 RAG 評測

用智譜 embedding-3 + FAISS 做語意檢索，測試翻譯品質
"""

import json
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_bleu, compute_wer, compute_exact_match, evaluate_results

import faiss

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


def vector_search(query: str, index, entries: list, top_k: int = 5) -> list:
    """用向量做語意檢索"""
    # Query embedding
    response = client.embeddings.create(
        model="embedding-3",
        input=[query],
    )
    query_emb = np.array([response.data[0].embedding], dtype=np.float32)
    
    # 歸一化
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm
    
    # 搜索
    scores, indices = index.search(query_emb, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(entries) and idx >= 0:
            entry = entries[idx]
            entry['similarity_score'] = float(scores[0][i])
            results.append(entry)
    
    return results


def translate_with_vector_rag(query: str, direction: str, index, entries: list, 
                               model: str = "glm-4-flash", top_k: int = 5) -> tuple:
    """向量 RAG 增強翻譯"""
    
    # 檢索
    retrieved = vector_search(query, index, entries, top_k)
    
    if not retrieved:
        if direction == "p2c":
            prompt = f"請將以下排灣族語翻譯成中文。只輸出翻譯結果。\n\n排灣語：{query}\n中文："
        else:
            prompt = f"請將以下中文翻譯成排灣族語。只輸出翻譯結果。\n\n中文：{query}\n排灣語："
    else:
        context_lines = []
        for r in retrieved:
            line = f"- {r['paiwan']} = {r['chinese']}"
            if r.get('grammar_note'):
                line += f"（{r['grammar_note']}）"
            context_lines.append(line)
        context = "\n".join(context_lines)
        
        if direction == "p2c":
            prompt = f"""你是排灣族語翻譯專家。以下是排灣語-中文對照的參考語料：

{context}

請根據以上參考語料，將以下排灣族語翻譯成中文。只輸出翻譯結果，不要加解釋。

排灣語：{query}
中文："""
        else:
            prompt = f"""你是排灣族語翻譯專家。以下是排灣語-中文對照的參考語料：

{context}

請根據以上參考語料，將以下中文翻譯成排灣族語。只輸出翻譯結果，不要加解釋。

中文：{query}
排灣語："""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip(), retrieved
    except Exception as e:
        return f"[ERROR: {e}]", retrieved


def run_benchmark(test_set_path: str, model: str = "glm-4-flash", direction: str = "p2c", top_k: int = 5):
    """跑向量 RAG benchmark"""
    
    with open(test_set_path) as f:
        test_set = json.load(f)
    
    # 載入語料和索引
    corpus_path = '/Users/sbb-mei/Desktop/paiwan_competition_2026/data/expanded_corpus.json'
    with open(corpus_path) as f:
        corpus = json.load(f)
    entries = corpus['entries']
    
    index_path = '/Users/sbb-mei/Desktop/paiwan_competition_2026/data/expanded_faiss.index'
    index = faiss.read_index(index_path)
    
    print(f"Corpus: {len(entries)} entries")
    print(f"FAISS index: {index.ntotal} vectors")
    
    results = []
    for i, item in enumerate(test_set):
        query = item['paiwan'] if direction == "p2c" else item['chinese']
        
        hypothesis, retrieved = translate_with_vector_rag(
            query=query, direction=direction,
            index=index, entries=entries,
            model=model, top_k=top_k
        )
        
        reference = item['chinese'] if direction == "p2c" else item['paiwan']
        
        bleu = compute_bleu(reference, hypothesis)
        wer = compute_wer(reference, hypothesis)
        exact = compute_exact_match(reference, hypothesis)
        
        result = {
            **item,
            'reference': reference,
            'hypothesis': hypothesis,
            'bleu': round(bleu, 4),
            'wer': round(wer, 4),
            'exact_match': exact,
            'retrieved_count': len(retrieved),
            'top_similarity': retrieved[0].get('similarity_score', 0) if retrieved else 0,
        }
        results.append(result)
        
        status = "✅" if exact else "⚠️" if bleu > 0 else "❌"
        sim = retrieved[0].get('similarity_score', 0) if retrieved else 0
        print(f"[{i+1}/{len(test_set)}] {status} {item['paiwan'][:20]}... → {hypothesis[:30]}... (BLEU={bleu:.4f}, sim={sim:.3f})")
        
        time.sleep(0.5)  # embedding API rate limit
    
    summary = evaluate_results(results)
    
    return {
        'model': model,
        'direction': direction,
        'rag_method': 'vector_faiss',
        'embedding_model': 'embedding-3',
        'rag_top_k': top_k,
        'kb_size': len(entries),
        'summary': summary,
        'results': results
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glm-4-flash')
    parser.add_argument('--direction', default='p2c')
    parser.add_argument('--test-set', default='benchmark/test_set.json')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    
    print(f"=== Vector RAG Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Method: embedding-3 + FAISS (cosine similarity)")
    print(f"Top-K: {args.top_k}")
    print()
    
    output = run_benchmark(args.test_set, args.model, args.direction, args.top_k)
    
    model_safe = args.model.replace('/', '_').replace('.', '_')
    out_path = f"benchmark/results/{model_safe}_{args.direction}_rag_vector_top{args.top_k}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))
    print(f"Results saved to: {out_path}")
