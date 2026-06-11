"""
benchmark/run_rag_benchmark.py — RAG 增強翻譯評測

用 RAG 檢索相關語料，注入 prompt，測試翻譯品質提升
"""

import json
import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_bleu, compute_wer, compute_exact_match, evaluate_results

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

# 載入全部語料作為 RAG 知識庫
def load_knowledge_base():
    """載入所有語料"""
    kb = []
    
    # 1. paiwan_phrases.txt
    phrases_path = os.path.expanduser('~/Desktop/paiwan_competition_2026/paiwan_phrases.txt')
    with open(phrases_path) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            parts = line.split('=', 1)
            if len(parts) == 2:
                kb.append({
                    'paiwan': parts[0].strip(),
                    'chinese': parts[1].strip(),
                })
    
    # 2. corpus
    corpus_path = os.path.expanduser('~/Desktop/paiwan_competition_2026/modules/paiwan_corpus.json')
    with open(corpus_path) as f:
        corpus = json.load(f)
    for cat_key, cat_data in corpus['categories'].items():
        for sent in cat_data.get('sentences', []):
            kb.append({
                'paiwan': sent['paiwan'],
                'chinese': sent['chinese'],
                'grammar_note': sent.get('grammar_note', ''),
            })
    
    # 3. CSV
    import csv
    csv_path = os.path.expanduser('~/Downloads/20260424族語彙整（固定格式） - 正式版本.csv')
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                trans = row['transcription'].strip()
                meaning = row['meaning'].strip()
                if trans and meaning:
                    kb.append({'paiwan': trans, 'chinese': meaning})
    except Exception:
        pass
    
    # 去重
    seen = set()
    unique = []
    for item in kb:
        key = (item['paiwan'], item['chinese'])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    return unique


def simple_retrieve(query: str, kb: list, top_k: int = 5) -> list:
    """
    簡單關鍵詞檢索（不用 embedding，快速 baseline）
    用於驗證 RAG 的效果增益
    """
    scored = []
    query_lower = query.lower()
    query_chars = set(query_lower)
    
    for item in kb:
        score = 0
        paiwan = item['paiwan'].lower()
        chinese = item['chinese']
        
        # 完全匹配
        if query_lower == paiwan or query_lower == chinese:
            score += 100
        
        # 排灣語包含查詢
        if query_lower in paiwan:
            score += 50
        
        # 查詢包含排灣語
        if paiwan in query_lower:
            score += 30
        
        # 字符重疊
        paiwan_chars = set(paiwan)
        overlap = len(query_chars & paiwan_chars)
        if overlap > 0:
            score += overlap * 2
        
        # 中文匹配
        for c in chinese:
            if c in query:
                score += 1
        
        if score > 0:
            scored.append((score, item))
    
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:top_k]]


def translate_with_rag(query: str, direction: str, kb: list, model: str = "glm-4-flash", top_k: int = 5) -> tuple:
    """
    RAG 增強翻譯
    
    Returns: (translation, retrieved_context)
    """
    if direction == "p2c":
        retrieved = simple_retrieve(query, kb, top_k)
    else:
        retrieved = simple_retrieve(query, kb, top_k)
    
    if not retrieved:
        # 沒有檢索到任何結果，退回零樣本
        if direction == "p2c":
            prompt = f"請將以下排灣族語翻譯成中文。只輸出翻譯結果。\n\n排灣語：{query}\n中文："
        else:
            prompt = f"請將以下中文翻譯成排灣族語。只輸出翻譯結果。\n\n中文：{query}\n排灣語："
    else:
        # 構建 RAG 上下文
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


def run_rag_benchmark(test_set_path: str, model: str = "glm-4-flash", direction: str = "p2c", top_k: int = 5):
    """跑 RAG 增強版 benchmark"""
    
    with open(test_set_path) as f:
        test_set = json.load(f)
    
    kb = load_knowledge_base()
    print(f"Knowledge base: {len(kb)} entries")
    
    results = []
    for i, item in enumerate(test_set):
        paiwan = item['paiwan']
        chinese = item['chinese']
        
        hypothesis, retrieved = translate_with_rag(
            query=paiwan if direction == "p2c" else chinese,
            direction=direction,
            kb=kb,
            model=model,
            top_k=top_k
        )
        
        reference = chinese if direction == "p2c" else paiwan
        
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
            'top_retrieved': retrieved[0] if retrieved else None,
        }
        results.append(result)
        
        status = "✅" if exact else "⚠️" if bleu > 0 else "❌"
        print(f"[{i+1}/{len(test_set)}] {status} {paiwan[:20]}... → {hypothesis[:30]}... (BLEU={bleu:.4f}, WER={wer:.2f})")
        
        time.sleep(0.3)
    
    summary = evaluate_results(results)
    
    return {
        'model': model,
        'direction': direction,
        'rag_top_k': top_k,
        'kb_size': len(kb),
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
    
    print(f"=== RAG Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Top-K: {args.top_k}")
    print()
    
    output = run_rag_benchmark(args.test_set, args.model, args.direction, args.top_k)
    
    model_safe = args.model.replace('/', '_').replace('.', '_')
    out_path = f"benchmark/results/{model_safe}_{args.direction}_rag_top{args.top_k}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))
    print(f"Results saved to: {out_path}")
