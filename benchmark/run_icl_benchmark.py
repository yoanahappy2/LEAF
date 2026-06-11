"""
benchmark/run_icl_benchmark.py — ICL (In-Context Learning) 對比實驗

實驗設計：
- 對照組 1: GLM-4-Plus 零樣本翻譯（已有 baseline）
- 對照組 2: GLM-4-Plus + 5-shot ICL
- 對照組 3: GLM-4-Plus + 10-shot ICL  
- 對照組 4: GLM-4-Plus + 20-shot ICL
- 實驗組:   GLM-4-Plus + RAG（已有 benchmark）

目的：用數據證明排灣語需要 RAG，而非只靠大模型 ICL 能力

引用：
- Li & Niehues (2025): "In-context Language Learning for Endangered Languages in Speech Recognition"
- arXiv 2026: "Multimodal In-context Learning for ASR of Low-resource Languages"
"""

import json
import time
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_bleu, compute_wer, compute_exact_match, evaluate_results

from openai import OpenAI

ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '') or os.environ.get('ZHIPUAI_API_KEY', '')
if not ZHIPU_API_KEY:
    env_path = os.path.expanduser('~/Desktop/paiwan_competition_2026/.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('ZHIPUAI_API_KEY') or line.startswith('ZHIPU_API_KEY'):
                    ZHIPU_API_KEY = line.split('=', 1)[1].strip()

client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 語料和測試集路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'data', 'merged_corpus.json')
TEST_SET_PATH = os.path.join(BASE_DIR, 'benchmark', 'test_set.json')


def load_corpus():
    """載入語料庫"""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    return data['entries']


def load_test_set():
    """載入測試集"""
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def get_icl_examples(corpus, test_set, n_examples, seed=42):
    """
    從語料庫中選取 ICL 範例，排除測試集
    
    策略：優先選擇多樣性高的範例（覆蓋不同來源和長度）
    """
    test_pairs = set()
    for item in test_set:
        test_pairs.add((item['paiwan'].strip(), item['chinese'].strip()))
    
    # 過濾掉測試集
    candidates = []
    for entry in corpus:
        p = entry['paiwan'].strip()
        c = entry['chinese'].strip()
        if (p, c) not in test_pairs and p and c and len(p) > 2:
            candidates.append(entry)
    
    print(f"Corpus: {len(corpus)} total, {len(candidates)} after filtering test set")
    
    # 按來源分層抽樣，確保多樣性
    random.seed(seed)
    
    if len(candidates) <= n_examples:
        return random.sample(candidates, len(candidates))
    
    # 分組
    by_source = {}
    for entry in candidates:
        src = entry.get('source', 'unknown')
        by_source.setdefault(src, []).append(entry)
    
    # 按比例從每個來源抽取
    examples = []
    remaining = n_examples
    
    sources = list(by_source.keys())
    random.shuffle(sources)
    
    # 先確保每個來源至少 1 個
    for src in sources:
        if remaining <= 0:
            break
        if by_source[src]:
            examples.append(random.choice(by_source[src]))
            remaining -= 1
    
    # 剩餘名額按比例分配
    if remaining > 0:
        pool = [e for e in candidates if e not in examples]
        random.shuffle(pool)
        examples.extend(pool[:remaining])
    
    return examples[:n_examples]


def build_icl_prompt(examples, paiwan_text):
    """
    構建 ICL prompt
    
    格式：先展示範例，再讓模型翻譯目標句
    """
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": "你是一個排灣族語-中文翻譯專家。請根據以下範例，將排灣族語翻譯成中文。只輸出翻譯結果，不要加任何解釋。"
    })
    
    # ICL 範例
    few_shot_text = "以下是一些排灣語-中文翻譯範例：\n\n"
    for i, ex in enumerate(examples, 1):
        few_shot_text += f"範例{i}：\n排灣語：{ex['paiwan']}\n中文：{ex['chinese']}\n\n"
    
    few_shot_text += f"請翻譯以下句子：\n排灣語：{paiwan_text}\n中文："
    
    messages.append({
        "role": "user",
        "content": few_shot_text
    })
    
    return messages


def translate_icl(paiwan_text, examples, model="glm-4-plus"):
    """ICL 翻譯"""
    messages = build_icl_prompt(examples, paiwan_text)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def run_icl_benchmark(n_shots_list=None, model="glm-4-plus"):
    """
    跑 ICL 對比實驗
    
    Args:
        n_shots_list: 要測試的 shot 數列表，如 [5, 10, 20]
        model: 模型名稱
    """
    if n_shots_list is None:
        n_shots_list = [5, 10, 20]
    
    corpus = load_corpus()
    test_set = load_test_set()
    
    print(f"=== ICL Benchmark ===")
    print(f"Model: {model}")
    print(f"Test set: {len(test_set)} items")
    print(f"Corpus: {len(corpus)} entries")
    print(f"Shots to test: {n_shots_list}")
    print()
    
    all_results = {}
    
    for n_shots in n_shots_list:
        print(f"\n{'='*50}")
        print(f"Running {n_shots}-shot ICL...")
        print(f"{'='*50}")
        
        # 選取 ICL 範例（固定 seed 確保可重現）
        examples = get_icl_examples(corpus, test_set, n_shots, seed=42)
        print(f"Selected {len(examples)} ICL examples")
        
        results = []
        for i, item in enumerate(test_set):
            paiwan = item['paiwan']
            chinese = item['chinese']
            
            hypothesis = translate_icl(paiwan, examples, model)
            reference = chinese
            
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
            }
            results.append(result)
            
            status = "✅" if exact else "❌"
            print(f"[{i+1}/{len(test_set)}] {status} {paiwan[:25]:25s} → {hypothesis[:30]:30s} (BLEU={bleu:.3f})")
            
            time.sleep(0.5)  # 避免 rate limit
        
        summary = evaluate_results(results)
        
        all_results[f"{n_shots}_shot"] = {
            'model': model,
            'method': f"ICL-{n_shots}shot",
            'n_shots': n_shots,
            'summary': summary,
            'results': results
        }
        
        print(f"\n📊 {n_shots}-shot Summary:")
        print(f"   BLEU: {summary['avg_bleu']:.4f}")
        print(f"   WER:  {summary['avg_wer']:.4f}")
        print(f"   Exact Match: {summary['exact_match_rate']:.2%}")
    
    # 儲存所有結果
    out_path = os.path.join(BASE_DIR, 'benchmark', 'results', 'icl_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"📊 ICL vs Baseline vs RAG 總對比")
    print(f"{'='*60}")
    
    # 讀取已有 baseline 結果做對比
    print(f"\n| Method          | BLEU    | WER    | Exact Match |")
    print(f"|-----------------|---------|--------|-------------|")
    
    # Baseline (from existing results)
    baseline_paths = [
        ('GLM-4-Flash Zero-shot', f'glm-4-flash_p2c_baseline.json'),
        ('GLM-4-Plus Zero-shot', f'glm-4-plus_p2c_baseline.json'),
    ]
    
    results_dir = os.path.join(BASE_DIR, 'benchmark', 'results')
    for label, fname in baseline_paths:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            s = data.get('summary', data)
            bleu = s.get('avg_bleu', 0)
            wer = s.get('avg_wer', 0)
            em = s.get('exact_match_rate', 0)
            print(f"| {label:16s} | {bleu:.4f} | {wer:.4f} | {em:.2%}      |")
    
    # ICL results
    for key in sorted(all_results.keys()):
        r = all_results[key]
        s = r['summary']
        label = f"ICL {r['n_shots']}-shot"
        print(f"| {label:16s} | {s['avg_bleu']:.4f} | {s['avg_wer']:.4f} | {s['exact_match_rate']:.2%}      |")
    
    # RAG results
    rag_paths = [
        ('RAG Keyword(242)', 'glm-4-flash_p2c_rag_top5.json'),
        ('RAG Vector(242)', 'glm-4-flash_p2c_rag_vector_top5.json'),
    ]
    for label, fname in rag_paths:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            s = data.get('summary', data)
            bleu = s.get('avg_bleu', 0)
            wer = s.get('avg_wer', 0)
            em = s.get('exact_match_rate', 0)
            print(f"| {label:16s} | {bleu:.4f} | {wer:.4f} | {em:.2%}      |")
    
    print(f"\nResults saved to: {out_path}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glm-4-plus', help='模型名稱')
    parser.add_argument('--shots', nargs='+', type=int, default=[5, 10, 20], help='ICL shot 數')
    args = parser.parse_args()
    
    run_icl_benchmark(n_shots_list=args.shots, model=args.model)
