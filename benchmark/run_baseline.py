"""
benchmark/run_baseline.py — 跑 baseline 翻譯評測

測試：給模型排灣語，讓它翻譯成中文
記錄 BLEU、WER、完全匹配率
"""

import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_bleu, compute_wer, compute_exact_match, evaluate_results

# 智譜 API 配置
from openai import OpenAI

ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')
if not ZHIPU_API_KEY:
    # 嘗試從 .env 讀取
    env_path = os.path.expanduser('~/Desktop/paiwan_competition_2026/.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('ZHIPU_API_KEY'):
                    ZHIPU_API_KEY = line.split('=', 1)[1].strip()

client = OpenAI(
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4"
)


def translate_paiwan_to_chinese(paiwan_text: str, model: str = "glm-4-flash") -> str:
    """零樣本翻譯：排灣語 → 中文"""
    prompt = f"""請將以下排灣族語翻譯成中文。只輸出翻譯結果，不要加任何解釋。

排灣語：{paiwan_text}
中文："""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def translate_chinese_to_paiwan(chinese_text: str, model: str = "glm-4-flash") -> str:
    """零樣本翻譯：中文 → 排灣語"""
    prompt = f"""請將以下中文翻譯成排灣族語。只輸出翻譯結果，不要加任何解釋。

中文：{chinese_text}
排灣語："""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def run_benchmark(test_set_path: str, model: str, direction: str = "p2c"):
    """
    跑完整 benchmark
    
    Args:
        test_set_path: 測試集 JSON 路徑
        model: 模型名稱
        direction: p2c (排灣→中) 或 c2p (中→排灣)
    """
    with open(test_set_path) as f:
        test_set = json.load(f)
    
    results = []
    for i, item in enumerate(test_set):
        paiwan = item['paiwan']
        chinese = item['chinese']
        
        if direction == "p2c":
            hypothesis = translate_paiwan_to_chinese(paiwan, model)
            reference = chinese
        else:
            hypothesis = translate_chinese_to_paiwan(chinese, model)
            reference = paiwan
        
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
        print(f"[{i+1}/{len(test_set)}] {status} {paiwan[:20]}... → {hypothesis[:30]}... (BLEU={bleu:.2f})")
        
        time.sleep(0.3)  # 避免 rate limit
    
    # 彙總
    summary = evaluate_results(results)
    
    return {
        'model': model,
        'direction': direction,
        'summary': summary,
        'results': results
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glm-4-flash', help='模型名稱')
    parser.add_argument('--direction', default='p2c', help='p2c 或 c2p')
    parser.add_argument('--test-set', default='benchmark/test_set.json')
    args = parser.parse_args()
    
    print(f"=== Baseline Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Direction: {args.direction}")
    print(f"Test set: {args.test_set}")
    print()
    
    output = run_benchmark(args.test_set, args.model, args.direction)
    
    # 儲存結果
    model_safe = args.model.replace('/', '_').replace('.', '_')
    out_path = f"benchmark/results/{model_safe}_{args.direction}_baseline.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Model: {output['summary']}")
    print(f"Results saved to: {out_path}")
