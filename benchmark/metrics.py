"""
benchmark/metrics.py — 評測指標計算

支持：BLEU、WER（詞錯誤率）、翻譯準確率
"""

import math
from collections import Counter


def tokenize(text: str) -> list:
    """簡單分詞：中文按字符，其他按空格"""
    text = text.strip()
    # 檢測是否主要是中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:
        # 中文：按字符分割，去掉標點
        return [c for c in text if c.strip() and c not in '，。！？、：；""''（）']
    else:
        # 非中文：按空格分割
        return text.split()


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    計算 BLEU 分數（0-1）
    reference: 參考翻譯
    hypothesis: 模型輸出
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if not hyp_tokens:
        return 0.0
    
    # brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    # n-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)
        )
        
        matches = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total = max(sum(hyp_ngrams.values()), 1)
        
        precisions.append(matches / total if matches > 0 else 0)
    
    # 如果有任何 n-gram precision 為 0，BLEU = 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    計算詞錯誤率 WER（0-1，越低越好）
    使用最小編輯距離
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    
    # Levenshtein distance
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m] / n


def compute_exact_match(reference: str, hypothesis: str) -> bool:
    """完全匹配"""
    return reference.strip() == hypothesis.strip()


def evaluate_results(results: list) -> dict:
    """
    批量計算評測結果
    results: [{reference, hypothesis, ...}, ...]
    """
    bleu_scores = []
    wer_scores = []
    exact_matches = 0
    
    for r in results:
        ref = r.get('reference', r.get('chinese', ''))
        hyp = r.get('hypothesis', '')
        
        bleu = compute_bleu(ref, hyp)
        wer = compute_wer(ref, hyp)
        exact = compute_exact_match(ref, hyp)
        
        bleu_scores.append(bleu)
        wer_scores.append(wer)
        if exact:
            exact_matches += 1
        
        r['bleu'] = round(bleu, 4)
        r['wer'] = round(wer, 4)
        r['exact_match'] = exact
    
    n = len(results)
    return {
        'total': n,
        'avg_bleu': round(sum(bleu_scores) / n, 4) if n else 0,
        'avg_wer': round(sum(wer_scores) / n, 4) if n else 0,
        'exact_match_rate': round(exact_matches / n, 4) if n else 0,
    }


if __name__ == '__main__':
    # 測試
    print("BLEU test:")
    print(f"  完全匹配: {compute_bleu('你好嗎', '你好嗎')}")
    print(f"  部分匹配: {compute_bleu('你好嗎', '你好')}")
    print(f"  完全不匹配: {compute_bleu('你好嗎', '天氣很好')}")
    
    print("\nWER test:")
    print(f"  完全正確: {compute_wer('你好嗎', '你好嗎')}")
    print(f"  一個錯: {compute_wer('你好嗎', '你好')}")
