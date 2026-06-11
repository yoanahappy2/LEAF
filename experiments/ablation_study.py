#!/usr/bin/env python3
"""
ablation_study.py — 四層品質保證框架消融實驗

實驗設計（7 組）：
- Full（4 層全開）: exact → RAG + grammar prompt → roundtrip verify
- NoVerify（3 層）: exact → RAG + grammar prompt（關閉往返驗證）
- NoGrammar（2 層）: exact → RAG only（關閉語法規則注入 + 往返驗證）
- NoRAG（1 層）: exact only + LLM zero-shot（無 RAG、無語法、無驗證）
- ExactOnly（0 層）: 只用精確匹配（完全不用 LLM）
- RAGOnly: exact → RAG + LLM（無語法規則、無驗證）
- ZeroShot: LLM 直接翻（無 exact、無 RAG、無語法、無驗證）

測試集：50 條中文短詞/短句 → 排灣語
評估：klokah 交叉驗證（精確匹配率）

作者: 地陪
日期: 2026-05-28
"""

import json
import os
import re
import sys
import time
import random
import traceback
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
client = OpenAI(api_key=ZHIPUAI_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")
MODEL = "glm-4-flash"  # 省 token

# ============================================
# 測試集
# ============================================

# 手工挑選的 50 條測試集（涵蓋日常用詞、動詞、名詞、短句）
TEST_SET = [
    # 日常用語
    "你好", "謝謝", "再見", "對不起", "請",
    "是", "不是", "有", "沒有", "在",
    # 人稱
    "我", "你", "他", "我們", "你們",
    # 家人
    "爸爸", "媽媽", "爺爺", "奶奶", "朋友",
    # 自然
    "水", "山", "海", "太陽", "月亮",
    "雨", "風", "火", "石頭", "樹",
    # 動物
    "狗", "豬", "雞", "魚", "鳥",
    # 身體
    "眼睛", "嘴巴", "手", "腳", "頭",
    # 動作
    "吃", "喝", "睡覺", "去", "來",
    # 顏色/形容
    "紅色", "白色", "黑色", "大", "小",
]

KLOKAH_URL = "https://web.klokah.tw/php/multiSearchResult.php"

# ============================================
# Prompt 模板
# ============================================

# Full prompt（含語法規則）— 從 translate_service.py 精簡版
GRAMMAR_PROMPT = """你是一個專業的**東排灣語**-中文翻譯助手。

## 任務
根據提供的參考語料，將輸入的中文翻譯為東排灣語。

## 參考語料（RAG 檢索結果）
{rag_context}

## 常用詞彙表
{vocab_table}

## ⚠️ 排灣語硬性語法規範（必須遵守）

### 1. 語序：VSO（動詞-主詞-賓語）
- 中文「我吃飯」→ 排灣語「吃 我 飯」

### 2. 格位標記
- **tua**：一般名詞賓語
- **tjai / tjanu**：人稱賓語
- **ti**：專有名詞
- **a**：不定名詞標記

### 3. 代名詞
- **aken** = 我（主格）、**sun** = 你、**timadju** = 他
- **itjen** = 我們、**mun** = 你們、**tiamadju** = 他們

### 4. 焦點系統
- -en（受事焦點）、-an（處所焦點）、em-/um-（施事焦點）
- 例：keman（吃，施事）、kakanen（食物/被吃的，受事）

### 5. 疑問詞
- **anema** = 什麼、**tima** = 誰、**pida** = 多少、**inua** / **inu** = 哪裡

### 6. 常見句型
- 存在句：izua（有/存在）
- 否定句：ini（不/沒有）
- 祈使句：動詞原形 + 對象

## Few-shot 範例
1. 我想念家人 → sengelit aken tua taqumaqanan
2. 請問你叫什麼名字 → anema su ngadan?
3. 這是什麼 → anema a izuwa?
4. 我喝水 → kasiw aken
5. 他在睡覺 → qemav timadju
6. 謝謝你 → masalu
7. 我們一起走 → tara itjen
8. 你要去哪裡 → inu a su tazaywan?
9. 她很漂亮 → bulay aravac aya

## 規則
1. 只輸出翻譯結果，不要解釋
2. 如果參考語料有完全匹配，直接使用
3. 不要自創詞彙
"""

# 無語法規則版
NO_GRAMMAR_PROMPT = """你是一個東排灣語-中文翻譯助手。

## 參考語料
{rag_context}

## 規則
1. 只輸出翻譯結果，不要解釋
2. 根據參考語料翻譯
3. 不要自創詞彙
"""

# Zero-shot
ZEROSHOT_PROMPT = """你是一個東排灣語-中文翻譯助手。請將中文翻譯為東排灣語。
只輸出翻譯結果，不要解釋。不要自創詞彙。"""


# ============================================
# 工具函數
# ============================================

def query_klokah(chinese_word, dialect=23):
    """查詢 klokah API"""
    url = f"{KLOKAH_URL}?d={dialect}&txt={urllib.parse.quote(chinese_word)}&f=fuzzy"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception:
        return []
    
    results = []
    try:
        root = ET.fromstring(xml_text)
        for section in root:
            for item in section.findall('.//item'):
                t = item.find('text')
                c = item.find('chinese')
                if t is not None and c is not None:
                    results.append({
                        "paiwan": (t.text or "").strip(),
                        "chinese": (c.text or "").strip(),
                    })
    except ET.ParseError:
        pass
    return results


def klokah_verify(chinese, generated_paiwan):
    """用 klokah 驗證翻譯正確性"""
    results = query_klokah(chinese)
    if not results:
        # 嘗試短詞
        keyword = chinese[:min(3, len(chinese))]
        if keyword != chinese:
            results = query_klokah(keyword)
    
    if not results:
        return "not_found", results
    
    klokah_paiwans = [r["paiwan"] for r in results]
    
    if any(generated_paiwan == kp for kp in klokah_paiwans):
        return "correct", results
    elif any(generated_paiwan in kp or kp in generated_paiwan for kp in klokah_paiwans):
        return "partial", results
    else:
        return "wrong", results


def rag_search(chinese, corpus, index, embedder, top_k=8):
    """RAG 向量檢索"""
    import numpy as np
    query_vec = embedder.encode([chinese], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(corpus):
            entry = corpus[idx]
            results.append(f"排灣語: {entry['paiwan']} ｜中文: {entry['chinese']}")
    return "\n".join(results) if results else "（未找到相關語料）"


def keyword_search(chinese, corpus, top_k=5):
    """簡易關鍵詞搜索"""
    results = []
    for e in corpus:
        if chinese in e.get("chinese", "") or e.get("chinese", "") in chinese:
            results.append(f"排灣語: {e['paiwan']} ｜中文: {e['chinese']}")
            if len(results) >= top_k:
                break
    return "\n".join(results) if results else "（未找到相關語料）"


def exact_match(chinese, corpus):
    """精確匹配"""
    for e in corpus:
        if e.get("chinese", "").strip() == chinese.strip():
            return e["paiwan"]
    return None


def call_llm(system_prompt, user_msg, temperature=0.2):
    """呼叫 LLM"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        # 清理解釋
        if '\n' in text:
            text = text.split('\n')[0].strip()
        return text
    except Exception as e:
        return f"[ERROR: {e}]"


def roundtrip_verify(original_chinese, generated_paiwan, threshold=0.35):
    """往返驗證（簡化版，用 LLM 翻回中文比對）"""
    if not generated_paiwan or generated_paiwan.startswith("["):
        return False, 0.0, ""
    
    # 翻回中文
    back = call_llm(
        "你是東排灣語-中文翻譯助手。只輸出中文翻譯，不要解釋。",
        f"將以下排灣語翻譯為中文：{generated_paiwan}",
        temperature=0.1,
    )
    
    if not back:
        return False, 0.0, ""
    
    # 簡單字元重疊比對
    orig_chars = set(original_chinese)
    back_chars = set(back)
    if orig_chars and back_chars:
        overlap = len(orig_chars & back_chars) / max(len(orig_chars | back_chars), 1)
    else:
        overlap = 0.0
    
    return overlap >= threshold, overlap, back


# ============================================
# 實驗函數
# ============================================

def run_experiment(name, test_items, corpus, index, embedder, config):
    """
    跑一組實驗
    config: {
        use_exact: bool,     # Layer 1: 精確匹配
        use_rag: bool,       # Layer 2: RAG 檢索
        use_grammar: bool,   # Layer 3: 語法規則
        use_verify: bool,    # Layer 4: 往返驗證
    }
    """
    results = []
    correct = 0
    partial = 0
    wrong = 0
    not_found = 0
    exact_hit = 0
    llm_hit = 0
    
    for i, chinese in enumerate(test_items):
        translation = None
        method = "none"
        
        # Layer 1: 精確匹配
        if config.get("use_exact"):
            em = exact_match(chinese, corpus)
            if em:
                translation = em
                method = "exact"
                exact_hit += 1
        
        # Layer 2+3: RAG + LLM
        if translation is None and config.get("use_rag"):
            rag_ctx = rag_search(chinese, corpus, index, embedder, top_k=8)
            
            if config.get("use_grammar"):
                prompt = GRAMMAR_PROMPT.format(rag_context=rag_ctx, vocab_table="（見參考語料）")
            else:
                prompt = NO_GRAMMAR_PROMPT.format(rag_context=rag_ctx, vocab_table="（見參考語料）")
            
            translation = call_llm(prompt, f"請將以下中文翻譯為排灣語：{chinese}")
            method = "rag_llm"
            llm_hit += 1
            
            # Layer 4: 往返驗證
            if config.get("use_verify") and translation and not translation.startswith("["):
                verified, score, back = roundtrip_verify(chinese, translation)
                if not verified:
                    # 重試一次
                    retry = call_llm(prompt, f"請將以下中文翻譯為排灣語：{chinese}", temperature=0.1)
                    if retry and not retry.startswith("["):
                        _, retry_score, _ = roundtrip_verify(chinese, retry)
                        if retry_score is not None and (score is None or retry_score > score):
                            translation = retry
                            method = "rag_llm_verified_retry"
        
        # Zero-shot（不用 RAG 不用 exact）
        if translation is None and not config.get("use_rag") and not config.get("use_exact"):
            translation = call_llm(ZEROSHOT_PROMPT, f"請將以下中文翻譯為排灣語：{chinese}")
            method = "zero_shot"
            llm_hit += 1
        
        if translation is None:
            translation = "[NO_TRANSLATION]"
            method = "none"
        
        # klokah 驗證
        verdict, klokah_results = klokah_verify(chinese, translation)
        
        if verdict == "correct":
            correct += 1
        elif verdict == "partial":
            partial += 1
        elif verdict == "wrong":
            wrong += 1
        else:
            not_found += 1
        
        results.append({
            "chinese": chinese,
            "translation": translation,
            "method": method,
            "verdict": verdict,
            "klokah_top3": [r["paiwan"] for r in klokah_results[:3]],
        })
        
        # 進度
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{len(test_items)} done")
        
        time.sleep(0.3)  # 溫和 klokah 請求
    
    total = len(test_items)
    effective = total - not_found
    accuracy = correct / effective * 100 if effective > 0 else 0
    
    return {
        "name": name,
        "config": config,
        "total": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "not_found": not_found,
        "accuracy": accuracy,
        "exact_hit": exact_hit,
        "llm_hit": llm_hit,
        "details": results,
    }


# ============================================
# 主程式
# ============================================

def main():
    print("=" * 70)
    print("品質保證框架消融實驗")
    print(f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"測試集：{len(TEST_SET)} 條")
    print("=" * 70)
    
    # 載入語料和索引
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    
    print("\n載入語料...")
    with open(DATA_DIR / "merged_corpus.json", "r") as f:
        raw = json.load(f)
        corpus = raw.get("entries", raw) if isinstance(raw, dict) else raw
    print(f"  語料：{len(corpus)} 筆")
    
    # 載入索引
    index = faiss.read_index(str(DATA_DIR / "local_faiss.index"))
    print(f"  FAISS 索引：{index.ntotal} 條，{index.d} 維")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 實驗配置
    experiments = [
        ("Full（4層全開）", {
            "use_exact": True, "use_rag": True, "use_grammar": True, "use_verify": True,
        }),
        ("NoVerify（3層）", {
            "use_exact": True, "use_rag": True, "use_grammar": True, "use_verify": False,
        }),
        ("NoGrammar（2層）", {
            "use_exact": True, "use_rag": True, "use_grammar": False, "use_verify": False,
        }),
        ("RAGOnly", {
            "use_exact": True, "use_rag": True, "use_grammar": False, "use_verify": False,
        }),
        ("ExactOnly", {
            "use_exact": True, "use_rag": False, "use_grammar": False, "use_verify": False,
        }),
        ("ZeroShot", {
            "use_exact": False, "use_rag": False, "use_grammar": False, "use_verify": False,
        }),
    ]
    
    all_results = []
    
    for name, config in experiments:
        print(f"\n{'='*60}")
        print(f"實驗：{name}")
        print(f"配置：{config}")
        print(f"{'='*60}")
        
        result = run_experiment(name, TEST_SET, corpus, index, embedder, config)
        all_results.append(result)
        
        eff = result["total"] - result["not_found"]
        print(f"\n  ✅ 正確：{result['correct']}（{result['correct']/result['total']*100:.1f}%）")
        print(f"  ⚠️ 部分匹配：{result['partial']}（{result['partial']/result['total']*100:.1f}%）")
        print(f"  ❌ 錯誤：{result['wrong']}（{result['wrong']/result['total']*100:.1f}%）")
        print(f"  ❓ 查無：{result['not_found']}（{result['not_found']/result['total']*100:.1f}%）")
        print(f"  📊 有效正確率：{result['correct']}/{eff} = {result['accuracy']:.1f}%")
        print(f"  🔍 精確匹配命中：{result['exact_hit']}，LLM 生成：{result['llm_hit']}")
    
    # ============================================
    # 匯總表
    # ============================================
    print("\n" + "=" * 70)
    print("📊 消融實驗匯總")
    print("=" * 70)
    print(f"| 實驗 | Exact | RAG | Grammar | Verify | 正確率 | 精確命中 | LLM |")
    print(f"|------|-------|-----|---------|--------|--------|---------|-----|")
    for r in all_results:
        c = r["config"]
        print(f"| {r['name']} | {'✅' if c.get('use_exact') else '❌'} | "
              f"{'✅' if c.get('use_rag') else '❌'} | "
              f"{'✅' if c.get('use_grammar') else '❌'} | "
              f"{'✅' if c.get('use_verify') else '❌'} | "
              f"{r['accuracy']:.1f}% | {r['exact_hit']} | {r['llm_hit']} |")
    
    # 儲存結果
    output_path = PROJECT_ROOT / "results" / "ablation_study.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_size": len(TEST_SET),
                "model": MODEL,
            },
            "experiments": [{k: v for k, v in r.items() if k != "details"} for r in all_results],
            "details": {r["name"]: r["details"] for r in all_results},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已存：{output_path}")
    
    # 寫入 memory
    mem_dir = Path("/Users/sbb-mei/.openclaw/workspace/memory")
    report_lines = [
        "# 品質保證框架消融實驗結果",
        f"\n> 時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 測試集：{len(TEST_SET)} 條中文短詞/短句 → 排灣語",
        f"> 驗證：klokah API 交叉驗證（d=23 東排灣語）",
        f"> LLM：{MODEL}",
        "",
        "## 匯總表",
        "",
        "| 實驗 | Exact | RAG | Grammar | Verify | 正確率 |",
        "|------|-------|-----|---------|--------|--------|",
    ]
    for r in all_results:
        c = r["config"]
        report_lines.append(
            f"| {r['name']} | {'✅' if c.get('use_exact') else '❌'} | "
            f"{'✅' if c.get('use_rag') else '❌'} | "
            f"{'✅' if c.get('use_grammar') else '❌'} | "
            f"{'✅' if c.get('use_verify') else '❌'} | "
            f"**{r['accuracy']:.1f}%** |"
        )
    
    report_lines.extend([
        "",
        "## 關鍵發現",
        "",
    ])
    
    # 分析各層貢獻
    full_acc = next(r["accuracy"] for r in all_results if "Full" in r["name"])
    no_verify = next(r["accuracy"] for r in all_results if "NoVerify" in r["name"])
    no_grammar = next(r["accuracy"] for r in all_results if "NoGrammar" in r["name"])
    exact_only = next(r["accuracy"] for r in all_results if "ExactOnly" in r["name"])
    zero_shot = next(r["accuracy"] for r in all_results if "ZeroShot" in r["name"])
    
    report_lines.append(f"- **Zero-shot baseline**：{zero_shot:.1f}%（LLM 直接翻，無任何參考）")
    report_lines.append(f"- **Exact-only**：{exact_only:.1f}%（語料庫精確匹配覆蓋率）")
    if no_grammar != no_verify:
        report_lines.append(f"- **Grammar 規則貢獻**：{no_grammar:.1f}% → {no_verify:.1f}%（+{no_verify - no_grammar:.1f}pp）")
    if no_verify != full_acc:
        report_lines.append(f"- **Roundtrip Verify 貢獻**：{no_verify:.1f}% → {full_acc:.1f}%（+{full_acc - no_verify:.1f}pp）")
    report_lines.append(f"- **完整框架**：{full_acc:.1f}%")
    report_lines.append(f"- **各層累積貢獻**：Zero-shot {zero_shot:.1f}% → +RAG → +Grammar → +Verify → {full_acc:.1f}%")
    
    report_lines.extend([
        "",
        "## 數據位置",
        f"- JSON：`results/ablation_study.json`",
        f"",
        "---",
        f"*地陪自動生成*",
    ])
    
    report_path = mem_dir / "2026-05-28-ablation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"📝 報告已存：{report_path}")


if __name__ == "__main__":
    main()
