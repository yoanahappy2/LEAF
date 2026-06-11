#!/usr/bin/env python3
"""
bilingual_ablation.py — 雙語言消融實驗（排灣語 + 阿美語）

實驗設計：
- Control: 四層全開（Core Vocab + Exact Match + RAG + Grammar + Verify）
- Exp 1: 關閉 Core Vocab / Exact Match
- Exp 2: 關閉 Grammar 語法注入
- Exp 3: 關閉往返驗證

兩個語言同時測，50 條測試集，klokah 交叉驗證。

作者: 地陪
日期: 2026-05-28
"""

import json, os, sys, time, random, re, traceback
import xml.etree.ElementTree as ET
import urllib.request, urllib.parse
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path("/Users/sbb-mei/Desktop/paiwan_competition_2026")
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

ZHIPUAI_KEY = os.getenv("ZHIPUAI_API_KEY")
client = OpenAI(api_key=ZHIPUAI_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")
MODEL = "glm-4-flash"

# ============================================
# 測試集（共用 50 條中文）
# ============================================
TEST_SET = [
    "你好","謝謝","再見","對不起","請",
    "是","不是","有","沒有","在",
    "我","你","他","我們","你們",
    "爸爸","媽媽","爺爺","奶奶","朋友",
    "水","山","海","太陽","月亮",
    "雨","風","火","石頭","樹",
    "狗","豬","雞","魚","鳥",
    "眼睛","嘴巴","手","腳","頭",
    "吃","喝","睡覺","去","來",
    "紅色","白色","黑色","大","小",
]

KLOKAH_URL = "https://web.klokah.tw/php/multiSearchResult.php"

# ============================================
# klokah 快取
# ============================================
klokah_cache = {}

def cache_klokah(dialect):
    """快取所有測試詞的 klokah 結果"""
    print(f"快取 klokah 結果（d={dialect}）...")
    for ch in TEST_SET:
        url = f"{KLOKAH_URL}?d={dialect}&txt={urllib.parse.quote(ch)}&f=fuzzy"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_text = resp.read().decode("utf-8")
            results = []
            root = ET.fromstring(xml_text)
            for s in root:
                for item in s.findall('.//item'):
                    t = item.find('text')
                    c = item.find('chinese')
                    if t is not None and c is not None:
                        results.append((t.text or "").strip())
            klokah_cache[(dialect, ch)] = results
        except:
            klokah_cache[(dialect, ch)] = []
        
        # 如果找不到，試短詞
        if not klokah_cache[(dialect, ch)] and len(ch) > 3:
            kw = ch[:3]
            url2 = f"{KLOKAH_URL}?d={dialect}&txt={urllib.parse.quote(kw)}&f=fuzzy"
            try:
                req = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    xml_text = resp.read().decode("utf-8")
                results = []
                root = ET.fromstring(xml_text)
                for s in root:
                    for item in s.findall('.//item'):
                        t = item.find('text')
                        c = item.find('chinese')
                        if t is not None and c is not None:
                            results.append((t.text or "").strip())
                klokah_cache[(dialect, ch)] = results
            except:
                pass
        time.sleep(0.3)
    print(f"  快取完成：{len(TEST_SET)} 詞")

def verify_cached(dialect, chinese, translation):
    """用快取的 klokah 結果驗證"""
    kl = klokah_cache.get((dialect, chinese), [])
    if not kl:
        return "not_found"
    if any(translation == kp for kp in kl):
        return "correct"
    if any(translation in kp or kp in translation for kp in kl):
        return "partial"
    return "wrong"

# ============================================
# 翻譯函數
# ============================================

GRAMMAR_PROMPT_TEMPLATE = """你是{lang_name}翻譯助手。

參考語料：
{rag_ctx}

{grammar_section}

只輸出翻譯，不解釋。不要自創詞彙。"""

NO_GRAMMAR_TEMPLATE = "你是{lang_name}翻譯助手。\n參考語料：\n{rag_ctx}\n\n只輸出翻譯，不解釋。"

ZEROSHOT_TEMPLATE = "你是{lang_name}翻譯助手。請將中文翻譯為{lang_name}。只輸出翻譯結果，不要解釋。"

# 排灣語語法規則
PAIWAN_GRAMMAR = """語法規範：
1. VSO 語序（動詞-主詞-賓語）
2. 格位標記：tua（事物）、tjai（人稱）、ti（專名）、a（不定）
3. 代名詞：aken=我, sun=你, timadju=他, itjen=我們, mun=你們
4. 焦點：-en（受事）、-an（處所）、em-/um-（施事）
5. 存在：izua, 否定：ini

範例：
我想念家人 → sengelit aken tua taqumaqanan
你叫什麼名字 → anema su ngadan?
謝謝你 → masalu
我們一起走 → tara itjen"""

def rag_search(chinese, corpus, index, embedder, top_k=8):
    qv = embedder.encode([chinese], normalize_embeddings=True)
    qv = np.array(qv, dtype=np.float32)
    scores, indices = index.search(qv, top_k)
    lines = []
    for s, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(corpus):
            e = corpus[idx]
            if 'amis' in e:
                lines.append(f"{e['amis']} | {e['chinese']}")
            else:
                lines.append(f"{e['paiwan']} | {e['chinese']}")
    return "\n".join(lines) if lines else "（未找到）"

def exact_match_search(chinese, corpus, core_reverse=None):
    """Exact match — 先查 core_reverse，再查語料"""
    if core_reverse and chinese in core_reverse:
        return core_reverse[chinese], True
    for e in corpus:
        if e.get('chinese', '').strip() == chinese.strip():
            key = 'amis' if 'amis' in e else 'paiwan'
            return e[key], True
    return None, False

def call_llm(prompt, user_msg, temp=0.2):
    try:
        r = client.chat.completions.create(model=MODEL, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ], temperature=temp, max_tokens=100)
        t = r.choices[0].message.content.strip()
        if '\n' in t: t = t.split('\n')[0].strip()
        # 清理解釋
        for marker in ['這個翻譯', '翻譯為', '以下是', '在排灣語', '根據']:
            if marker in t:
                parts = re.split(r'[。！？\n]', t)
                for p in parts:
                    p = p.strip()
                    if p and re.match(r'^[a-záéíóúàèìòùâêîôûäëïöüāēīōūʉəʼ\']', p, re.IGNORECASE) and len(p) > 3:
                        return p.rstrip('.')
                break
        return t
    except Exception as e:
        return f"[ERR:{e}]"

def roundtrip_verify(original, generated, lang_name, threshold=0.35):
    if not generated or generated.startswith("["): return False, 0.0
    back = call_llm(f"你是{lang_name}-中文翻譯助手。只輸出中文翻譯。", f"翻譯：{generated}", 0.1)
    if not back: return False, 0.0
    orig_c = set(original); back_c = set(back)
    if orig_c and back_c:
        overlap = len(orig_c & back_c) / max(len(orig_c | back_c), 1)
    else:
        overlap = 0.0
    return overlap >= threshold, overlap

# ============================================
# 實驗執行
# ============================================

def run_experiment(name, lang_name, dialect, test_set, corpus, index, embedder,
                   core_reverse, grammar_section, cfg):
    """跑一組實驗"""
    c = w = nf = 0
    details = []
    total_llm = 0
    
    for chinese in test_set:
        trans = None
        method = "none"
        
        # Layer 0: Core Vocab / Exact Match
        if cfg.get('use_exact'):
            em, is_exact = exact_match_search(chinese, corpus, core_reverse if cfg.get('use_core') else None)
            if not em and cfg.get('use_exact'):
                em, is_exact = exact_match_search(chinese, corpus, None)
            if em:
                trans = em
                method = "exact"
        
        # Layer 1: RAG + LLM
        if trans is None and cfg.get('use_rag'):
            rag_ctx = rag_search(chinese, corpus, index, embedder)
            if cfg.get('use_grammar') and grammar_section:
                prompt = GRAMMAR_PROMPT_TEMPLATE.format(lang_name=lang_name, rag_ctx=rag_ctx, grammar_section=grammar_section)
            else:
                prompt = NO_GRAMMAR_TEMPLATE.format(lang_name=lang_name, rag_ctx=rag_ctx)
            
            trans = call_llm(prompt, f"翻譯為{lang_name}：{chinese}")
            method = "rag_llm"
            total_llm += 1
            
            # Layer 2: Roundtrip Verify
            if cfg.get('use_verify') and trans and not trans.startswith("["):
                ok, score = roundtrip_verify(chinese, trans, lang_name)
                if not ok:
                    retry = call_llm(prompt, f"翻譯為{lang_name}：{chinese}", 0.1)
                    if retry and not retry.startswith("["):
                        _, rs = roundtrip_verify(chinese, retry, lang_name)
                        if rs > score:
                            trans = retry
        
        # Zero-shot fallback
        if trans is None:
            trans = call_llm(ZEROSHOT_TEMPLATE.format(lang_name=lang_name), f"翻譯：{chinese}")
            method = "zero_shot"
            total_llm += 1
        
        if trans is None: trans = "[NONE]"
        
        # Verify
        v = verify_cached(dialect, chinese, trans)
        if v == "correct": c += 1
        elif v == "partial": c += 1
        elif v == "not_found": nf += 1
        else: w += 1
        
        details.append({"chinese": chinese, "translation": trans, "method": method, "verdict": v})
    
    eff = len(test_set) - nf
    acc = c / eff * 100 if eff > 0 else 0
    return {"name": name, "correct": c, "wrong": w, "not_found": nf, "total": len(test_set),
            "accuracy": acc, "llm_calls": total_llm, "details": details}

# ============================================
# 載入語料
# ============================================

def load_paiwan():
    """載入排灣語語料（清洗後的，透過 translate_service）"""
    from translate_service import PaiwanTranslator
    t = PaiwanTranslator()
    t.load()
    
    corpus = t.corpus
    index = t.index
    
    # Core vocab reverse
    core_reverse = getattr(t, '_core_vocab_reverse', {})
    
    # Embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    return corpus, index, embedder, core_reverse, PAIWAN_GRAMMAR

def load_amis():
    """載入阿美語語料"""
    amis_dir = PROJECT_ROOT / "data" / "amis"
    
    with open(amis_dir / "amis_corpus.json") as f:
        raw = json.load(f)
    corpus = raw.get("entries", [])
    
    index = faiss.read_index(str(amis_dir / "amis_faiss.index"))
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 阿美語沒有 core vocab 和語法規則
    return corpus, index, embedder, {}, ""

# ============================================
# 主程式
# ============================================

def main():
    print("=" * 70)
    print("🔬 雙語言消融實驗")
    print(f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"測試集：{len(TEST_SET)} 條")
    print("=" * 70)
    
    # 快取 klokah
    cache_klokah(23)   # 排灣語
    cache_klokah(1)    # 阿美語
    
    # 載入語料
    print("\n載入排灣語...")
    pw_corpus, pw_index, pw_embedder, pw_core, pw_grammar = load_paiwan()
    print(f"  排灣語：{len(pw_corpus)} 筆，索引 {pw_index.ntotal}")
    
    print("載入阿美語...")
    am_corpus, am_index, am_embedder, am_core, am_grammar = load_amis()
    print(f"  阿美語：{len(am_corpus)} 筆，索引 {am_index.ntotal}")
    
    # 實驗配置
    experiments = [
        ("Control（4層全開）", {
            "use_core": True, "use_exact": True, "use_rag": True, "use_grammar": True, "use_verify": True,
        }),
        ("Exp1：關閉CoreVocab+ExactMatch", {
            "use_core": False, "use_exact": False, "use_rag": True, "use_grammar": True, "use_verify": True,
        }),
        ("Exp2：關閉Grammar語法注入", {
            "use_core": True, "use_exact": True, "use_rag": True, "use_grammar": False, "use_verify": True,
        }),
        ("Exp3：關閉往返驗證", {
            "use_core": True, "use_exact": True, "use_rag": True, "use_grammar": True, "use_verify": False,
        }),
    ]
    
    all_results = {}
    
    for lang, corpus, index, embedder, core, grammar, dialect in [
        ("排灣語", pw_corpus, pw_index, pw_embedder, pw_core, pw_grammar, 23),
        ("阿美語", am_corpus, am_index, am_embedder, am_core, am_grammar, 1),
    ]:
        print(f"\n{'='*70}")
        print(f"🧪 {lang}（d={dialect}，{len(corpus)} 筆語料）")
        print(f"{'='*70}")
        
        for exp_name, cfg in experiments:
            print(f"\n  --- {exp_name} ---")
            result = run_experiment(exp_name, lang, dialect, TEST_SET, corpus, index, embedder, core, grammar, cfg)
            
            eff = result["total"] - result["not_found"]
            print(f"    ✅ {result['correct']}/{eff} = {result['accuracy']:.1f}%  (wrong={result['wrong']}, nf={result['not_found']}, llm={result['llm_calls']})")
            
            if lang not in all_results:
                all_results[lang] = {}
            all_results[lang][exp_name] = result
    
    # ============================================
    # 對比表格
    # ============================================
    print("\n" + "=" * 70)
    print("📊 雙語言消融實驗對比表")
    print("=" * 70)
    
    exp_names = [e[0] for e in experiments]
    
    # Header
    print(f"| 實驗組 | 排灣語正確率 | 降幅 | 阿美語正確率 | 降幅 |")
    print(f"|--------|------------|------|------------|------|")
    
    pw_control = all_results["排灣語"]["Control（4層全開）"]["accuracy"]
    am_control = all_results["阿美語"]["Control（4層全開）"]["accuracy"]
    
    for exp_name in exp_names:
        pw = all_results["排灣語"][exp_name]["accuracy"]
        am = all_results["阿美語"][exp_name]["accuracy"]
        pw_drop = pw - pw_control
        am_drop = am - am_control
        
        pw_drop_str = f"{pw_drop:+.1f}pp" if exp_name != "Control（4層全開）" else "—"
        am_drop_str = f"{am_drop:+.1f}pp" if exp_name != "Control（4層全開）" else "—"
        
        print(f"| {exp_name} | **{pw:.1f}%** | {pw_drop_str} | **{am:.1f}%** | {am_drop_str} |")
    
    # 儲存結果
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_size": len(TEST_SET),
            "model": MODEL,
            "paiwan_corpus": len(pw_corpus),
            "amis_corpus": len(am_corpus),
        },
        "results": {},
    }
    for lang in ["排灣語", "阿美語"]:
        output["results"][lang] = {}
        for exp_name in exp_names:
            r = all_results[lang][exp_name]
            output["results"][lang][exp_name] = {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "wrong": r["wrong"],
                "not_found": r["not_found"],
                "llm_calls": r["llm_calls"],
            }
    
    out_path = PROJECT_ROOT / "results" / "bilingual_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已存：{out_path}")
    
    # 寫 memory 報告
    report_lines = [
        f"# 雙語言消融實驗結果",
        f"\n> 時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 測試集：{len(TEST_SET)} 條中文詞",
        f"> 驗證：klokah API（排灣 d=23, 阿美 d=1）",
        f"> 模型：{MODEL}",
        f"> P0 修復後（語料清洗 + 來源排序）",
        "",
        "## 對比表",
        "",
        "| 實驗組 | 排灣語 | 降幅 | 阿美語 | 降幅 |",
        "|--------|--------|------|--------|------|",
    ]
    for exp_name in exp_names:
        pw = all_results["排灣語"][exp_name]["accuracy"]
        am = all_results["阿美語"][exp_name]["accuracy"]
        pw_drop = f"{pw - pw_control:+.1f}pp" if exp_name != "Control（4層全開）" else "—"
        am_drop = f"{am - am_control:+.1f}pp" if exp_name != "Control（4層全開）" else "—"
        report_lines.append(f"| {exp_name} | {pw:.1f}% | {pw_drop} | {am:.1f}% | {am_drop} |")
    
    report_lines.extend([
        "",
        "## 各層貢獻（排灣語）",
    ])
    for exp_name in exp_names[1:]:
        pw = all_results["排灣語"][exp_name]["accuracy"]
        drop = pw - pw_control
        report_lines.append(f"- {exp_name}：{drop:+.1f}pp")
    
    report_lines.extend([
        "",
        "---",
        "*地陪自動生成*",
    ])
    
    mem_path = Path("/Users/sbb-mei/.openclaw/workspace/memory/2026-05-28-bilingual-ablation.md")
    with open(mem_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"📝 報告已存：{mem_path}")

if __name__ == "__main__":
    main()
