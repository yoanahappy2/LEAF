#!/usr/bin/env python3
"""Generate ablation plot for LEAF Poster"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

# 中文字體
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(PROJECT_ROOT / "experiment_results" / "ablation_v2_20260604_224233.json"))

# === 圖 1: Accuracy Comparison Bar Chart ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

names = ["LLM Direct", "RAG Only", "Single Agent", "Multi-Agent"]
keys = ["llm_direct", "rag_only", "single_agent", "multi_agent"]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
accuracies = [data[k]["accuracy"] * 100 for k in keys]
avg_scores = [data[k]["avg_score"] for k in keys]

# Plot 1: Accuracy
ax1 = axes[0]
bars = ax1.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Translation Accuracy")
ax1.set_ylim(0, 100)
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{val:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.axhline(y=70, color="gray", linestyle="--", alpha=0.3, label="70% baseline")

# Plot 2: Error Breakdown
ax2 = axes[1]
gt = json.load(open(PROJECT_ROOT / "benchmark" / "ground_truth.json"))
gt_map = {q["id"]: q for q in gt["questions"]}

# 分類: all correct, MA-only, all wrong
all_correct = 0
ma_only = 0
rag_only_fail = 0  # RAG fails, MA also fails
llm_fail_all = 0

for i in range(20):
    qid = data["llm_direct"]["results"][i]["id"]
    rag_ok = data["rag_only"]["results"][i].get("correct", False)
    sa_ok = data["single_agent"]["results"][i].get("correct", False)
    ma_ok = data["multi_agent"]["results"][i].get("correct", False)
    
    if rag_ok and ma_ok:
        all_correct += 1
    elif ma_ok and not rag_ok:
        ma_only += 1
    elif not ma_ok:
        rag_only_fail += 1

categories = ["All Correct\n(RAG+MA)", "MA Rescued\n(MA-only)", "All Failed"]
counts = [all_correct, ma_only, rag_only_fail]
cat_colors = ["#2ca02c", "#1f77b4", "#d62728"]
bars2 = ax2.bar(categories, counts, color=cat_colors, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Number of Questions")
ax2.set_title("Error Breakdown (n=20)")
for bar, val in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

# Plot 3: Avg Score (exact vs variant)
ax3 = axes[2]
# Count exact vs variant matches for each model
exact_counts = []
variant_counts = []
fail_counts = []

for k in keys:
    exact = sum(1 for r in data[k]["results"] if r.get("match_level") == "exact")
    variant = sum(1 for r in data[k]["results"] if r.get("match_level") == "variant")
    fail = sum(1 for r in data[k]["results"] if r.get("match_level") not in ("exact", "variant"))
    exact_counts.append(exact)
    variant_counts.append(variant)
    fail_counts.append(fail)

x = np.arange(len(names))
width = 0.6
p1 = ax3.bar(x, exact_counts, width, label="Exact Match (1.0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
p2 = ax3.bar(x, variant_counts, width, bottom=exact_counts, label="Variant Match (0.8)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
p3 = ax3.bar(x, fail_counts, width, bottom=[e+v for e,v in zip(exact_counts, variant_counts)], label="No Match (0.0)", color="#d62728", edgecolor="black", linewidth=0.5)

ax3.set_ylabel("Number of Questions")
ax3.set_title("Match Type Distribution")
ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=9)
ax3.legend(loc="upper left", fontsize=8)

plt.tight_layout()
out_path = PROJECT_ROOT / "benchmark" / "ablation_plot.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", out_path)

# Also save a summary JSON for the poster
summary = {
    "results": {k: {"accuracy": data[k]["accuracy"], "avg_score": data[k]["avg_score"],
                     "correct": data[k]["correct"], "total": data[k]["total"]} for k in keys},
    "ma_rescued": ma_only,
    "all_failed": rag_only_fail,
    "all_correct": all_correct,
}
with open(PROJECT_ROOT / "benchmark" / "ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: ablation_summary.json")
