#!/usr/bin/env python3
"""Generate A-line poster chart: ReAct-Orchestrated Framework"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(PROJECT_ROOT / "experiment_results" / "ablation_v2_20260604_224233.json"))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle("LEAF: ReAct-Orchestrated Translation for Endangered Languages", 
             fontsize=14, fontweight="bold", y=1.02)

# === Labels ===
names = ["LLM\nDirect", "Single-Shot\nRAG", "Single\nAgent", "ReAct\nOrchestrator"]
keys = ["llm_direct", "rag_only", "single_agent", "multi_agent"]
colors = ["#bdbdbd", "#ff7f0e", "#2ca02c", "#1f77b4"]
accuracies = [data[k]["accuracy"] * 100 for k in keys]

# === Plot 1: Accuracy ===
ax1 = axes[0]
bars = ax1.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("Translation Accuracy (n=20)", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{val:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

# Arrow showing the gap
ax1.annotate("", xy=(3, 85), xytext=(1, 70),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
ax1.text(2.0, 80, "+15pp", ha="center", fontsize=11, color="#1f77b4", fontweight="bold")

# === Plot 2: Error Breakdown ===
ax2 = axes[1]

all_correct = sum(1 for i in range(20) 
                  if data["rag_only"]["results"][i].get("correct") 
                  and data["multi_agent"]["results"][i].get("correct"))
react_rescued = sum(1 for i in range(20)
                    if not data["rag_only"]["results"][i].get("correct")
                    and data["multi_agent"]["results"][i].get("correct"))
all_failed = sum(1 for i in range(20)
                 if not data["multi_agent"]["results"][i].get("correct"))

categories = ["Both Correct\n(Single-shot OK)", "ReAct Rescued\n(RAG fail → OK)", "All Failed"]
counts = [all_correct, react_rescued, all_failed]
cat_colors = ["#2ca02c", "#1f77b4", "#d62728"]

bars2 = ax2.bar(categories, counts, color=cat_colors, edgecolor="black", linewidth=0.5, width=0.55)
ax2.set_ylabel("Questions", fontsize=11)
ax2.set_title("Error Breakdown", fontsize=12, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for bar, val in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", fontweight="bold", fontsize=13)

# === Plot 3: Match Type ===
ax3 = axes[2]

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
p3 = ax3.bar(x, fail_counts, width, 
             bottom=[e+v for e,v in zip(exact_counts, variant_counts)], 
             label="No Match (0.0)", color="#d62728", edgecolor="black", linewidth=0.5)

ax3.set_ylabel("Questions", fontsize=11)
ax3.set_title("Match Type Distribution", fontsize=12, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=9)
ax3.legend(loc="upper left", fontsize=8)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
out = PROJECT_ROOT / "benchmark" / "ablation_plot_A.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved:", out)
