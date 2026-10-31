"""Benchmark bar chart: base Qwen3.5-4B vs Qwenputer vs base + Python tool.

python docs/figures/benchmarks.py  ->  docs/benchmarks.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

sets = ["NUPA\n(3-20 digits)", "DeepMind\ndiv_big", "DeepMind\nmul_big", "DeepMind\n6 sets pooled",
        "DeepMind clean\nadd / mul"]
base = [48.6, 77, 66, 80.8, 69.5]
qp = [100.0, 97, 93, 92.3, 99.0]
tool = [100.0, 85, 91, 94.2, 98.5]

x = np.arange(len(sets))
w = 0.26
fig, ax = plt.subplots(figsize=(11, 4.6))
cols = {"base": "#b9b9b9", "qp": "#7b6cf0", "tool": "#e3a76f"}
b1 = ax.bar(x - w, base, w, color=cols["base"], label="Qwen3.5-4B")
b2 = ax.bar(x, qp, w, color=cols["qp"], label="Qwenputer (computer in the weights)")
b3 = ax.bar(x + w, tool, w, color=cols["tool"], label="Qwen3.5-4B + Python tool")
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():g}",
                ha="center", va="bottom", fontsize=10.5, color="#333")
ax.set_xticks(x, sets, fontsize=11.5)
ax.set_ylabel("accuracy (%)", fontsize=12.5)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
ax.tick_params(axis="y", labelsize=11)
out = Path(__file__).resolve().parents[1] / "benchmarks.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(out)
