"""Figure: Gated DeltaNet state used as RAM: (a) write, (b) read, (c) a CPU laid out in the state.

python docs/figures/gdn_ram.py  ->  docs/gdn-ram.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

BLUE, LIGHT, GREY, RED, DARK, NOTE = "#1f5fae", "#e8eef7", "#8a8a8a", "#d65f5f", "#222222", "#4a4a4a"
ROWS, COLS, R = 8, 6, 3
CELL = 0.5
TITLE_Y, FORMULA_Y, TOP, CAPTION_Y = 9.55, 8.55, 7.0, 0.85   # shared rows across all panels

rng = np.random.default_rng(7)
S0 = rng.integers(0, 2, (ROWS, COLS))
V = np.array([1, 0, 1, 1, 0, 1])
S1 = S0.copy()
S1[R] = V
ONEHOT = np.eye(ROWS, dtype=int)[R]
ROW_Y = TOP - (R + 0.5) * CELL                     # centre height of row r


def grid(ax, x, y, M, cell=CELL, hi_row=None):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   facecolor=BLUE if M[i, j] else LIGHT, edgecolor="white", lw=1.2))
    if hi_row is not None:
        ax.add_patch(Rectangle((x, y - (hi_row + 1) * cell), M.shape[1] * cell, cell,
                               fill=False, edgecolor=RED, lw=2.4))


def vec(ax, x, y, v, cell=CELL, vertical=True, color=BLUE):
    for i, b in enumerate(v):
        px, py = (x, y - (i + 1) * cell) if vertical else (x + i * cell, y - cell)
        ax.add_patch(Rectangle((px, py), cell, cell, facecolor=color if b else LIGHT,
                               edgecolor="white", lw=1.2))


def label(ax, x, text, y=TOP + 0.22, **kw):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=15, **kw)


def arrow(ax, p, q, color=DARK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=color, lw=1.5,
                                 connectionstyle=f"arc3,rad={rad}"))


def panel(ax, x0, x1, tag, title):
    ax.add_patch(FancyBboxPatch((x0, 0.25), x1 - x0, 9.85, boxstyle="round,pad=0,rounding_size=0.25",
                                fill=False, edgecolor="#c9c9c9", lw=1.3))
    ax.text(x0 + 0.35, TITLE_Y - 0.18, tag, fontsize=19, fontweight="bold", va="baseline")
    ax.text(x0 + 0.9, TITLE_Y - 0.18, title, fontsize=17, va="baseline")
    return (x0 + x1) / 2


fig = plt.figure(figsize=(22, 6.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 34.0)
ax.set_ylim(0, 10.6)
ax.set_aspect("equal")
ax.axis("off")

# (a) write ------------------------------------------------------------------------------------
cx = panel(ax, 0.2, 12.6, "a", "Write a row")
ax.text(cx, FORMULA_Y, r"$S_t = \alpha_t S_{t-1} + \beta_t\, k_t\,(v_t - \alpha_t S_{t-1}^{T} k_t)^{T}$",
        ha="center", va="center", fontsize=16)
vec(ax, 0.9, TOP, ONEHOT, color=RED)
label(ax, 0.9 + CELL / 2, r"key $k_t = e_r$")
grid(ax, 2.6, TOP, S0)
label(ax, 2.6 + COLS * CELL / 2, r"state $S_{t-1}$")
arrow(ax, (5.9, ROW_Y), (7.0, ROW_Y))
grid(ax, 7.3, TOP, S1, hi_row=R)
label(ax, 7.3 + COLS * CELL / 2, r"state $S_t$")
ax.text(10.5, ROW_Y, "row $r$\noverwritten", color=RED, fontsize=12.5, va="center")
vec(ax, 2.6, 2.35, V, vertical=False, color=RED)
ax.text(2.35, 2.35 - CELL / 2, r"value $v_t$", ha="right", va="center", fontsize=15)
ax.text(cx, CAPTION_Y, r"$\alpha_t = 1$ (gate open), $\beta_t = 1$, $k_t = e_r$: row $r$ becomes $v_t$ exactly",
        ha="center", va="center", fontsize=12.5, color=NOTE)

# (b) read -------------------------------------------------------------------------------------
cx = panel(ax, 12.9, 21.5, "b", "Read it back")
ax.text(cx, FORMULA_Y, r"$o_t = S_t^{T} q_t$", ha="center", va="center", fontsize=16)
vec(ax, 14.0, TOP, ONEHOT, color=RED)
label(ax, 14.0 + CELL / 2, r"query $q_t = e_r$")
grid(ax, 15.2, TOP, S1, hi_row=R)
label(ax, 15.2 + COLS * CELL / 2, r"$S_t$")
arrow(ax, (18.35, ROW_Y), (19.05, ROW_Y), color=RED)
vec(ax, 19.2, ROW_Y + 0.17, V, cell=0.34, vertical=False, color=BLUE)
label(ax, 19.2 + 3 * 0.34, r"$o_t$", y=ROW_Y + 0.35)
ax.text(cx, CAPTION_Y, "a one-hot query returns exactly one row", ha="center", va="center",
        fontsize=12.5, color=NOTE)

# (c) a CPU in the state -----------------------------------------------------------------------
cx = panel(ax, 21.8, 33.8, "c", "Run a CPU on it")
ax.text(cx, FORMULA_Y, "rows hold registers, memory bytes and the program counter",
        ha="center", va="center", fontsize=13, color=NOTE)
C = 0.45
heads = [("registers", 22.9, np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]]), None),
         ("RAM", 26.1, rng.integers(0, 2, (5, 4)), 1),
         ("program counter", 29.7, np.array([[0, 1, 1, 0]]), None)]
for name, x, M, hi in heads:
    grid(ax, x, TOP, M, cell=C, hi_row=hi)
    label(ax, x + M.shape[1] * C / 2, name)
tape = [("ADDR 0x661", GREY), ("VAL 2", RED), ("ADDR PC", GREY), ("VAL 0x20C", RED), ("STEP", DARK)]
widths = [2.15, 1.45, 1.75, 2.0, 1.2]
x, tops = 22.35, []
for (text, col), w in zip(tape, widths):
    ax.add_patch(Rectangle((x, 2.0), w, 0.8, facecolor="white", edgecolor=col, lw=1.6))
    ax.text(x + w / 2, 2.4, text, ha="center", va="center", fontsize=11, family="monospace",
            color="#555" if col == GREY else col)
    tops.append(x + w / 2)
    x += w + 0.18
arrow(ax, (tops[1], 2.85), (26.1 + 2 * C, TOP - 1.5 * C - 0.05), color=RED, rad=-0.12)
arrow(ax, (tops[3], 2.85), (29.7 + 2 * C, TOP - C - 0.05), color=RED, rad=0.18)
ax.text(cx, CAPTION_Y, "each generated trace token is one write; a few tokens make one instruction",
        ha="center", va="center", fontsize=12.5, color=NOTE)

out = Path(__file__).resolve().parents[1] / "gdn-ram.png"
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print(out)
