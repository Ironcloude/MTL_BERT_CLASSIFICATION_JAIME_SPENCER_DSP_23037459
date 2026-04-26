"""
Grouped bar chart: F1 by region (UK/US) and timeframe (Cont/Arch)
for selected golden set models.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from theme import Colours, Fonts

plt.style.use(Path(__file__).parent / "style.mplstyle")

FIG_DIR = Path(__file__).parents[2] / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
MODELS = {
    "EX-2: DeBERTav3 STL (512)":          {"uk": 0.6056, "us": 0.4701, "cont": 0.5588, "arch": 0.4970},
    "EX-12: ModernBERT MTL (λ=0.25, 1024)": {"uk": 0.5086, "us": 0.4894, "cont": 0.5562, "arch": 0.4139},
}

CATEGORIES = ["UK", "US", "Contemporary", "Archival"]
KEYS = ["uk", "us", "cont", "arch"]
MODEL_COLOURS = [Colours.BLUE, Colours.RED]

# ── Plot ────────────────────────────────────────────────────────────────────
model_names = list(MODELS.keys())
n_models = len(model_names)
n_cats = len(CATEGORIES)
x = np.arange(n_cats)
width = 0.7 / n_models

fig, ax = plt.subplots(figsize=(6.5, 3.5))

for i, name in enumerate(model_names):
    vals = [MODELS[name][k] for k in KEYS]
    offset = (i - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width, label=name,
                  color=MODEL_COLOURS[i % len(MODEL_COLOURS)],
                  edgecolor="white", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES)
ax.set_ylabel("Macro F1")
ax.set_ylim(0, 0.75)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Golden Set - Geographic & Temporal Breakdown")

fig.tight_layout()
fig.savefig(FIG_DIR / "golden_f1_breakdown.png")
plt.close()
print(f"Saved → {FIG_DIR / 'golden_f1_breakdown.png'}")