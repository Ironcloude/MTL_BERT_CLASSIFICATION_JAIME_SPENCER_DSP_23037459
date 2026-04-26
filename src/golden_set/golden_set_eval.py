"""
Evaluates trained models on held-out Golden Set.

Figure generation is largely AI-generated via Claude.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from safetensors.torch import load_file
from sklearn.metrics import f1_score, accuracy_score, classification_report
from types import SimpleNamespace
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "figures"))
from pathlib import Path
from data_cleaning.data_cleaner import DataCleaner
from theme import Colours, Fonts, DISPLAY_NAMES, POLITICAL_CMAP
style_path = Path(__file__).parents[1] / "figures" / "style.mplstyle"
plt.style.use(style_path)

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

GOLDEN_PATH = os.path.join(project_root, "data/golden_set/golden_articles.csv")
METRICS_OUT = os.path.join(project_root, f"results/metrics/golden_set_results_{int(time.time())}.csv")
FIGURES_DIR = os.path.join(project_root, "results/figures")


# Config
STRICT_LABELS_ONLY = True
CHERRY_PICK_MODEL = True # scans checkpoints and keeps only the best per experiment
SELECT_MODELS = [ "EX-2", "EX-12"]  
SAVE = True
PLOT_OUTLET = False
PRINT_BREAKDOWN = False
PLOT_HEATMAP = False
PLOT_COMPARISON = True  # side-by-side figures for SELECT_MODELS
# label mappings 
LABEL_NAMES = ["Left", "Centre", "Right"]

BIAS_TO_INT = {
    "left":   0,
    "center": 1,
    "right":  2,
}

OUTLET_REGION = {
    "The Guardian":  "UK",
    "BBC News":      "UK",
    "Daily Mail":    "UK",
    "Newsweek":      "US",
    "HuffPost":      "US",
    "New York Post": "US",
}

# Model repo lookup for tokenizer + MTL encoder loading
MODEL_REPOS = {
    "Mod": "answerdotai/ModernBERT-base",
    "DeB": "microsoft/deberta-v3-base",
    "BER": "bert-base-uncased",
    "ELE": "google/electra-base-discriminator",
}

# Figure colours

BIAS_COLOUR = {
    "left":       Colours.BLUE,
    "lean left":  Colours.LIGHT_BLUE,
    "center":     Colours.GREY_LIGHT,
    "lean right": Colours.LIGHT_RED,
    "right":      Colours.RED,
}
PREDICTION_COLOURS = {
    "Left":   Colours.BLUE,
    "Centre": Colours.GREY_LIGHT,
    "Right":  Colours.RED,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_models():
    """Scan models/final/ for saved models with model.safetensors at root."""
    final_dir = os.path.join(project_root, "models/final")
    if not os.path.isdir(final_dir):
        print(f"Models directory not found: {final_dir}")
        exit(1)

    models = []
    for folder in sorted(os.listdir(final_dir)):
        folder_path = os.path.join(final_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        if not os.path.exists(os.path.join(folder_path, "model.safetensors")):
            print(f"  [SKIP] No model.safetensors in {folder}")
            continue

        parts = folder.split("-")
        if len(parts) < 2 or not parts[0].startswith("EX"):
            print(f"  [SKIP] Not an experiment folder: {folder}")
            continue
        try:
            seq_len = int(parts[-3])
        except (ValueError, IndexError):
            print(f"  [SKIP] Could not parse seq_len from: {folder}")
            continue

        models.append((folder, folder_path, seq_len))

    models.sort(key=lambda model: int(model[0].split("-")[1]))
    return models


def get_display_name(model_name):
    """Extract EX-ID and return the mapped display name, falling back to raw name."""
    parts = model_name.split("-")
    if len(parts) >= 2 and parts[0] == "EX":
        ex_id = f"{parts[0]}-{parts[1]}"
        disp = DISPLAY_NAMES.get(ex_id)
        if disp:
            return f"{ex_id}: {disp}"
    return model_name

def _infer_repo(name, ckpt_path=None):
    """Infer HuggingFace model repo from experiment folder name.
    For DeBERTa models, checks config.json to distinguish v1 from v3."""
    if "DeB" in name and ckpt_path:
        import json
        cfg_file = os.path.join(ckpt_path, "config.json")
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                model_type = json.load(f).get("model_type", "")
            if model_type == "deberta":
                return "microsoft/deberta-base"
            return "microsoft/deberta-v3-base"
    for prefix, repo in MODEL_REPOS.items():
        if prefix in name:
            return repo
    raise ValueError(f"Cannot infer model repo from: {name}")

# Custom laoder for MTL  models 
class _MTLEval(nn.Module):
    """Inference wrapper matching train_bulk.py's MultiTaskModel structure.
    MeanPool => Dense(768,768) => GELU => LayerNorm => Linear(768, n)"""

    def __init__(self, model_path, num_labels=3):
        super().__init__()
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            self.encoder = AutoModel.from_pretrained(model_path)
        else:
            exp_folder = os.path.basename(os.path.dirname(model_path))
            self.encoder = AutoModel.from_pretrained(_infer_repo(exp_folder))

        hidden = self.encoder.config.hidden_size
        # Political head matching train_bulk.py MultiTaskModel
        self.pol_dense = nn.Linear(hidden, hidden, bias=False)
        self.pol_act = nn.GELU()
        self.pol_norm = nn.LayerNorm(hidden)
        self.political_head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        pol = self.pol_norm(self.pol_act(self.pol_dense(pooled)))
        return SimpleNamespace(logits=self.political_head(pol))


def load_model(ckpt_path, model_name=""):
    """Load a checkpoint, handling both standard HF and custom MT layouts."""
    is_mt = "-mt-" in model_name.lower() or "λ" in model_name

    if is_mt:
        model = _MTLEval(ckpt_path)
        weights = load_file(os.path.join(ckpt_path, "model.safetensors"))
        weights = {k: v for k, v in weights.items() if not k.startswith("sent")}
        model.load_state_dict(weights, strict=True)
        return model
    else:
        return AutoModelForSequenceClassification.from_pretrained(ckpt_path)


def predict_proba(model, tokenizer, texts, seq_len, batch_size=16):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(all_probs)


def plot_heatmap(df_results, model_name, out_path):
    """3×3 confusion heatmap (column-normalised)."""
    disp_name = get_display_name(model_name)
    col_order = ["left", "center", "right"]
    matrix = pd.crosstab(
        df_results["predicted_label"],
        df_results["bias"],
        normalize="columns",
    ).reindex(index=LABEL_NAMES, columns=col_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=POLITICAL_CMAP,
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title(f"Golden Set — {disp_name}\n(column-normalised, n={len(df_results)})")
    ax.set_xlabel("AllSides Bias Category")
    ax.set_ylabel("Predicted Class")
    fig.tight_layout()
    if SAVE and not CHERRY_PICK_MODEL:
        fig.savefig(out_path)
    plt.close()


def plot_outlet_bars(df_results, model_name, out_path):
    """Stacked bar chart per outlet+timeframe showing prediction distribution."""
    disp_name = get_display_name(model_name)
    rows = []
    for outlet, grp in df_results.groupby("outlet"):
        bias = grp["bias"].iloc[0]
        for tf in ["contemporary", "archival"]:
            grp_tf = grp[grp["timeframe"] == tf]
            if len(grp_tf) == 0:
                continue
            dist = grp_tf["predicted"].value_counts(normalize=True).to_dict()
            rows.append({
                "outlet": outlet, "timeframe": tf, "bias": bias,
                "region": OUTLET_REGION.get(outlet, "?"),
                "n": len(grp_tf),
                "pct_Left": dist.get(0, 0), "pct_Centre": dist.get(1, 0), "pct_Right": dist.get(2, 0),
            })
    summary = pd.DataFrame(rows)
    summary["label"] = summary.apply(
        lambda r: f"{r['outlet']}  [{r['bias']}, {r['region']}] ({r['timeframe'][0].upper()}, n={r['n']})",
        axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(summary) + 2))
    lefts = np.zeros(len(summary))
    for label, col in PREDICTION_COLOURS.items():
        vals = summary[f"pct_{label}"].values
        ax.barh(range(len(summary)), vals, left=lefts, color=col,
                label=label, edgecolor="white", linewidth=0.4)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 0.08:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["label"].values, fontsize=8)
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.get_yticklabels()[i].set_color(BIAS_COLOUR[row["bias"]])
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of predictions")
    ax.set_title(f"Per-outlet prediction distribution\n{disp_name}")
    ax.axvline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.legend(title="Predicted", loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    if SAVE and not CHERRY_PICK_MODEL:
        fig.savefig(out_path)
    plt.close()


def plot_heatmap_comparison(results_by_model, out_path):
    """Side-by-side confusion heatmaps for SELECT_MODELS."""
    models = [m for m in SELECT_MODELS if m in results_by_model]
    fig, axes = plt.subplots(1, len(models), figsize=(6.5, 3.5))
    if len(models) == 1:
        axes = [axes]
    col_order = ["left", "center", "right"]
    for ax, ex_id in zip(axes, models):
        df_res, model_name = results_by_model[ex_id]
        disp_name = get_display_name(model_name)
        matrix = pd.crosstab(
            df_res["predicted_label"], df_res["bias"],
            normalize="columns",
        ).reindex(index=LABEL_NAMES, columns=col_order, fill_value=0)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=POLITICAL_CMAP,
                    vmin=0, vmax=1, linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(f"{disp_name}\n(n={len(df_res)})", fontsize=9)
        ax.set_xlabel("AllSides Bias Category")
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Predicted Class")
    fig.supxlabel("Golden Set Confusion Heatmaps (STL vs. MTL)")
    fig.tight_layout()
    if SAVE:
        fig.savefig(out_path)
    plt.close()


def plot_outlet_comparison(results_by_model, out_path):
    """Side-by-side outlet bar charts for SELECT_MODELS."""
    models = [m for m in SELECT_MODELS if m in results_by_model]
    _BIAS_ORDER = {"left": 0, "lean left": 1, "center": 2, "lean right": 3, "right": 4}

    def _build_summary(df_res):
        rows = []
        for outlet, grp in df_res.groupby("outlet"):
            bias = grp["bias"].iloc[0]
            for tf in ["contemporary", "archival"]:
                grp_tf = grp[grp["timeframe"] == tf]
                if len(grp_tf) == 0:
                    continue
                dist = grp_tf["predicted"].value_counts(normalize=True).to_dict()
                rows.append({
                    "outlet": outlet, "timeframe": tf, "bias": bias,
                    "region": OUTLET_REGION.get(outlet, "?"),
                    "n": len(grp_tf),
                    "pct_Left": dist.get(0, 0), "pct_Centre": dist.get(1, 0), "pct_Right": dist.get(2, 0),
                })
        summary = pd.DataFrame(rows)
        summary["bias_rank"] = summary["bias"].map(_BIAS_ORDER)
        summary = summary.sort_values(["bias_rank", "outlet", "timeframe"],
                                       ascending=[True, True, False]).reset_index(drop=True)
        return summary

    summaries = {ex: _build_summary(results_by_model[ex][0]) for ex in models}
    max_rows = max(len(s) for s in summaries.values())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5, 0.6 * max_rows + 1.5))
    if len(models) == 1:
        axes = [axes]

    for ax, ex_id in zip(axes, models):
        df_res, model_name = results_by_model[ex_id]
        summary = summaries[ex_id]
        disp_name = get_display_name(model_name)
        lefts = np.zeros(len(summary))
        for label, col in PREDICTION_COLOURS.items():
            vals = summary[f"pct_{label}"].values
            ax.barh(range(len(summary)), vals, left=lefts, color=col,
                    label=label, edgecolor="white", linewidth=0.4)
            for i, (v, l) in enumerate(zip(vals, lefts)):
                if v >= 0.08:
                    ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            lefts += vals
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_title(disp_name, fontsize=9)
        ax.invert_yaxis()

        # Two-line y-labels on left panel only
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels([""] * len(summary))
        if ax == axes[0]:
            trans = ax.get_yaxis_transform()
            for i, (_, row) in enumerate(summary.iterrows()):
                ax.text(-0.04, i, row["outlet"],
                        transform=trans, color=BIAS_COLOUR[row["bias"]],
                        **{**Fonts.EX_MAIN, "fontsize": 8.5})
                detail = f"{row['bias']}, {row['region']}  ({row['timeframe'][0].upper()}, n={row['n']})"
                ax.text(-0.04, i + 0.3, detail,
                        transform=trans, **{**Fonts.EX_SUB, "fontsize": 7})

        if ax == axes[-1]:
            ax.legend(title="Predicted", loc="lower right", fontsize=7)

    fig.suptitle("Per-outlet prediction distribution")
    fig.supxlabel("Proportion of predictions")
    plt.subplots_adjust(left=0.25)
    fig.tight_layout()
    plt.subplots_adjust(left=0.25)
    if SAVE:
        fig.savefig(out_path)
    plt.close()


def plot_model_comparison_overview(master_df, out_path, ckpt_histories=None):
    """Combined bar chart comparing all models' macro F1, grouped by experiment type.
    If ckpt_histories is provided, overlays checkpoint F1 dots on each bar."""
    df = master_df.sort_values("macro_f1", ascending=True).copy()

    _EX_GROUPS = {
        "EX-1": "Architecture", "EX-2": "Architecture", "EX-3": "Architecture",
        "EX-4": "Architecture", "EX-5": "Architecture",
        "EX-6": "Seq Length", "EX-7": "Seq Length",
        "EX-8": "Ablation",
        "EX-9": "MTL", "EX-10": "MTL", "EX-11": "MTL", "EX-12": "MTL", "EX-13": "MTL", "EX-14": "MTL",
    }

    def group_of(name):
        ex = name.split("-")[0] + "-" + name.split("-")[1]
        return _EX_GROUPS.get(ex, "Other")

    df["group"] = df["model"].apply(group_of)
    group_colours = {
        "Architecture": Colours.BLUE,
        "Seq Length":   Colours.LIGHT_BLUE,
        "MTL":          Colours.RED,
        "Ablation":     Colours.GREY_LIGHT,
    }
    colours = df["group"].map(group_colours).values
    n = len(MODELS)
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * n + 1.5))
    bars = ax.barh(range(len(df)), df["macro_f1"].values, color=colours, edgecolor="white", linewidth=0.4)

    for i, v in enumerate(df["macro_f1"].values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([])
    for i, (_, row) in enumerate(df.iterrows()):
        eid = row["model"].split("-")[0] + "-" + row["model"].split("-")[1]
        desc = DISPLAY_NAMES.get(eid, "")
        ax.text(-0.02, i + 0.15, eid, **Fonts.EX_MAIN,
                transform=ax.get_yaxis_transform())
        ax.text(-0.02, i - 0.2, desc, **Fonts.EX_SUB,
                transform=ax.get_yaxis_transform())

    ax.set_xlabel("Macro F1 (Golden Set)")
    ax.set_xlim(0, max(0.6, df["macro_f1"].max() + 0.05))

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colours.items()
                       if g in df["group"].values]
    if ckpt_histories:
        legend_elements.append(Line2D([0], [0], marker="|", color=Colours.GREY_DARK,
                               linestyle="None", markersize=8, label="Checkpoints"))
        legend_elements.append(Line2D([0], [0], marker="d", color="black",
                               linestyle="None", markersize=5, label="Saved model"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.suptitle("Model Comparison — Golden Set Macro F1", x=0.5, ha='center')
    fig.tight_layout()
    if SAVE:
        fig.savefig(out_path)
        print(f"Saved comparison chart → {out_path}")
    plt.close()

def plot_checkpoint_progression(all_ckpt_histories, out_path):
    from matplotlib.patches import Patch

    _EX_GROUPS = {
        "EX-1": "Architecture", "EX-2": "Architecture", "EX-3": "Architecture",
        "EX-4": "Architecture", "EX-5": "Architecture",
        "EX-6": "Seq Length", "EX-7": "Seq Length",
        "EX-8": "Ablation",
        "EX-9": "MTL", "EX-10": "MTL", "EX-11": "MTL", "EX-12": "MTL",
        "EX-13": "MTL", "EX-14": "MTL",
    }
    group_colours = {
        "Architecture": Colours.BLUE,
        "Seq Length":   Colours.LIGHT_BLUE,
        "MTL":          Colours.RED,
        "Ablation":     Colours.GREY_LIGHT,
    }

    def ex_id(name):
        parts = name.split("-")
        return f"{parts[0]}-{parts[1]}"

    def parse_step(lbl):
        if lbl == "best_eval":
            return float('inf')
        try:
            return int(lbl.split("-")[1])
        except (ValueError, IndexError):
            return -1

    # Sort by best F1 (ascending so best is at top of horizontal chart)
    ex_order = sorted(all_ckpt_histories.keys(),
                      key=lambda n: max(h["f1"] for h in all_ckpt_histories[n]))
    n_exps = len(ex_order)
    max_ckpts = max(len(h) for h in all_ckpt_histories.values())

    bar_h = 0.75 / max_ckpts 

    # Control overall figure height (0.35 inches per experiment keeps it compact)
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * n_exps + 1.5))
    y_positions = []

    for row_idx, model_name in enumerate(ex_order):
        history = sorted(all_ckpt_histories[model_name], key=lambda h: parse_step(h["label"]))
        n_ckpts = len(history)
        offsets = np.linspace(-bar_h * (n_ckpts - 1) / 2, bar_h * (n_ckpts - 1) / 2, n_ckpts)

        eid = ex_id(model_name)
        group = _EX_GROUPS.get(eid, "Other")
        base_colour = group_colours.get(group, Colours.GREY_LIGHT)

        # Find best checkpoint F1 (excluding best_eval) to highlight when it beats saved
        best_ckpt_f1 = max((h["f1"] for h in history if h["label"] != "best_eval"), default=-1)
        saved_f1 = next((h["f1"] for h in history if h["label"] == "best_eval"), -1)

        for ckpt, offset in zip(history, offsets):
            y = row_idx + offset
            is_saved = ckpt["label"] == "best_eval"
            is_best_ckpt = (not is_saved and ckpt["f1"] == best_ckpt_f1
                            and best_ckpt_f1 > saved_f1)

            if is_saved:
                alpha, edge, lw, zorder = 1.0, "black", 0.8, 3
            elif is_best_ckpt:
                alpha, edge, lw, zorder = 0.55, "none", 0.5, 2
            else:
                alpha, edge, lw, zorder = 0.35, "none", 0, 2

            ax.barh(y, ckpt["f1"], height=bar_h * 0.85, color=base_colour,
                    alpha=alpha, edgecolor=edge, linewidth=lw, zorder=zorder)

            if is_saved:
                step_label = f'ES  ({ckpt["f1"]:.3f})'
            else:
                step = ckpt["label"].replace("checkpoint-", "")
                step_label = f'Step {step} ({ckpt["f1"]:.3f})'

            ax.text(ckpt["f1"] - 0.005, y, step_label,
                    va="center", ha="right", fontsize=5, zorder=5,
                    color="white" if is_saved else Colours.GREY_DARK,
                    fontweight="bold" if is_saved or is_best_ckpt else "normal")

        y_positions.append(row_idx)

        # Divider line between experiments
        if row_idx < n_exps - 1:
            ax.axhline(row_idx + 0.5, color="#000000", linewidth=0.5, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([get_display_name(m) for m in ex_order], fontsize=7)
    ax.set_xlabel("Macro F1 (Golden Set)")
    ax.set_xticks(np.arange(0, 0.6, 0.05))

    legend_elements = [Patch(facecolor=c, label=g) for g, c in group_colours.items()
                       if g in [_EX_GROUPS.get(ex_id(m), "Other") for m in ex_order]]
    legend_elements.append(Patch(facecolor=Colours.GREY_LIGHT, edgecolor="black",
                                 linewidth=1, label="Early-stopped model"))
    legend_elements.append(Patch(facecolor=Colours.GREY_LIGHT, label="Checkpoint"))
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6)
    fig.suptitle("Model Comparison - Golden Set", x=0.5, ha="center")

    fig.tight_layout()
    if SAVE:
        fig.savefig(out_path)
        print(f"Saved checkpoint progression => {out_path}")
    plt.close()


def outlet_summary(df_results):
    rows = []
    for outlet, grp in df_results.groupby("outlet"):
        bias = grp["bias"].iloc[0]
        dist = grp["predicted"].value_counts(normalize=True).to_dict()
        grp_c = grp[grp["timeframe"] == "contemporary"]
        grp_a = grp[grp["timeframe"] == "archival"]
        acc_c = round((grp_c["predicted"] == grp_c["true_int"]).mean(), 3) if len(grp_c) > 0 else float("nan")
        acc_a = round((grp_a["predicted"] == grp_a["true_int"]).mean(), 3) if len(grp_a) > 0 else float("nan")

        rows.append({
            "outlet": outlet, "bias": bias,
            "region": OUTLET_REGION.get(outlet, "?"),
            "n": len(grp), "n_C": len(grp_c), "n_A": len(grp_a),
            "acc": round((grp["predicted"] == grp["true_int"]).mean(), 3),
            "acc_C": acc_c, "acc_A": acc_a,
            "pct_L": round(dist.get(0, 0), 3),
            "pct_C": round(dist.get(1, 0), 3),
            "pct_R": round(dist.get(2, 0), 3),
        })
    df = pd.DataFrame(rows).sort_values("acc").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def _slice_f1(df_results, col, value):
    """Compute macro F1 + accuracy for a slice of df_results."""
    grp = df_results[df_results[col] == value]
    if len(grp) == 0:
        return {}
    mf1 = f1_score(grp["true_int"], grp["predicted"], average="macro", zero_division=0)
    acc = accuracy_score(grp["true_int"], grp["predicted"])
    return {"n": len(grp), "macro_f1": round(mf1, 4), "accuracy": round(acc, 4)}


# Populate MODELS with model directories
MODELS = discover_models()

if __name__ == "__main__":
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GOLDEN SET EVALUATION — {len(MODELS)} models")
    print(f"{'='*60}")

    # Display names for each model
    for i, (name, path, seq_len) in enumerate(MODELS, 1):
        print(f"  {i:2d}. {get_display_name(name)}  (seq={seq_len})")
    print()

    # load golden set
    golden_set = pd.read_csv(GOLDEN_PATH, quoting=csv.QUOTE_ALL)
    if STRICT_LABELS_ONLY:
        golden_set = golden_set[golden_set["bias"].isin(["left", "center", "right"])].reset_index(drop=True)
    print(f"Golden set (raw): {len(golden_set)} articles")

    # Ensure same text cleaning as training data (URL removal, source metadata stripping)
    golden_set["text"] = golden_set["text"].astype(str)
    golden_set["text"] = golden_set["text"].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    golden_set["text"] = golden_set["text"].apply(DataCleaner._strip_source_metadata)
    golden_set["text"] = golden_set["text"].str.replace(r"\n+", " ", regex=True).str.strip()
    
    # Strip outlet name self-references to prevent leaking source identity
    for outlet in golden_set["outlet"].unique():
        mask = golden_set["outlet"] == outlet
        golden_set.loc[mask, "text"] = golden_set.loc[mask, "text"].str.replace(outlet, "", case=False, regex=False)
    print(f"Golden set (cleaned): {len(golden_set)} articles  "
          f"(L:{sum(golden_set['bias']=='left')} C:{sum(golden_set['bias']=='center')} R:{sum(golden_set['bias']=='right')})")
    print(f"Timeframe:  contemporary={sum(golden_set['timeframe']=='contemporary')}  "
          f"archival={sum(golden_set['timeframe']=='archival')}\n")

    texts = golden_set["text"].tolist()
    biases = golden_set["bias"].tolist()
    true_ints = [BIAS_TO_INT[bias] for bias in biases]

    # Initial collection of checkpoints for cherry-picking
    best_checkpoints = []  # (model_name, best_path, seq_len, best_f1, ckpt_label)
    all_ckpt_histories = {}  # model_name -> [{"label": ..., "f1": ...}, ...]

    if CHERRY_PICK_MODEL:
        print(f"\n{'='*60}")
        print("Scanning checkpoints for best golden F1")
        print(f"{'='*60}\n")

        if SELECT_MODELS:
            print(f"Selecting only specified models: {SELECT_MODELS}\n")
            models_to_scan = [
                model for model in MODELS 
                if any("-".join(model[0].split("-")[:2]) == selection for selection in SELECT_MODELS)
            ]
            print(models_to_scan)
        else:
            models_to_scan = MODELS  
        if not models_to_scan:
            print("No models found")
            exit(0)
            
        for model_name, model_root, seq_len in models_to_scan:
            disp_name = get_display_name(model_name)
            print(f"\n{'─'*60}")
            print(f"  {disp_name}")
            print(f"{'─'*60}")

            # Infer HF repo from model dir name and get assigned tokeniser
            try:
                repo = _infer_repo(model_name, model_root)
                tokenizer = AutoTokenizer.from_pretrained(repo)
            except Exception as e:
                print(f"  [ERROR] {e}")
                continue

            candidates = []
            # Get model weights; assign safetnsors in root as 'best_eval' and label individual checkpoints
            if os.path.exists(os.path.join(model_root, "model.safetensors")):
                candidates.append(("best_eval", model_root))
            for entry in sorted(os.listdir(model_root)):
                entry_path = os.path.join(model_root, entry)
                if entry.startswith("checkpoint-") and os.path.isdir(entry_path):
                    if os.path.exists(os.path.join(entry_path, "model.safetensors")):
                        candidates.append((entry, entry_path))

            # ITerate through all checkpoints (and best_eval) and find best predictor
            best_f1, best_path, best_label = -1, model_root, "best_eval"
            # track f1 for checkpoitn comparison
            ckpt_history = []
            for ckpt_label, ckpt_path in candidates:
                try:
                    model = load_model(ckpt_path, model_name).to(device).eval()
                except Exception as e:
                    print(f"    [ERROR] {ckpt_label}: {e}")
                    continue

                probs = predict_proba(model, tokenizer, texts, seq_len)
                preds = probs.argmax(axis=1)
                mf1 = f1_score(true_ints, preds, average="macro")
                print(f"    {ckpt_label:>20s}  F1={mf1:.4f}")

                # Track checkpoint performance for visualization
                ckpt_history.append({"label": ckpt_label, "f1": mf1})

                if mf1 > best_f1:
                    best_f1, best_path, best_label = mf1, ckpt_path, ckpt_label

                del model
                torch.cuda.empty_cache()

            print(f" Best: {best_label} (F1={best_f1:.4f})")
            best_checkpoints.append((model_name, best_path, seq_len, best_f1, best_label))
            all_ckpt_histories[model_name] = ckpt_history
    else:
        print(f"\n{'='*60}")
        print("Using saved models (cherry-picking disabled)")
        print(f"{'='*60}\n")
        models_to_eval = MODELS
        if SELECT_MODELS:
            print(f"Selecting only specified models: {SELECT_MODELS}\n")
            models_to_eval = [
                model for model in MODELS
                if any("-".join(model[0].split("-")[:2]) == sel for sel in SELECT_MODELS)
            ]
        for model_name, model_root, seq_len in models_to_eval:
            print(f"  {get_display_name(model_name)}")
            best_checkpoints.append((model_name, model_root, seq_len, -1, "best_eval"))

    # Evaluation

    master_rows = []
    comparison_results = {}  # ex_id → (df_res, model_name)

    for model_name, ckpt_path, seq_len, scan_f1, ckpt_label in best_checkpoints:
        disp_name = get_display_name(model_name)
        print(f"\n{'─'*60}")
        print(f"  {disp_name}  [{ckpt_label}]")
        print(f"{'─'*60}")

        t0 = time.time()
        try:
            repo = _infer_repo(model_name, ckpt_path)
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = load_model(ckpt_path, model_name).to(device).eval()
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        probs = predict_proba(model, tokenizer, texts, seq_len)
        preds = probs.argmax(axis=1)

        macro_f1 = f1_score(true_ints, preds, average="macro")
        accuracy = accuracy_score(true_ints, preds)
        pcf1 = f1_score(true_ints, preds, average=None, zero_division=0)
        elapsed = time.time() - t0


        # Build per-article results
        df_res = golden_set[["outlet", "bias", "url", "timeframe"]].copy()
        df_res["predicted"]       = preds
        df_res["predicted_label"] = [LABEL_NAMES[p] for p in preds]
        df_res["true_int"]        = true_ints
        df_res["region"]          = df_res["outlet"].map(OUTLET_REGION)

        safe = model_name.replace(" ", "_").replace("=", "").replace(".", "")
        ts = int(time.time())

        # Per-model figures
        if PLOT_HEATMAP:
            plot_heatmap(df_res, model_name,
                        os.path.join(FIGURES_DIR, f"golden_heatmap_{safe}_{ts}.png"))
        if PLOT_OUTLET:
            plot_outlet_bars(df_res, model_name,
                         os.path.join(FIGURES_DIR, f"golden_outlets_{safe}_{ts}.png"))

        # Stash for side-by-side comparison
        ex_id = model_name.split("-")[0] + "-" + model_name.split("-")[1]
        if ex_id in SELECT_MODELS:
            comparison_results[ex_id] = (df_res.copy(), model_name)

        if PRINT_BREAKDOWN:
            print(f"  Macro F1 : {macro_f1:.4f}   Accuracy: {accuracy:.4f}   Time: {elapsed:.1f}s")
            print(f"  Per-class: L={pcf1[0]:.4f}  C={pcf1[1]:.4f}  R={pcf1[2]:.4f}")
            print(classification_report(true_ints, preds, target_names=LABEL_NAMES, zero_division=0))

            # Per-outlet breakdown
            print("  Per-outlet breakdown:")
            print(outlet_summary(df_res).to_string(index=False))

            # Region breakdown
            print("\n  Per-region:")
            for region in ["UK", "US"]:
                s = _slice_f1(df_res, "region", region)
                print(f"    {region}: n={s['n']}  F1={s['macro_f1']}  Acc={s['accuracy']}")

            # Timeframe breakdown
            print("\n  Per-timeframe:")
            for tf in ["contemporary", "archival"]:
                s = _slice_f1(df_res, "timeframe", tf)
                print(f"    {tf}: n={s['n']}  F1={s['macro_f1']}  Acc={s['accuracy']}")

        # Build master row (wide format)
        row = {
            "model": model_name,
            "display_name": disp_name,
            "checkpoint": ckpt_label,
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "f1_left":   round(pcf1[0], 4),
            "f1_centre": round(pcf1[1], 4),
            "f1_right":  round(pcf1[2], 4),
            "time_sec":  round(elapsed, 1),
        }
        for region in ["UK", "US"]:
            s = _slice_f1(df_res, "region", region)
            if s:
                row[f"{region.lower()}_f1"] = s["macro_f1"]
                row[f"{region.lower()}_acc"] = s["accuracy"]
        for tf in ["contemporary", "archival"]:
            s = _slice_f1(df_res, "timeframe", tf)
            if s:
                row[f"{tf[:4]}_f1"] = s["macro_f1"]
                row[f"{tf[:4]}_acc"] = s["accuracy"]
        master_rows.append(row)

        del model
        torch.cuda.empty_cache()

    # Save master summary CSV
    if SAVE:
        master_df = pd.DataFrame(master_rows)
        print(f"\n{'='*60}")
        print("GOLDEN SET SUMMARY")
        print(f"{'='*60}")
        print(master_df.to_string(index=False))
        master_df.to_csv(METRICS_OUT, index=False)
        print(f"\nSaved → {METRICS_OUT}")

    # Plot model comparison overview figure
    if not CHERRY_PICK_MODEL:
        if len(master_df) > 1:
            output_path = os.path.join(FIGURES_DIR, f"golden_model_comparison_{int(time.time())}.png")
            plot_model_comparison_overview(master_df, output_path,
                                        ckpt_histories=all_ckpt_histories if CHERRY_PICK_MODEL else None)

    # Plot golden checkpoint comparison figure
    if CHERRY_PICK_MODEL and all_ckpt_histories:
        prog_path = os.path.join(FIGURES_DIR, f"golden_checkpoint_progression_{int(time.time())}.png")
        plot_checkpoint_progression(all_ckpt_histories, prog_path)

        # Export all checkpoint F1 scores to CSV
        ckpt_rows = []
        for model_name, history in all_ckpt_histories.items():
            disp = get_display_name(model_name)
            for h in history:
                ckpt_rows.append({
                    "model": model_name,
                    "display_name": disp,
                    "checkpoint": h["label"],
                    "macro_f1": round(h["f1"], 4),
                })
        ckpt_df = pd.DataFrame(ckpt_rows)
        ckpt_csv = os.path.join(os.path.dirname(METRICS_OUT), f"golden_checkpoint_scan_{int(time.time())}.csv")
        ckpt_df.to_csv(ckpt_csv, index=False)
        print(f"Saved checkpoint scan → {ckpt_csv}")

    # Plot comparison figures (intended for best STL vs. MTL)
    if PLOT_COMPARISON and len(comparison_results) == len(SELECT_MODELS) == 2:
        ts = int(time.time())
        plot_heatmap_comparison(comparison_results,
                                os.path.join(FIGURES_DIR, f"golden_heatmap_comparison_{ts}.png"))
        plot_outlet_comparison(comparison_results,
                               os.path.join(FIGURES_DIR, f"golden_outlet_comparison_{ts}.png"))
        print(f"Saved comparison figures for {SELECT_MODELS}")


