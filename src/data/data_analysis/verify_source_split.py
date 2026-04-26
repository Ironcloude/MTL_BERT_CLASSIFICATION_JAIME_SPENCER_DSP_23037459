"""
Verify source-stratified split:
1. Normalise source names to collapse variants
2. Run GroupShuffleSplit
3. Check for leakage (no normalised source in both train and eval)
4. Print class balance and source lists for both splits

Informed 
"""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "figures"))
style_path = Path(__file__).parent.parent.parent / "figures" / "style.mplstyle"
plt.style.use(style_path)

import sys, os, re
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from theme import Colours, Fonts

SEED = 3189
SAVE = True
GOLDEN_SET_HOLD_OUT = ["The Guardian", "HuffPost", "BBC", "Newsweek", "Daily Mail", "New York Post"]

dataset_path = "/home/jaime/DSP/Project/data/processed/composite_dataset_keep_source"
dataset = load_from_disk(dataset_path)
df = dataset.to_pandas()
# golden_mask = df["source"].apply(lambda s: any(g.lower() in s.lower() for g in GOLDEN_SET_HOLD_OUT))
# print(f"Removing {golden_mask.sum()} articles matching golden set outlets")
# df = df[~golden_mask].reset_index(drop=True)
print(f"\nTotal articles (excl golden holdout): {len(df)}")
print(f"Total raw sources: {df['source'].nunique()}")

# Normalise source names
SOURCE_ALIASES = {
    # AP
    "AP Fact Check": "Associated Press",
    # CNN
    "CNN (Web News)": "CNN",
    "CNN (Opinion)": "CNN",
    "CNN - Editorial": "CNN",
    "CNN Fact Check": "CNN",
    # NBC (not CNBC/MSNBC — those are separate outlets)
    "NBC (Web News)": "NBC News",
    "NBC News (Online)": "NBC News",
    "NBC Today Show": "NBC News",
    "NBCNews.com": "NBC News",
    "NBC 5 Chicago": "NBC News",
    # NPR
    "NPR News": "NPR",
    "NPR Online News": "NPR",
    "NPR Editorial": "NPR",
    # Newsmax
    "Newsmax (News)": "Newsmax",
    "Newsmax (Opinion)": "Newsmax",
    "Newsmax - News": "Newsmax",
    "Newsmax - Opinion": "Newsmax",
    # Guest writers (same aggregated source)
    "Guest Writer - Left": "Guest Writer",
    "Guest Writer - Right": "Guest Writer",
    "Guest Writer - Center": "Guest Writer",
    # Fox News (catch variants without parens/dashes)
    "Fox Online News": "Fox News",
    "Fox News Opinion": "Fox News",
    # Breitbart
    "Breitbart News": "Breitbart",
    "Breitbart Fact Check": "Breitbart",
    # BuzzFeed
    "Buzzfeed": "BuzzFeed News",
    # CBS
    "CBS SFBayArea": "CBS News",
    # CNS News
    "CNSNews.com": "CNS News",
    # Des Moines Register (spelling variant)
    "DesMoines Register": "Des Moines Register",
    # Western Journal ("The" prefix variant)
    "The Western Journal": "Western Journal",
    # Boston Herald
    "Boston Herald Editorial": "Boston Herald",
    # Michelle Malkin
    "MichelleMalkin.com": "Michelle Malkin",
    # American Spectator ("The" prefix variant)
    "The American Spectator": "American Spectator",
    # Others
    "Reason Foundation": "Reason",
    "RedState": "Red State",
    "Voice of America (VOA)": "Voice of America",
}

def normalise_source(s):
    """Collapse outlet variants to a canonical parent name."""
    # Explicit aliases first
    if s in SOURCE_ALIASES:
        return SOURCE_ALIASES[s]
    s = re.sub(r'\s*\(.*?\)', '', s)                        # (Opinion), (Online), (blog)
    s = re.sub(r'\s*-\s*(News|Opinion|Editorial|Blog).*', '', s)  # - News, - Opinion
    s = re.sub(r'\s+(Digital|Online|Latino)$', '', s)        # Fox News Digital
    s = re.sub(r'\s+Fact Check$', '', s)                     # CNN Fact Check
    s = re.sub(r'\s+Editorial Board$', '', s)                # NYT Editorial Board
    return s.strip()


if __name__ == "__main__":
    df["source_normalised"] = df["source"].apply(normalise_source)
    print(f"Normalised sources: {df['source_normalised'].nunique()}")

    # Show what got merged
    print(f"\n{'='*60}")
    print("SOURCE NORMALISATION MAPPING (only changed names)")
    print(f"{'='*60}")
    changed = df[df["source"] != df["source_normalised"]][["source", "source_normalised"]].drop_duplicates()
    for _, row in changed.sort_values("source_normalised").iterrows():
        print(f"  {row['source']:<50} → {row['source_normalised']}")


    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X=range(len(df)), groups=df["source_normalised"]))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Verify no source leakage
    train_norm = set(train_df["source_normalised"])
    test_norm = set(test_df["source_normalised"])
    overlap_norm = train_norm & test_norm

    # Also check raw source names
    train_raw = set(train_df["source"])
    test_raw = set(test_df["source"])
    overlap_raw = train_raw & test_raw

    print(f"\n{'='*60}")
    print("LEAKAGE CHECK")
    print(f"{'='*60}")
    print(f"Normalised source overlap: {len(overlap_norm)} (MUST be 0)")
    if overlap_norm:
        print(f"  LEAKED: {overlap_norm}")
    print(f"Raw source name overlap:   {len(overlap_raw)}")
    if overlap_raw:
        print(f"  Raw overlaps (may be OK if normalised names differ): {overlap_raw}")

    # Show class balance
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} articles ({len(train_df)/len(df)*100:.1f}%) from {train_df['source_normalised'].nunique()} sources")
    print(f"Eval:  {len(test_df)} articles ({len(test_df)/len(df)*100:.1f}%) from {test_df['source_normalised'].nunique()} sources")

    label_names = {0: "Left", 1: "Centre", 2: "Right"}
    print(f"\nTrain class balance:")
    for lbl, name in label_names.items():
        n = sum(train_df["label"] == lbl)
        print(f"  {name}: {n} ({n/len(train_df)*100:.1f}%)")

    print(f"\nEval class balance:")
    for lbl, name in label_names.items():
        n = sum(test_df["label"] == lbl)
        print(f"  {name}: {n} ({n/len(test_df)*100:.1f}%)")

    #Show eval sources
    print(f"\n{'='*60}")
    print("EVAL SOURCES (sorted by size)")
    print(f"{'='*60}")
    eval_cross = pd.crosstab(test_df["source"], test_df["label"])
    eval_cross.columns = ["Left", "Centre", "Right"]
    eval_cross["total"] = eval_cross.sum(axis=1)
    eval_cross = eval_cross.sort_values("total", ascending=False)
    print(eval_cross.to_string())

    # Check for variant leakage
    # For each normalised source in eval, check all its raw variants are also in eval (not train)
    print(f"\n{'='*60}")
    print("VARIANT LEAK CHECK")
    print(f"{'='*60}")
    leaks = 0
    for norm_src in test_norm:
        all_variants = df[df["source_normalised"] == norm_src]["source"].unique()
        for v in all_variants:
            if v in train_raw:
                print(f"  LEAK: '{v}' (normalised='{norm_src}') is in TRAIN but normalised group is in EVAL")
                leaks += 1
    if leaks == 0:
        print("  No variant leaks detected. All clear.")

    # Export full outlet list for manual evaluation
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "source_split_verification.txt")
    if SAVE:
        with open(out_path, "w") as f:
            f.write("SOURCE-STRATIFIED SPLIT VERIFICATION\n")
            f.write(f"{'='*80}\n\n")

            # Normalisation mapping
            f.write("NORMALISATION MAPPING (only changed names)\n")
            f.write(f"{'-'*80}\n")
            for _, row in changed.sort_values("source_normalised").iterrows():
                f.write(f"  {row['source']:<50} → {row['source_normalised']}\n")

            # Summary
            f.write(f"\n\nSPLIT SUMMARY\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Train: {len(train_df)} articles ({len(train_df)/len(df)*100:.1f}%) from {train_df['source_normalised'].nunique()} normalised sources\n")
            f.write(f"Eval:  {len(test_df)} articles ({len(test_df)/len(df)*100:.1f}%) from {test_df['source_normalised'].nunique()} normalised sources\n")
            f.write(f"\nNormalised source overlap: {len(overlap_norm)} (MUST be 0)\n")
            f.write(f"Raw source name overlap:   {len(overlap_raw)}\n")
            f.write(f"Variant leaks: {leaks}\n")

            # Train class balance
            f.write(f"\nTrain class balance:\n")
            for lbl, name in label_names.items():
                n = sum(train_df["label"] == lbl)
                f.write(f"  {name}: {n} ({n/len(train_df)*100:.1f}%)\n")
            f.write(f"\nEval class balance:\n")
            for lbl, name in label_names.items():
                n = sum(test_df["label"] == lbl)
                f.write(f"  {name}: {n} ({n/len(test_df)*100:.1f}%)\n")

            # Full train sources
            train_cross = pd.crosstab(train_df["source"], train_df["label"])
            train_cross.columns = ["Left", "Centre", "Right"]
            train_cross["total"] = train_cross.sum(axis=1)
            train_cross = train_cross.sort_values("total", ascending=False)

            f.write(f"\n\n{'='*80}\n")
            f.write(f"TRAIN SOURCES ({len(train_cross)} raw sources)\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Source':<50} {'Left':>6} {'Centre':>7} {'Right':>6} {'Total':>6}\n")
            f.write(f"{'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*6}\n")
            for src, row in train_cross.iterrows():
                f.write(f"{src:<50}(train) {int(row['Left']):>6} {int(row['Centre']):>7} {int(row['Right']):>6} {int(row['total']):>6}\n")

            # Full eval sources
            f.write(f"\n\n{'='*80}\n")
            f.write(f"EVAL SOURCES ({len(eval_cross)} raw sources)\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Source':<50} {'Left':>6} {'Centre':>7} {'Right':>6} {'Total':>6}\n")
            f.write(f"{'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*6}\n")
            for src, row in eval_cross.iterrows():
                f.write(f"{src:<50} {int(row['Left']):>6} {int(row['Centre']):>7} {int(row['Right']):>6} {int(row['total']):>6}\n")

        print(f"\nFull verification exported to: {out_path}")

    # Generate split diagram

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5, 3.8))

    # Class distribution comparison
    classes = ["Left", "Centre", "Right"]
    train_pcts = [sum(train_df["label"] == c) / len(train_df) * 100 for c in range(3)]
    eval_pcts = [sum(test_df["label"] == c) / len(test_df) * 100 for c in range(3)]
    overall_pcts = [sum(df["label"] == c) / len(df) * 100 for c in range(3)]

    x = np.arange(len(classes))
    w = 0.25
    ax1.bar(x,     train_pcts,   w, label=f"Train (n={len(train_df):,})", color=Colours.BLUE)
    ax1.bar(x + w, eval_pcts,    w, label=f"Eval (n={len(test_df):,})",  color=Colours.RED)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Class Distribution by Split")

    ax1.set_ylim(0, max(max(train_pcts), max(eval_pcts)) * 1.45)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9, edgecolor="none")

    for i in range(3):
        ax1.text(x[i],     train_pcts[i] + 0.5,   f"{train_pcts[i]:.1f}",   **Fonts.DELTA)
        ax1.text(x[i] + w, eval_pcts[i] + 0.5,    f"{eval_pcts[i]:.1f}",    **Fonts.DELTA)

    # Split size 
    split_data = pd.DataFrame({
        "Split": ["Train", "Eval"],
        "Articles": [len(train_df), len(test_df)],
        "Pct": [len(train_df)/len(df)*100, len(test_df)/len(df)*100],
    })
    bars = ax2.bar(split_data["Split"], split_data["Articles"],
                color=[Colours.BLUE, Colours.RED])
    for bar, pct, n in zip(bars, split_data["Pct"], split_data["Articles"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(split_data["Articles"]) * 0.05),
                f"{n:,}\n({pct:.1f}%)", **Fonts.DELTA)
    ax2.set_ylabel("Number of Articles")
    ax2.set_title("Train / Eval Split Size")
    ax2.set_ylim(0, max(split_data["Articles"]) * 1.30)

    # Number of sources

    n_train_src = train_df["source_normalised"].nunique()
    n_eval_src = test_df["source_normalised"].nunique()
    n_total_src = df["source_normalised"].nunique()
    src_data = pd.DataFrame({
        "Split": ["Train", "Eval"],
        "Sources": [n_train_src, n_eval_src],
        "Pct": [n_train_src/n_total_src*100, n_eval_src/n_total_src*100],
    })
    bars = ax3.bar(src_data["Split"], src_data["Sources"],
                color=[Colours.BLUE, Colours.RED])
    for bar, n, pct in zip(bars, src_data["Sources"], src_data["Pct"]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(src_data["Sources"]) * 0.05),
                f"{n}\n({pct:.1f}%)", **Fonts.DELTA)
    ax3.set_ylabel("Number of Outlets")
    ax3.set_title("Normalised Source Count")
    ax3.set_ylim(0, max(src_data["Sources"]) * 1.30)

    fig.suptitle(f"Source-Stratified GroupShuffleSplit (seed={SEED}, test_size=0.2)",
                fontweight="bold")
    plt.tight_layout()
    plt.show()
    if SAVE:
        fig_path = os.path.join(out_dir, "source_split_verification.png")
        fig.savefig(fig_path)
        plt.close()
        print(f"Figure saved to: {fig_path}")