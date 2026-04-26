"""
Brute-force search for GroupShuffleSplit seed with best class balance.

Scores each seed by how close the eval split's class proportions are to the
overall dataset proportions (L1 distance). Prints top candidates.
"""

import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd
from datasets import load_from_disk
from data_analysis.verify_source_split import SOURCE_ALIASES

SEED_RANGE = range(0, 10_000)
TEST_SIZE = 0.2
TOP_K = 20  # how many best seeds to print
DATASET_PATH = "/home/jaime/DSP/Project/data/processed/combined_multi_task_source"
GOLDEN_SET_HOLD_OUT = ["The Guardian", "HuffPost", "BBC", "Newsweek", "Daily Mail", "New York Post"]

# Scoring heuristic
CLASS_IMBALANCE_WEIGHT = 0.6
SPLIT_DEVIATION_WEIGHT = 0.25
SOURCE_DIVERSITY_WEIGHT = 0.15

def normalise_source(source):
    if source in SOURCE_ALIASES:
        return SOURCE_ALIASES[source]
    source = re.sub(r'\s*\(.*?\)', '', source)
    source = re.sub(r'\s*-\s*(News|Opinion|Editorial|Blog).*', '', source)
    source = re.sub(r'\s+(Digital|Online|Latino)$', '', source)
    source = re.sub(r'\s+Fact Check$', '', source)
    source = re.sub(r'\s+Editorial Board$', '', source)
    return source.strip()

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    df = dataset.to_pandas()
    golden_mask = df["source"].apply(lambda source: any(golden.lower() in source.lower() for golden in GOLDEN_SET_HOLD_OUT))
    df = df[~golden_mask].reset_index(drop=True)
    df["source_norm"] = df["source"].apply(normalise_source)
    labels = df["label"].values
    groups = df["source_norm"].values
    n = len(df)
    X = np.arange(n)

    # Overall class proportions (target)
    overall_dist = np.array([
        (labels == 0).sum() / n,  # Left
        (labels == 1).sum() / n,  # Centre
        (labels == 2).sum() / n,  # Right
    ])
    total_sources = df["source_norm"].nunique()
    print(f"\nDataset: {n:,} articles, {total_sources} normalised sources")
    print(f"Overall distribution: Left={overall_dist[0]:.3f}  Centre={overall_dist[1]:.3f}  Right={overall_dist[2]:.3f}")
    print(f"\nSearching {len(SEED_RANGE):,} seeds...")
    print(f"Scoring: {CLASS_IMBALANCE_WEIGHT}*class_imbalance + {SPLIT_DEVIATION_WEIGHT}*split_deviation + {SOURCE_DIVERSITY_WEIGHT}*(1-source_diversity)")

    # Search
    results = []
    for seed in SEED_RANGE:
        group_split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(group_split.split(X, groups=groups))

        test_labels = labels[test_idx]
        n_test = len(test_idx)
        eval_dist = np.array([
            (test_labels == 0).sum() / n_test,
            (test_labels == 1).sum() / n_test,
            (test_labels == 2).sum() / n_test,
        ])

        # Component 1: Class balance - L1 distance from overall distribution
        class_imbalance = np.abs(eval_dist - overall_dist).sum()

        # Component 2: Split ratio - deviation from target test_size
        actual_test_pct = n_test / n
        split_deviation = abs(actual_test_pct - TEST_SIZE)

        # Component 3: Eval source diversity - fraction of total normalised sources in eval
        n_eval_sources = len(set(groups[test_idx]))
        source_fraction = n_eval_sources / total_sources

        # Composite score (lower = better)
        # Class balance is most important, split ratio moderate, source diversity bonus
        score = (CLASS_IMBALANCE_WEIGHT * class_imbalance) + (SPLIT_DEVIATION_WEIGHT* split_deviation) + (SOURCE_DIVERSITY_WEIGHT* (1.0 - source_fraction))

        # Also compute train distribution for reporting
        train_labels = labels[train_idx]
        n_train = len(train_idx)
        train_dist = np.array([
            (train_labels == 0).sum() / n_train,
            (train_labels == 1).sum() / n_train,
            (train_labels == 2).sum() / n_train,
        ])

        results.append({
            "seed": seed,
            "score": score,
            "class_imbalance": class_imbalance,
            "split_dev": split_deviation,
            "n_train": n_train,
            "n_eval": n_test,
            "eval_pct": actual_test_pct * 100,
            "n_eval_sources": n_eval_sources,
            "eval_left": eval_dist[0],
            "eval_centre": eval_dist[1],
            "eval_right": eval_dist[2],
            "train_left": train_dist[0],
            "train_centre": train_dist[1],
            "train_right": train_dist[2],
        })

        if seed % 1000 == 0 and seed > 0:
            print(f"  ...checked {seed:,} seeds")

    # Results
    results_df = pd.DataFrame(results).sort_values("score")

    print(f"\n{'='*110}")
    print(f"TOP {TOP_K} SEEDS (composite score: class balance + split ratio + source diversity)")
    print(f"{'='*110}")
    print(f"{'Seed':>6}  {'Score':>6}  {'ClsImb':>6}  {'SplDev':>6}  "
        f"{'Eval%':>5}  {'#ESrc':>5}  "
        f"{'EvalL':>6}  {'EvalC':>6}  {'EvalR':>6}  "
        f"{'TrainL':>6}  {'TrainC':>6}  {'TrainR':>6}")
    print("-" * 110)
    for _, row in results_df.head(TOP_K).iterrows():
        print(f"{int(row['seed']):>6}  {row['score']:>6.4f}  {row['class_imbalance']:>6.3f}  {row['split_dev']:>6.3f}  "
            f"{row['eval_pct']:>5.1f}  {int(row['n_eval_sources']):>5}  "
            f"{row['eval_left']:>6.1%}  {row['eval_centre']:>6.1%}  {row['eval_right']:>6.1%}  "
            f"{row['train_left']:>6.1%}  {row['train_centre']:>6.1%}  {row['train_right']:>6.1%}")

    print(f"\nTarget: Left={overall_dist[0]:.1%}  Centre={overall_dist[1]:.1%}  Right={overall_dist[2]:.1%}  |  test_size={TEST_SIZE}  |  total_sources={total_sources}")

    # Save full results
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "metrics", "seed_search_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"Full results saved to: {out_path}")
