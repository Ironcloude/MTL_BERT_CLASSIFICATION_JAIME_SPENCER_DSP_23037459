"""
Batch IG attribution - generates heatmaps for every row in golden_per_article_eval.csv.
Runs STL only. Output: results/figures/ig_batch/idx_<N>_<true>_<pred>.png

Usage:
  python batch_ig.py                # all rows
  python batch_ig.py --start 10     # from row 10 onward (resume after interruption)
"""

import sys, argparse
import pandas as pd
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.xai.IG import explain_bias

STL_PATH = str(root_path / "models/final/EX-2-DeB-st-lr-2.5e-05-100pct-512-20260407-170059/checkpoint-1372")
CSV_PATH = root_path / "results" / "metrics" / "golden_per_article_eval.csv"
OUT_DIR  = root_path / "results" / "figures" / "ig_batch"

LABEL_MAP = {"Left": 0, "Centre": 1, "Right": 2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start from this row offset (for resuming)")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, index_col="idx")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df)
    print(f"Generating STL IG heatmaps for {total} articles → {OUT_DIR}")
    print(f"Starting from row offset {args.start}\n")

    for i, (idx, row) in enumerate(df.iterrows()):
        if i < args.start:
            continue

        true_label = LABEL_MAP[row["true"]]
        true_tag = row["true"].lower()
        pred_tag = row["pred_stl"].lower()
        prefix = f"idx_{idx}_{true_tag}_{pred_tag}"

        # Skip if already generated
        if (OUT_DIR / f"{prefix}.png").exists():
            print(f"[{i+1}/{total}] idx={idx} — already exists, skipping")
            continue

        print(f"[{i+1}/{total}] idx={idx} | {row['outlet']} | true={row['true']} | pred_stl={row['pred_stl']}")
        try:
            explain_bias(
                row["full_text"], STL_PATH,
                is_mtl=False, true_label=true_label,
                out_dir=str(OUT_DIR), file_prefix=prefix,
                clean_text=False,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone. Figures saved to {OUT_DIR}")
