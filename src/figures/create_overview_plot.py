import sys
import pandas as pd
sys.path.insert(0, '/home/jaime/DSP/Project/src/golden_set')
from golden_set_eval import plot_checkpoint_progression, plot_model_comparison_overview
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "metrics",
                        "best_golden_checkpoint_scan_1775603202.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures", "golden_set")
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

all_ckpt_histories = {}

for _, row in df.iterrows():
    model_id = row['model']
    
    if model_id not in all_ckpt_histories:
        all_ckpt_histories[model_id] = []
        
    all_ckpt_histories[model_id].append({
        "label": row['checkpoint'],
        "f1": row['macro_f1']
    })


# plot_checkpoint_progression(all_ckpt_histories, "checkpoint_variance_chart.png")
master_df = df[df["checkpoint"] == "best_eval"].copy()
plot_model_comparison_overview(master_df,"golden_set_overview.png")