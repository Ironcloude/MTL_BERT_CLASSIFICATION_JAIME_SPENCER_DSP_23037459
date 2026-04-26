"""
Designed for identification of apt case-studies for XAI analysis.

Per-article golden set predictions for two models (STL vs MTL).
Outputs a CSV with columns: idx, outlet, bias, timeframe, true_label,
pred_stl, pred_mtl, conf_stl, conf_mtl, agree, text_preview.
"""

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import os, sys, torch
import numpy as np
import pandas as pd
from pathlib import Path
from golden_set.golden_set_eval import load_model, _infer_repo, BIAS_TO_INT, LABEL_NAMES
from transformers import AutoTokenizer
from data_cleaning.data_cleaner import DataCleaner

# Config
GOLDEN_PATH = project_root / "data" / "golden_set" / "golden_articles.csv"
OUT_CSV     = project_root / "results" / "metrics" / "golden_per_article_eval.csv"

MODELS = {
    "EX-2": {
        "folder": "EX-2-DeB-st-lr-2.5e-05-100pct-512-20260407-170059",
        "checkpoint": "checkpoint-1372",
        "seq_len": 512,
    },
    "EX-12": {
        "folder": "EX-12-Mod-mt-λ0.25-lr-5e-05-100pct-1024-20260408-123841",
        "checkpoint": "checkpoint-1176",
        "seq_len": 1024,
    },
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Load Golden Set
    df = pd.read_csv(GOLDEN_PATH)

    # Light text cleaning (URLs, source metadata, newlines) without dropping by label/length
    df["clean_text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    df["clean_text"] = df["clean_text"].apply(DataCleaner._strip_source_metadata)
    df["clean_text"] = df["clean_text"].str.replace(r"\n+", " ", regex=True).str.strip()

    df["true_label"] = df["bias"].map(BIAS_TO_INT)
    # Drop rows without a strict mapping (lean left / lean right)
    df = df.dropna(subset=["true_label"]).reset_index(drop=True)
    df["true_label"] = df["true_label"].astype(int)

    texts = df["clean_text"].tolist()

    # prediction per model
    for experiment_tag, config in MODELS.items():
        ckpt_path = str(project_root / "models" / "final" / config["folder"] / config["checkpoint"])
        folder_name = config["folder"]

        print(f"Loading {experiment_tag} from {config['checkpoint']}...")
        model = load_model(ckpt_path, folder_name)
        model.to(device).eval()

        repo = _infer_repo(folder_name, ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(repo)

        all_probs = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=config["seq_len"], padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

        probs = np.vstack(all_probs)
        df[f"pred_{experiment_tag}"] = probs.argmax(axis=1)
        df[f"conf_{experiment_tag}"] = probs.max(axis=1)
        df[f"pred_{experiment_tag}_label"] = df[f"pred_{experiment_tag}"].map(dict(enumerate(LABEL_NAMES)))
        print(f"  {experiment_tag} done — accuracy: {(df[f'pred_{experiment_tag}'] == df['true_label']).mean():.3f}")

    df["agree"] = df["pred_EX-2"] == df["pred_EX-12"]
    df["text_preview"] = df["clean_text"].str[:120]
    df["true_label_name"] = df["true_label"].map(dict(enumerate(LABEL_NAMES)))

    out = df[[
        "outlet", "bias", "timeframe", "true_label_name",
        "pred_EX-2_label", "conf_EX-2", "pred_EX-12_label", "conf_EX-12",
        "agree", "text_preview", "clean_text",
    ]].copy()
    out.columns = [
        "outlet", "bias", "timeframe", "true",
        "pred_stl", "conf_stl", "pred_mtl", "conf_mtl",
        "agree", "text_preview", "full_text",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=True, index_label="idx")
    print(f"\nSaved {len(out)} rows → {OUT_CSV}")

    print("\n" + "=" * 70)
    print("CASE STUDY CANDIDATES")
    print("=" * 70)

    # Case1 : Centre where EX-2 wrong, EX-12 correct
    case1 = out[
        (out["true"] == "Centre") &
        (out["pred_stl"] != "Centre") &
        (out["pred_mtl"] == "Centre")
    ].sort_values("conf_mtl", ascending=False)
    print(f"\n── Case 1: Centre — EX-2 wrong, EX-12 correct ({len(case1)} found) ──")
    for idx, r in case1.head(5).iterrows():
        print(f"  idx={idx} [{r['outlet']}] [{r['timeframe']}] STL→{r['pred_stl']}({r['conf_stl']:.0%}) MTL→{r['pred_mtl']}({r['conf_mtl']:.0%})")
        print(f"    {r['text_preview']}")

    # Case 2: Left where EX-12 says Right, EX-2 correct
    case2 = out[
        (out["true"] == "Left") &
        (out["pred_mtl"] == "Right") &
        (out["pred_stl"] == "Left")
    ].sort_values("conf_stl", ascending=False)
    print(f"\n── Case 2: Left — EX-12→Right, EX-2 correct ({len(case2)} found) ──")
    for idx, r in case2.head(5).iterrows():
        print(f"  idx={idx} [{r['outlet']}] [{r['timeframe']}] STL→{r['pred_stl']}({r['conf_stl']:.0%}) MTL→{r['pred_mtl']}({r['conf_mtl']:.0%})")
        print(f"    {r['text_preview']}")

    # any disagreements sorted by confidence gap
    disagree = out[~out["agree"]].copy()
    disagree["conf_gap"] = (disagree["conf_stl"] - disagree["conf_mtl"]).abs()
    disagree = disagree.sort_values("conf_gap", ascending=False)
    print(f"\n── All disagreements ({len(disagree)} total) — top 10 by confidence gap ──")
    for idx, r in disagree.head(10).iterrows():
        print(f"  idx={idx} [{r['outlet']}] [{r['timeframe']}] true={r['true']} STL→{r['pred_stl']}({r['conf_stl']:.0%}) MTL→{r['pred_mtl']}({r['conf_mtl']:.0%})")
        print(f"    {r['text_preview']}")
