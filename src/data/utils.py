import os
import pandas as pd
from datasets import load_from_disk

# Archival Data Paths
#   Allsides Kaggle 2014-2025
#   Webis - 2013 - 2018
#   ABP - 2012 - 2020
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_ABP_PATH    = os.path.join(PROJECT_ROOT, "data/raw/alignment/Article-Bias-Prediction-main/Article-Bias-Prediction-main/data/arrow_news_dataset")
RAW_KAGGLE_PATH = os.path.join(PROJECT_ROOT, "data/raw/alignment/kaggle_AllSides_scraped_10k/kaggle_AllSides_scraped_10k.csv")
RAW_WEBIS_PATH  = os.path.join(PROJECT_ROOT, "data/raw/alignment/corpus-webis-bias-flipper-18/allsides-collection/data_public.csv")

# Label names for ecoding
STANDARD_MAP = {"left": 0, "right": 2, "center": 1}
WEBIS_MAP = {"from the left": 0, "from the center": 1, "from the right": 2}
# Hold-out sources for golden-set candidates
# Matched with str.contains to catch variants (e.g. "New York Post (Opinion)", "BBC Fact Check")
GOLDEN_SET_HOLD_OUT = ["The Guardian", "HuffPost", "BBC", "Newsweek", "Daily Mail", "New York Post"]

# Label names for reporting
LABEL_NAMES = {0: "Left", 1: "Centre", 2: "Right"}

# Dataset loading config
DATASET_CONFIGS = {
    "kaggle": {"path": RAW_KAGGLE_PATH, "map": STANDARD_MAP, "text": "page_text", "label": "bias", "source": "site"},
    "webis": {"path": RAW_WEBIS_PATH, "map": WEBIS_MAP, "text": "original_body", "label": "bias", "source": "source"},
    "ABP": {"path": RAW_ABP_PATH, "map": None, "text": "content", "label": "bias", "source": "source"}}


def print_source_breakdown(df, dataset_name):
    total = len(df)
    print(f"\n{'='*70}")
    print(f"  {dataset_name}  ({total:,} articles)")
    print(f"{'='*70}")
    class_totals = df["label"].value_counts(dropna=False).sort_index()
    avg_words = df["text"].str.split().str.len().mean()
    for lbl, n in class_totals.items():
        name = LABEL_NAMES.get(lbl, f"UNKNOWN({lbl})")
        print(f"  {name:<8}: {n:>5,}  ({n/total:.1%})")
    print(f"  Max word article: {df['text'].str.split().str.len().max():.1f}")
    print(f"  Min word article: {df['text'].str.split().str.len().min():.1f}")
    print(f"  Average words per article: {avg_words:.1f}")
    print(f"  Average est tokens per article: {(avg_words*1.25):.1f}")

def load_alignment_dataset(raw_datasets = DATASET_CONFIGS):
    cleaned_datasets = []
    print("Loading datasets...")
    for name, raw_dataset in raw_datasets.items():
        # Load datsets into standard Dataframe
        if raw_dataset['path'].endswith(".csv"):  # CSV
            df = pd.read_csv(raw_dataset['path'], on_bad_lines="skip")
        else:    # Arrow
            ds = load_from_disk(raw_dataset['path'])
            df = ds.to_pandas()

        # Set target columns
        df = df[[raw_dataset['text'], raw_dataset['label'], raw_dataset['source']]].copy()
        df.columns = ["text", "label", "source"]
        df = df.dropna(subset=["text", "label"])

        # Apply Label encoding mapping
        if raw_dataset['map'] is not None:
            df["label"] = df["label"].str.lower().str.strip().map(raw_dataset['map'])
        # print_source_breakdown(df, f"BEFORE {name}")

        # Clean remaining NAs
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df["source"] = df["source"].fillna("Unknown").str.strip()

        cleaned_datasets.append(df[["text", "label", "source"]])

        print_source_breakdown(df, name)

    return pd.concat(cleaned_datasets, ignore_index=True)

def print_archival_timeframes(df):
    """
    Prints a year-by-year breakdown of article volume.
    Includes an ASCII bar chart for quick visual inspection of 'data clumps'.
    """
    # Ensure date is datetime and filter outliers
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    mask = (df["date"] >= pd.Timestamp("2000-01-01")) & (df["date"] <= pd.Timestamp("2026-12-31"))
    clean_df = df[mask].copy()
    
    if clean_df.empty:
        print(f"\n[!] No valid dates in range for distribution.")
        return

    # Group by year
    dist = clean_df["date"].dt.year.value_counts().sort_index()
    max_val = dist.max()
    total = dist.sum()

    print(f"\n{'='*70}")
    print(f"  TEMPORAL DISTRIBUTION: (Cleaned)")
    print(f"{'='*70}")
    print(f"  {'Year':<6} | {'Count':>6} | {'%':>5} |  Visualization")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*5}-+--{'-'*30}")

    for year, count in dist.items():
        percentage = (count / total) * 100
        # Create a simple ASCII bar (max 30 chars)
        bar_len = int((count / max_val) * 30)
        bar = "█" * bar_len
        print(f"  {int(year):<6} | {count:>6,} | {percentage:>4.1f}% |  {bar}")
    
    print(f"{'='*70}")
    print(f"  Total Cleaned Articles: {total:,}")
    print(f"  Dropped (Outliers/NaT): {len(df) - total:,}")

def print_golden_candidates(df):
    """
    For each class, list sources sorted by count descending.
    Highlights the sweet spot: sources large enough to scrape 20+ articles
    from their live site, but small enough that holding them out won't
    destabilise training.
    """
    print(f"\n{'='*70}")
    print("  GOLDEN SET CANDIDATE ANALYSIS")
    print(f"  (sources with ≥20 articles per class, sorted by count)")
    print(f"{'='*70}")

    for lbl, name in LABEL_NAMES.items():
        subset = df[df["label"] == lbl]
        class_total = len(subset)
        counts = (
            subset["source"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["source", "n"]
        counts["pct_of_class"] = (counts["n"] / class_total * 100).round(1)
        counts = counts[counts["n"] >= 20]   # must have enough to scrape

        print(f"\n  ── {name} (class total = {class_total:,}) ──")
        print(f"  {'Source':<40}  {'N':>5}  {'% of class':>10}")
        print(f"  {'-'*40}  {'-'*5}  {'-'*10}")
        for _, row in counts.iterrows():
            flag = "  <= consider" if 0.5 <= row["pct_of_class"] <= 8 else ""
            print(f"  {row['source']:<40}  {row['n']:>5,}  {row['pct_of_class']:>9.1f}%{flag}")

if __name__ == "__main__":
    ds = load_from_disk(DATASET_CONFIGS['ABP']['path'])
    df = ds.to_pandas()
    print_archival_timeframes(df)