"""
Build Composite Training Dataset
---------------------------------
Loads the three source datasets (ABP, Kaggle AllSides, Webis Bias Flipper) into 
a combined Dataframe with standardised format. Saves to disk for training.

Also pprints a per-source breakdown to help identify candidates for the
held-out golden evaluation set (sources that are recognisable, have plenty
of articles, but do not dominate any one class).

"""

import os
from datasets import Dataset
from data_cleaning.data_cleaner import DataCleaner
from utils import GOLDEN_SET_HOLD_OUT, load_alignment_dataset, print_golden_candidates, print_source_breakdown

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(project_root, "data/processed/composite_dataset_keep_source")
SAVE = True
KEEP_SOURCE = True
PRINT_CANDIDATES = False

if __name__ == "__main__":
    combined = load_alignment_dataset()
    if combined is None:
        print("Error loading datasets. Exiting.")
        exit(1)

    # Display golden set candidates 
    if PRINT_CANDIDATES:
        print_golden_candidates(combined)
    print_source_breakdown(combined, "Combined Dataset (base)")
    # REMOVE HOLD_OUT sources for golden set (str.contains to catch variants like "BBC Fact Check", "New York Post (Opinion)")
    golden_mask = combined["source"].apply(lambda s: any(g.lower() in s.lower() for g in GOLDEN_SET_HOLD_OUT))
    print(f"Removing {golden_mask.sum()} articles matching golden set outlets")
    combined = combined[~golden_mask].reset_index(drop=True)
    # Clean text and filter out unrealistically short, long or empty articles
    print_source_breakdown(combined, "Combined Dataset (removed golden set candidates)")
    combined = DataCleaner.clean_data(combined)
    print_source_breakdown(combined, "Combined Dataset (after cleaning)")

    if not DataCleaner.validate_data(combined):
        print("Error validating datasets. Exiting.")
        exit(1)
        


    if SAVE:
        os.makedirs(OUT_DIR, exist_ok=True)
        if KEEP_SOURCE:
            ds = Dataset.from_pandas(combined[["text", "label", "source"]].reset_index(drop=True))
        else:
            ds = Dataset.from_pandas(combined[["text", "label"]].reset_index(drop=True))
        ds.save_to_disk(OUT_DIR)
        print(f"\n\nSaved combined dataset ({len(combined):,} articles) => {OUT_DIR}")
