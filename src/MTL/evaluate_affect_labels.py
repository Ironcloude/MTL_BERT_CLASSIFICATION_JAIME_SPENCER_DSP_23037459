import pandas as pd
from datasets import load_from_disk
import textwrap

def validate_extremes(dataset_path, num_examples=5):
    print(f"Loading dataset from {dataset_path}...\n")
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Handle both standalone Datasets and DatasetDicts
    if hasattr(dataset, "keys") and "train" in dataset.keys():
        df = dataset["train"].to_pandas()
        print("Detected DatasetDict. Slicing from the 'train' split...")
    else:
        df = dataset.to_pandas()

    # Identify the sentiment column
    sentiment_column = "sentiment_label"
    if sentiment_column not in df.columns:
        print(f"[ERROR] Column '{sentiment_column}' not found. Available columns: {df.columns.tolist()}")
        return

    # Look for both the string representation and the integer just in case
    targets = {
        "STRONGLY NEGATIVE": ["strongly_neg", 0],
        "NEUTRAL": ["neutral", 2],
        "STRONGLY POSITIVE": ["strongly_pos", 4]
    }

    for display_name, possible_values in targets.items():
        # Filter rows matching the strong labels
        strong_labels = df[df[sentiment_column].isin(possible_values)]
        count = len(strong_labels)
        
        print(f"\n{'='*80}")
        print(f" {display_name}  (Total found: {count:,})")
        print(f"{'='*80}")
        
        if count == 0:
            print(f"No examples found. (Check column mappings)")
            continue
            
        # Grab n random examples
        sample = strong_labels.sample(min(num_examples, count))
        
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            text = str(row['text'])
            pol_label = row.get('label', 'Unknown')
            sent_val = row[sentiment_column]
            source = row.get('source', 'Unknown')
            
            display_text = text[:800] + ("..." if len(text) > 800 else "")
            wrapped_text = textwrap.fill(display_text, width=80)
            
            print(f"\n[Example {i}] | Source: {source} | Pol Label: {pol_label} | Sent: {sent_val}")
            print("-" * 80)
            print(wrapped_text)
            print()


if __name__ == "__main__":
    DATASET_PATH = "/home/jaime/DSP/Project/data/processed/combined_multi_task_source"
    validate_extremes(DATASET_PATH, num_examples=5)