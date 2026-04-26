import sys, random
sys.path.insert(0, 'src/data')
from utils import load_alignment_dataset, GOLDEN_SET_HOLD_OUT
import pandas as pd
import numpy as np

df = load_alignment_dataset()
df = df[~df['source'].isin(GOLDEN_SET_HOLD_OUT)].reset_index(drop=True)
source_distribution = df['source'].value_counts()

print(f"Total available outlets: {len(source_distribution)}")
print("\nSample volume per outlet:")
print(source_distribution)
TARGET_PCT = 0.15  # aim for ~15% holdout
MIN_ARTICLES = 50  # don't hold out tiny sources (noisy eval)

# For each source, determine its dominant class (>90% of articles)
cross = pd.crosstab(df['source'], df['label'])
cross.columns = ['Left', 'Centre', 'Right']
cross['total'] = cross.sum(axis=1)
cross['dominant'] = cross[['Left','Centre','Right']].idxmax(axis=1)
cross['dominant_pct'] = cross[['Left','Centre','Right']].max(axis=1) / cross['total']

# Only consider single-class sources with enough articles
candidates = cross[(cross['dominant_pct'] > 0.9) & (cross['total'] >= MIN_ARTICLES)].copy()

print(f'Total articles (excl golden): {len(df)}')
print(f'Target holdout: ~{int(len(df)*TARGET_PCT)} articles ({TARGET_PCT:.0%})')
print(f'Candidate sources (single-class, >={MIN_ARTICLES} articles): {len(candidates)}')
print()

# Per class: sort candidates by size, greedily add until we hit target
class_totals = df['label'].value_counts().to_dict()
class_names = {0: 'Left', 1: 'Centre', 2: 'Right'}

np.random.seed(42)
holdout_sources = []

for label, name in class_names.items():
    class_candidates = candidates[candidates['dominant'] == name].copy()
    class_target = int(class_totals[label] * TARGET_PCT)
    
    # Shuffle then sort by size (medium-first: not too big, not too small)
    # Prefer sources in 100-1000 range for diversity, then fill with larger if needed
    medium = class_candidates[(class_candidates['total'] >= 100) & (class_candidates['total'] <= 1000)].sample(frac=1, random_state=42)
    large = class_candidates[class_candidates['total'] > 1000].sample(frac=1, random_state=42)
    small = class_candidates[(class_candidates['total'] >= 50) & (class_candidates['total'] < 100)].sample(frac=1, random_state=42)
    
    ordered = pd.concat([medium, large, small])
    
    accumulated = 0
    selected = []
    for source, row in ordered.iterrows():
        if accumulated >= class_target:
            break
        selected.append((source, name, int(row['total'])))
        accumulated += int(row['total'])
    
    print(f'{name} (target={class_target}, selected={accumulated}):')
    for s, c, n in selected:
        print(f'  {s:<35} {n:>5} articles')
    holdout_sources.extend([s[0] for s in selected])
    print()

# Summary
holdout_df = df[df['source'].isin(holdout_sources)]
train_df = df[~df['source'].isin(holdout_sources)]
print(f'Holdout: {len(holdout_df)} articles ({len(holdout_df)/len(df)*100:.1f}%) from {len(holdout_sources)} sources')
print(f'Train:   {len(train_df)} articles ({len(train_df)/len(df)*100:.1f}%)')
print(f'Holdout class balance: L={sum(holdout_df["label"]==0)} C={sum(holdout_df["label"]==1)} R={sum(holdout_df["label"]==2)}')
print(f'Train class balance:   L={sum(train_df["label"]==0)} C={sum(train_df["label"]==1)} R={sum(train_df["label"]==2)}')
print()
print('HOLDOUT_SOURCES =', sorted(holdout_sources))