# Digital Systems Project: Multi-task BERT-based Classification of Political Alignment in Online Media 
<b>Jaime Spencer (23037459)</b>

# Outline
- `src/train.py` is the primary training file, designed for executing bulk experiments in sequence as defined in `configs/experiments`, which extends the base architecture configurations in `configs/defaults`.
- Affect labels are affixed in `src/MTL/data_assign_affect.py`.
  
- The following subdirectories contain scripts for:
   - `src/data/`
      - Raw data processing
      - Composite dataset creation
      - Group split seed selection
      - Corpus analysis
  - `src/MTL/`
      - Silver labelling
      - Silver label evaluation 
  - `src/golden-set/`
      - Golden Set scraping
        
      - Golden Set construction
      - Golden Set evaluation and figure generation.
  - `src/figures/`
      - figure generation
- Key select final evaluation metric logs can be found in  `results/metrics/`
- AI generated Integrated Gradients (IG) token attribute examples can be found in `results/ig_demo`
    - Unused due to time limitations.
      
# Datasets
- `/data/processed` includes both the non-silver-labelled training corpus `composite_dataset_keep_source` and the MTL-ready `combined_multi_task_source`.
- `/data/golden_set` contains golden set (HF Dataset) as well as a `.csv` version.
  
# Evidence of training runs (wandb)
<img width="1369" height="716" alt="image" src="https://github.com/user-attachments/assets/42b6f6c1-47bc-4baa-814b-b1ed7e5e2436" />

