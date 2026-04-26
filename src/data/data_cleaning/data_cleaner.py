import os
from datasets import Dataset, load_from_disk
import pandas as pd
import os

# CONFIG
SUPPORTED_EXTS = (".csv", ".arrow")
LIMIT_ROWS = None # Set to integer for debugging with smaller datasets

LABEL2ID = {
            # Kaggle: Reduce from granular 5 labels down to 3
            "left":0, "leaning-left": 0, 
            "centre": 1, "center": 1, 
            "leaning-right": 2, "right": 2,
            
            # Webis Bias Flipper
            "from the left": 0,
            "from the right": 2,
            "from the center": 1, 

            # ABP
            "0": 0,
            "1": 1,
            "2": 2
            } 

COLUMN_NAMES = {
    "kaggle": {"text": "page_text", "label": "bias"},
    "webis": {"text": "original_body", "label": "bias"},
    "ABP": {"text": "content", "label": "bias_text"}}

LABEL_COL_NAME = "bias"

class DataCleaner:
    """Class to handle data cleaning and validation"""

    # @staticmethod
    # def process_dataset(raw_dataset_path: str, output_dir, verbose = False, reprocess = False): 
    #     """
    #     Process the raw dataset:
    #     - Load raw data
    #     - Clean text and labels and keep 'text' and 'label' columns only
    #     - Save cleaned data to new dataset
    #     - Validate cleaned data
    #     Return path to cleaned dataset        
    #     """
    #     is_csv = is_arrow = False
    #     if raw_dataset_path.endswith(".csv"):
    #         is_csv = True
    #     elif raw_dataset_path.endswith(".arrow") or os.path.isdir(raw_dataset_path):
    #         is_arrow = True
    #     else:
    #         raise ValueError(f"[process_dataset] Unsupported format for '{raw_dataset_path}'. Expected {SUPPORTED_EXTS}")
        
    #     # Define cleaned dataset path
    #     base_name_ext = os.path.basename(raw_dataset_path)
    #     base_name, _ = os.path.splitext(base_name_ext)
    #     cleaned_name = base_name + "_cleaned.arrow"
    #     cleaned_path = os.path.join(output_dir, cleaned_name)

    #     if not reprocess:
    #         if os.path.exists(cleaned_path):
    #             print(f"[process_dataset] Found existing cleaned dataset at {cleaned_path}. Skipping processing.")
    #             return cleaned_path, ".arrow"
        
    #     print(f"[process_dataset] Cleaning dataset: {raw_dataset_path}")
        
    #     try:
    #         if LIMIT_ROWS is not None:
    #             print(f"[process_dataset] Reading first {LIMIT_ROWS} (LIMIT_ROWS) rows.")
    #         if is_csv:
    #             if LIMIT_ROWS is not None:
    #                 df = pd.read_csv(raw_dataset_path, nrows=LIMIT_ROWS, on_bad_lines='skip', engine='python') # For malformed lines
    #             else:
    #                 df = pd.read_csv(raw_dataset_path, on_bad_lines='skip', engine='python')
    #         elif is_arrow:
    #             print(f"[process_dataset] Loading Arrow dataset from {raw_dataset_path}")
    #             hf = load_from_disk(raw_dataset_path)
    #             df = hf.to_pandas()
    #             if LIMIT_ROWS is not None:
    #                 df = df.head(LIMIT_ROWS)
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"[process_dataset] Could not find file at {raw_dataset_path}")
        
    #     # Clean data, removing bad entries and mapping labels. Return df with only 'text' and 'label' columns.    
    #     df_clean = DataCleaner.clean_data(df, verbose) 
        
    #     if DataCleaner.validate_data(df_clean):
    #         if reprocess:
    #             os.makedirs(output_dir, exist_ok=True)
    #             cleaned_dataset = Dataset.from_pandas(df_clean)
    #             cleaned_dataset.save_to_disk(cleaned_path)
    #         print(f"Cleaned {base_name_ext} saved to {cleaned_path} as Arrow dataset.\n")
    #         return cleaned_path, ".arrow"
    #     else:
    #         raise Exception("[process_dataset] ERROR: Data no validated.")

    # @staticmethod
    # def clean_data(df: pd.DataFrame, verbose = True) -> pd.DataFrame:
    #     """
    #     Cleans the raw dataframe by:
    #     - Removing rows with missing labels or text
    #     - Cleaning text (removing URLs, newlines, extra spaces)
    #     - Mapping string labels to integers
    #     """

    #     if 'page_text' in df.columns: # Kaggle dataset
    #         print("[_clean_data] Detected Kaggle Schema")
    #         text_col = 'page_text' 
    #         label_col = 'bias'
    #     elif 'body' in df.columns and 'title' in df.columns: #Webis Bias Flipper
    #         print("[_clean_data] Detected Webis Schema")
    #         # df['text'] = df['original_title'].astype(str) + " " + df['original_body'].astype(str) # Combine title and body?
    #         df['text'] = df['original_body'].astype(str) 
    #         text_col = 'text'
    #         label_col = 'bias'
    #     elif 'content' in df.columns: # ABP
    #         print("[_clean_data] Detected ABP Schema")
    #         text_col = 'content'
    #         label_col = 'bias_text'
    #     else:
    #         raise ValueError(f"[_clean_data] Could not find text column. Available: {df.columns}")

    #     # Drop rows with missing labels or text
    #     print(f"[_clean_data] Initial dataset size ('bias'): {len(df)}") if verbose else None
    #     df = df.dropna(subset=['bias', text_col])
    #     print(f"[_clean_data] After dropping NAs ('bias'): {len(df)}") if verbose else None

    #     # TEXT CLEANING
    #     df[text_col] = DataCleaner.clean_text(df[text_col])        
    #     # Map string labels to integers
    #     df['label_temp'] = df[label_col].astype(str).str.lower().str.strip() # Standardise labels for mapping
    #     df['label'] = df['label_temp'].map(LABEL2ID)

    #     # Drop rows where mapping failed
    #     print(f"[_clean_data] Initial dataset size ('label'): {len(df)}") if verbose else None
    #     df = df.dropna(subset=['label'])
    #     print(f"[_clean_data] After dropping NAs ('label'): {len(df)}") if verbose else None

    #     return df[['text', 'label']] # Return only text and label

    @staticmethod
    def clean_data(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
        df[text_col] = df[text_col].astype(str)
        len_before = df[text_col].str.split().str.len().describe()
        # Remove URLs (http, https, www until whitespace)
        df[text_col] = df[text_col].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        # IMPORTANT - Remove parentheses occasionally contained source at the start of the string.
        # (e.g., "( CNN )", "WASHINGTON (Reuters) - ")
        df[text_col] = df[text_col].apply(DataCleaner._strip_source_metadata)
        # Remove one or more newlines and strip whitespace around text entry
        df[text_col] = df[text_col].str.replace(r"\n+", " ", regex=True).str.strip()
        # Drop trivially short entries and excessively long ones (live blogs/news feeds)
        word_counts = df[text_col].str.split().str.len()
        df = df[(word_counts > 100) & (word_counts < 5000)]
        # Drop duplicates and na
        df = df.dropna(subset=["text", "label"])
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        len_after = df[text_col].str.split().str.len().describe()
        print(f"\n[clean_text] Text length before cleaning: {len_before}")
        print(f"\n[clean_text] Text length after cleaning: {len_after}")
        return df

    @staticmethod
    def _strip_source_metadata(text):
        """
        Look at start of given string and identify brackets.
        If brackets contain a short string (<=7 words), assume it's source metadata and remove it.
        E.g. "Washington (CNN) - " or "(Reuters) " at the start.
        """
        prefix_zone = text[:80]
        start_bracket = prefix_zone.find('(')
        end_bracket = prefix_zone.find(')')
        
        if start_bracket != -1 and end_bracket > start_bracket:
            content_inside = text[start_bracket + 1 : end_bracket].strip()
            words_before = text[:start_bracket].split()
            
            # LOGIC:
            # 1. If it's very early (starts with bracket)
            # 2. OR if there are only a few words before it (City name)
            # 3. AND the content inside is short (Publisher/Agency name < 4 words)
            if len(words_before) <= 3 and len(content_inside.split()) <= 3:
                return text[end_bracket + 1:].lstrip(' -—').strip()
                        
        return text

    @staticmethod 
    def validate_data(df: pd.DataFrame) -> bool:
        """
        Validates that dataset has valid labels.
        Asssumes that there are classes 0, 1, 2 only.
        """
        if 'label' not in df.columns:
            print("[Validation] Error: 'label' column missing.")
            return False

        # Check if we have valid labels
        unique_labels = sorted(df['label'].unique())
        print(f"[Validation]: Found labels {unique_labels}")
        print(f"[Validation]: Dataset size {len(df)}")
        print(f"[Validation]: Class balance {df['label'].value_counts()}")
        if set(unique_labels).issubset({0, 1, 2}):
            print("[Validation] Success: All labels are valid.")
            return True
        else:
            print("[Validation] Error: Invalid labels found.")
            return False

    @staticmethod
    def oversample_dataset(dataset: Dataset) -> Dataset:
        """
        Balances a Hugging Face Dataset object by oversampling minority classes.
        Returns a new balanced Dataset object.
        """
        print("[DataCleaner] Balancing Dataset... ")
        
        # Convert to pandas for easier manipulation
        df = dataset.to_pandas()
        
        # Oversampling (co-pilot)
        max_size = df['label'].value_counts().max()
        df_balanced = pd.concat([df] + [
            df[df['label'] == label].sample(max_size - count, replace=True, random_state=42)
            for label, count in df['label'].value_counts().items()
            if count < max_size
        ])
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Original Size: {len(df)}")
        print(f"Balanced Size: {len(df_balanced)}")
        print(f"New Class Distribution:\n{df_balanced['label'].value_counts()}")
        print("------------------------------------------")
        
        # Convert back to Hugging Face Dataset
        return Dataset.from_pandas(df_balanced)     
    



# # CONFIG
# MAX_CORES = 1
# ALIGNMENT_DATASETS = {"kaggle": "/home/jaime/DSP/Project/data/raw/alignment/kaggle_AllSides_scraped_10k/kaggle_AllSides_scraped_10k.csv",
#             "webis": "/home/jaime/DSP/Project/data/raw/alignment/corpus-webis-bias-flipper-18/allsides-collection/data_public.csv",
#             "ABP": "/home/jaime/DSP/Project/data/raw/alignment/Article-Bias-Prediction-main/Article-Bias-Prediction-main/data/arrow_news_dataset"}

#             # Process ALIGNMENT_DATASETS
# REPROCESS = True
# output_dir = "/home/jaime/DSP/Project/data/cleaned/alignment"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)
    
# for dataset_name, dataset_path in ALIGNMENT_DATASETS.items():
#     dataset_path, ext = DataCleaner.process_dataset(dataset_path, output_dir=output_dir, verbose=True, reprocess=REPROCESS)
