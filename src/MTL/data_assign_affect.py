from datasets import load_from_disk
import os
import numpy as np
import pandas as pd
from transformers import pipeline
import torch

DATASET_PATH = "/home/jaime/DSP/Project/data/processed/composite_dataset_keep_source"
OUTPUT_PATH = "/home/jaime/DSP/Project/data/processed/combined_multi_task_source"
REQUIRED_COLUMNS = ["text", "label", "source"]
RERUN_LABELING = True

# Define which affective dimensions to generate silver labels for
EMOTION = False
# Sentiment
SENTIMENT = True
SENTIMENT_MODEL_REPO = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_BATCH_SIZE = 64
SENTIMENT_MAX_LENGTH = 512

# Load dataset in Hugging Face Format
dataset = load_from_disk(DATASET_PATH)
print(dataset)

# Drop extra columns if they exist
extra_columns = [col for col in dataset.column_names if col not in REQUIRED_COLUMNS]
if extra_columns:
    print(f"Dropping extra columns: {extra_columns}")
    dataset = dataset.remove_columns(extra_columns)

if SENTIMENT:
    # Uses 'cardiffnlp/twitter-roberta-base-sentiment-latest' (Barbieri et al., 2020) to generate silver sentiment labels.
    # Labels: 0 = negative; 1 = neutral; 2 = positive
    OUTPUT_PATH = "/home/jaime/DSP/Project/data/processed/combined_multi_task_source"

    # Weighted expectation: P(pos) - P(neg) => continuous score in [-1, 1], then bin into 5 classes
    SENTIMENT_WEIGHTS = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    SENTIMENT_BINS   = [-0.6, -0.2, 0.2, 0.6]  # 5 bins
    SENTIMENT_NAMES  = {0: "strongly_neg", 1: "negative", 2: "neutral", 3: "positive", 4: "strongly_pos"}

    if os.path.exists(OUTPUT_PATH) and not RERUN_LABELING:
        print(f"Labeled dataset already exists at:\n  {OUTPUT_PATH}\nSet RERUN_LABELING=True to overwrite.")
    else:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Running sentiment inference on {'GPU' if device == 0 else 'CPU'}...")
        sentiment_pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL_REPO,
            device=device,
            batch_size=SENTIMENT_BATCH_SIZE,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
            top_k=None,  # return all class probabilities
        )

        def add_sentiment_labels(batch):
            batch_results = sentiment_pipeline(batch["text"])
            labels = []
            for article_results in batch_results:
                article_probs = {result["label"].lower(): result["score"] for result in article_results}
                article_score = sum(SENTIMENT_WEIGHTS[key] * article_probs.get(key, 0.0) for key in SENTIMENT_WEIGHTS)
                bin_label = int(np.digitize(article_score, SENTIMENT_BINS))  # bins 0-4 [-0.6, -0.2, 0.2, 0.6]
                labels.append(bin_label)
            batch["sentiment_label"] = labels
            return batch

        labeled_dataset = dataset.map(
            add_sentiment_labels,
            batched=True,
            batch_size=SENTIMENT_BATCH_SIZE,
            desc="Sentiment labeling",
        )

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        labeled_dataset.save_to_disk(OUTPUT_PATH)
        print(f"\nSaved multi-task dataset ({len(labeled_dataset)} articles) to:\n  {OUTPUT_PATH}")

        df = labeled_dataset.to_pandas()
        print("\nSentiment label distribution (5-class):")
        print(df["sentiment_label"].map(SENTIMENT_NAMES).value_counts().sort_index())
        print("\nSentiment x Political leaning cross-tab:")
        print(pd.crosstab(df["label"], df["sentiment_label"].map(SENTIMENT_NAMES)))

# if EMOTION:
# # Implement Goemotion model https://huggingface.co/cirimus/modernbert-base-go-emotions
# # This model is multi-label, so it returns probabilities for all 28 emotions
#     emotion_teacher = pipeline(
#     "text-classification", 
#     model="cirimus/modernbert-base-go-emotions", 
#     top_k=None, 
#     device=0    
# )
    
#     def get_affective_scores(batch):
#         # Run inference
#         results = emotion_teacher(batch["text"], truncation=True, max_length=1024)
        
#         batch_valence = []
#         batch_arousal = []
#         batch_dominance = []

#         for res in results:
#             # Convert list of dicts to a flat dict for easy summing
#             probs = {item['label']: item['score'] for item in res}

#             # Pool GoEmotions into VAD dimensions (Russell, 1980; Warriner et al., 2013)
#             # Each emotion assigned to exactly one dimension based on its strongest VAD loading.
#             # (Neutral/ambiguous emotions omitted)
            
#             # Valence: pleasure–displeasure (emotional tone)
#             #   +V: joy, love, gratitude, amusement, optimism, caring, relief
#             #   -V: sadness, grief, disappointment, disapproval, remorse, annoyance
#             valence = (
#                 (probs.get('joy', 0) + probs.get('love', 0) +
#                  probs.get('gratitude', 0) + probs.get('amusement', 0) +
#                  probs.get('optimism', 0) + probs.get('caring', 0) +
#                  probs.get('relief', 0))
#                 -
#                 (probs.get('sadness', 0) + probs.get('grief', 0) +
#                  probs.get('disappointment', 0) + probs.get('disapproval', 0) +
#                  probs.get('remorse', 0) + probs.get('annoyance', 0))
#             )

#             # Arousal: activation–deactivation (intensity)
#             #   +A: anger, excitement, fear, surprise, nervousness, desire, disgust
#             #   Deactivated emotions already captured in valence (sadness, relief)
#             arousal = (
#                 probs.get('anger', 0) + probs.get('excitement', 0) +
#                 probs.get('fear', 0) + probs.get('surprise', 0) +
#                 probs.get('nervousness', 0) + probs.get('desire', 0) +
#                 probs.get('disgust', 0)
#             )

#             # Dominance: control–submission (agency)
#             #   +D: pride, admiration, approval (assertive, in-control)
#             #   -D: embarrassment, confusion (submissive, uncertain)
#             dominance = (
#                 (probs.get('pride', 0) + probs.get('admiration', 0) +
#                  probs.get('approval', 0))
#                 -
#                 (probs.get('embarrassment', 0) + probs.get('confusion', 0))
#             )
                
#             batch_valence.append(valence)
#             batch_arousal.append(arousal)
#             batch_dominance.append(dominance)

#         return {"valence": batch_valence, "arousal": batch_arousal, "dominance": batch_dominance}

#     # Apply emotion labelsto dataset
#     affect_dataset = dataset.map(get_affective_scores, batched=True, batch_size=32)
#     save_path = f"political_bias_with_affect_GoEmotion"

#     print(f"Saving dataset to {save_path}...")
#     affect_dataset.save_to_disk(save_path)