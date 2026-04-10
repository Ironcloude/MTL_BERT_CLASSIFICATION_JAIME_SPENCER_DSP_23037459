"""
SID: 23037459
PROJECT: Multi-task BERT-based Classification of Political Alignment in Online Media
MODULE: UXCFXK-30-3 Digital Systems Project (2025-26) 

CORE ARCHITECTURES, HF MODEL ID & ATTRIBUTION:
- BERT:                 'bert-base-uncased' (Devlin et al., 2018)).
- DeBERTa:              'microsoft/deberta-base'  (He et al. (2020)) 
- ELECTRA:              'google/electra-base-discriminator (Clark et al. (2020)).
- DeBERTav3:            'microsoft/deberta-v3-base'(He et al. (2021)).
                            - Utilizes Replaced Token Detection (RTD) and Gradient Disentangled Attention.
- ModernBERT:           'answerdotai/ModernBERT-base' (Warner et al. (2024))
                            - Implements GeLU activations and mean-pooled hidden states for classification.
- RoBERTa (TweetEval):  'cardiffnlp/twitter-roberta-base-sentiment-latest' (Barbieri et al. (2022)).
                            - Sentiment Backbone. Used for auxiliary silver-label generation.

ORIGINAL CONTRIBUTIONS:
- Custom MultiTaskModel:    Shared encoder backbone with independent task-specific MLP heads.

- Loss Function:            Joint optimization using Weighted Cross-Entropy (Primary: Political) 
                            and standard Cross-Entropy (Auxiliary: Sentiment) controlled by lambda (λ).

- Sampling:                 Source-stratified partitioning logic to prevent outlet-style leakage.
"""

import math
import sys
from datasets import load_from_disk
import datasets
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, AutoModel, PreTrainedTokenizer,
                          EarlyStoppingCallback, TrainerCallback)
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from datetime import datetime
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import os
import gc
import yaml
import json
import re



# Define custom model to extend to multi-task
#  Returns two sets of logits for each head. The issue is that 

@dataclass
class MTOutput:
    """Wrapper so the HuggingFace Trainer can unwrap logits from a tuple-returning model.
    The Trainer does: if len(output) == 1: logits = output[0]
    So __len__ returns 1 and __getitem__ returns the logits tuple."""
    logits: Tuple  # (political_logits, sentiment_logits)
    loss: Optional[torch.Tensor] = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.logits

class MultiTaskModel(nn.Module):
    """
    Clasification summary
    ----------------------
    BERT	Pooler(Dense+Tanh) => Dropout => Linear
    DeBERTa	Pooler(Dense+GELU+LayerNorm) => Dropout => Linear
    ELECTRA	Dense(GELU) => Dropout => Linear
    ModernBERT	Dense => GELU => LayerNorm => Dropout => Linear
    RoBERTa	Dense(Tanh) => Dropout => Linear
    
    This implementatiojn
    ---------
    Matches ModernBertForSequenceClassification head structure:
    MeanPool => Dense(768,768) => GELU => LayerNorm => Linear(768, n)
    (classifier_dropout=0.0, classifier_pooling="mean")"""
    def __init__(self, model_repo, num_political=3, num_sentiment=5):
        super().__init__()
        # https://huggingface.co/docs/transformers/en/model_doc/modernbert
        self.encoder = AutoModel.from_pretrained(model_repo)
        hidden = self.encoder.config.hidden_size
        drop_p = self.encoder.config.classifier_dropout

        #Explicility define separate heads for political and sentiment tasks, 
        # each with its own Dense => GELU => LayerNorm => Dropout => Linear structure,
        #  matching ModernBERT's classification head. 
        # Political head (matches ModernBertPredictionHead)
        self.pol_dense = nn.Linear(hidden, hidden, bias=False)
        self.pol_act = nn.GELU()
        self.pol_norm = nn.LayerNorm(hidden)
        self.pol_drop = nn.Dropout(drop_p)
        self.political_head = nn.Linear(hidden, num_political)

        # Sentiment head (same structure)
        self.sent_dense = nn.Linear(hidden, hidden, bias=False)
        self.sent_act = nn.GELU()
        self.sent_norm = nn.LayerNorm(hidden)
        self.sent_drop = nn.Dropout(drop_p)
        self.sentiment_head = nn.Linear(hidden, num_sentiment)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Mean pooling over non-padding tokens (matches classifier_pooling="mean") as specified in standard ST config.

        Returns tuple of (political_logits, sentiment_logits) wrapped in custom MTOutput for HF Trainer compatibility.
        """
        # Batch size (number of articles) = B, sequence length = L, hidden size = H
        # final layer tensor: [B, L, H] => [64, 512, 768]
        token_embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        # Remove padding tokens from mean pooling using attention mask
        # attention_mask has shape [B, L] => unsquueze adds a dummy dimension for multiplication => [B, L, 1]
        padding_mask = attention_mask.unsqueeze(-1).float() 
        # Mean pooling over non-padding tokens; contract token embeeddings into article embeddings.
        # Numerator: (final_hidden_states * mask).sum(dim=1) => [B, H] sums hidden states of non-padding tokens per article
        # Denominator: mask.sum(dim=1) ==> [B, 1] gives count of non-padding tokens per article; get the mean rather than sum.
        article_embeddings = (token_embeddings * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)

        # Pass mean-pooled embeddings to task-specific heads
        pol_features = self.pol_drop(self.pol_norm(self.pol_act(self.pol_dense(article_embeddings))))
        sent_features = self.sent_drop(self.sent_norm(self.sent_act(self.sent_dense(article_embeddings))))
        pol_logits = self.political_head(pol_features)
        sent_logits = self.sentiment_head(sent_features)
        return MTOutput(logits=(pol_logits, sent_logits))


# Define source aliases for source normalisation 
SOURCE_ALIASES = {
    "AP Fact Check": "Associated Press",
    "CNN (Web News)": "CNN", "CNN (Opinion)": "CNN",
    "CNN - Editorial": "CNN", "CNN Fact Check": "CNN",
    "NBC (Web News)": "NBC News", "NBC News (Online)": "NBC News",
    "NBC Today Show": "NBC News", "NBCNews.com": "NBC News",
    "NBC 5 Chicago": "NBC News",
    "NPR News": "NPR", "NPR Online News": "NPR", "NPR Editorial": "NPR",
    "Newsmax (News)": "Newsmax", "Newsmax (Opinion)": "Newsmax",
    "Newsmax - News": "Newsmax", "Newsmax - Opinion": "Newsmax",
    "Guest Writer - Left": "Guest Writer",
    "Guest Writer - Right": "Guest Writer",
    "Guest Writer - Center": "Guest Writer",
    "Fox Online News": "Fox News", "Fox News Opinion": "Fox News",
    "Breitbart News": "Breitbart", "Breitbart Fact Check": "Breitbart",
    "Buzzfeed": "BuzzFeed News",
    "CBS SFBayArea": "CBS News",
    "CNSNews.com": "CNS News",
    "DesMoines Register": "Des Moines Register",
    "The Western Journal": "Western Journal",
    "Boston Herald Editorial": "Boston Herald",
    "MichelleMalkin.com": "Michelle Malkin",
    "The American Spectator": "American Spectator",
    "Reason Foundation": "Reason",
    "RedState": "Red State",
    "Voice of America (VOA)": "Voice of America",
}

def _normalise_source(s):
    if s in SOURCE_ALIASES:
        return SOURCE_ALIASES[s]
    s = re.sub(r'\s*\(.*?\)', '', s)
    s = re.sub(r'\s*-\s*(News|Opinion|Editorial|Blog).*', '', s)
    s = re.sub(r'\s+(Digital|Online|Latino)$', '', s)
    s = re.sub(r'\s+Fact Check$', '', s)
    s = re.sub(r'\s+Editorial Board$', '', s)
    return s.strip()

# CONFIG
DEBUG = False # True to test with a small subset of the data and fewer training steps
MAX_CORES = 8
WANDB_PROJECT = "political-bias-detection-stratified-final"
STRATIFY_BY_SOURCE = True
FRESH_TOKENISATION = True
MTL_EVAL_NO_LOSS = False
# Set total log steps for metrics (lower => faster training but coarser metrics. HOWEVER, with early stopping enabled more evals == more chances to stop early)
LOG_NUM = 30


# Load training configurations
_config_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "configs")

with open(os.path.join(_config_dir, "default.yaml")) as f:
    _defaults = yaml.safe_load(f)

with open(os.path.join(_config_dir, "experiments.yaml")) as f:
    _experiments = yaml.safe_load(f)

def _resolve(exp_cfg: dict) -> dict:
    """Merge model-level defaults with experiment-level overrides."""
    model_key = exp_cfg["model"]
    base = dict(_defaults.get("default", {}))   # shared defaults 
    base.update(_defaults.get(model_key, {}))    # model-specific overrides
    base.update(exp_cfg)                         # apply experiment-specific overrides
    return base

TO_TRAIN = {name: _resolve(cfg) for name, cfg in _experiments.items()}
_order = [ "EX-12"]
TO_TRAIN = {key: TO_TRAIN[key] for key in _order if key in TO_TRAIN}
if __name__ == "__main__":
    for experiment, _ in TO_TRAIN.items():
        print(f"\n{'='*20}\nStarting experiment: {experiment}\n{'='*20}\n{TO_TRAIN[experiment]['description']}\n")
        print(TO_TRAIN[experiment])
        try:

            # Resume from checkpoint specified
            RESUME_FROM_CHECKPOINT = ""
            if RESUME_FROM_CHECKPOINT:
                os.environ["WANDB_RUN_ID"] = "" #  ls /home/jaime/DSP/Project/results/wandb/ | sort | tail -10
                os.environ["WANDB_RESUME"] = "must"
            else:
                os.environ.pop("WANDB_RUN_ID", None)
                os.environ.pop("WANDB_RESUME", None)

            # MODEL SELECTION
            model = TO_TRAIN[experiment]
            model_repo = model["repo"]
            model_name = model["model"][:3]
            # MULTI-TASK
            multi_task = model["multi_task"]
            LAMBDA     = model.get("lambda", 1)

            # DATASET
            FRESH_TOKENISATION = True
            DATASET_PERCENTAGE = 1
            TEST_SPLIT = 0.2
            seq_len = model["sequence_len"]

            # TRAINING PARAMETERS
            FREEZE = model.get("freeze", False)
            FREEZE_EMBEDDINGS = model.get("freeze_embeddings", False)
            FREEZE_LAYERS = model.get("freeze_layers", 0)  # 0=none, 3=first 3 layers, 9=first 9 layers, etc.
            use_bf16 = model["bf16"]
            native_bf16 = model.get("native_bf16", False)
            EPOCHS = model.get("epochs", 10)
            BATCH_SIZE = model["batch_size"] #512 = 16/8; 1024 = 4; 2048 = 1
            GRADIENT_ACCUMULATION_STEPS = model["grad_acc_steps"] # 512=4; 1024=16; 2048=64
            effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS #Should == 64
            WARMUP_STEPS = 500
            WARMUP_RATIO = None
            weight_decay = float(model["weight_decay"])
            learning_rate = float(model["lr"])
            adam_epsilon = float(model["adam_e"])
            lr_scheduler_type = model.get("lr_scheduler_type", "linear")
            early_stopping_patience = model.get("early_stopping_patience", 3)

            # DATASET PATH
            dataset_path = "/home/jaime/DSP/Project/data/processed/combined_multi_task_source"
            # SAVING & LOGGING
            if not RESUME_FROM_CHECKPOINT:
                run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
                task_type = f"mt-λ{LAMBDA}" if multi_task else "st"
                log_name_short = f"{'EVALF1-' if MTL_EVAL_NO_LOSS else ''}{experiment}-{model_name}-{task_type}-lr-{learning_rate}-{int(DATASET_PERCENTAGE*100)}pct-{seq_len}"
                log_name = f"{log_name_short}-{run_id}"

            if RESUME_FROM_CHECKPOINT:
                log_name = RESUME_FROM_CHECKPOINT
                log_name_short = "-".join(RESUME_FROM_CHECKPOINT.split("-")[:-2])
                
            project_root = os.path.abspath(".") 
            multi_task_suffix = "multi_task" if multi_task else ""
            processed_data_path = os.path.join(project_root, "data", "tokenized", f"{experiment}_{seq_len}_{multi_task_suffix}")
            # LOAD DATA
            # Load datasets in Hugging Face Format and concatenate

            dataset = load_from_disk(dataset_path)
            print(dataset)
            
            # SPLIT DATASET — source-stratified (no outlet appears in both train and eval)
            # Ensures eval metrics reflect cross-source generalisation, not in-distribution performance
            if STRATIFY_BY_SOURCE:
                sources_normalised = [_normalise_source(s) for s in dataset["source"]]
                gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=3189) # 3189 was best seed from search_split_seed.py
                train_idx, test_idx = next(gss.split(X=range(len(dataset)), groups=sources_normalised))
                dataset = datasets.DatasetDict({
                    "train": dataset.select(train_idx),
                    "test":  dataset.select(test_idx),
                })
                train_sources = set(sources_normalised[i] for i in train_idx)
                test_sources  = set(sources_normalised[i] for i in test_idx)
                print(f"Training set: {len(dataset['train'])} articles from {len(train_sources)} normalised sources")
                print(f"Eval set:     {len(dataset['test'])} articles from {len(test_sources)} normalised sources")
                overlap = train_sources & test_sources
                print(f"Source overlap: {len(overlap)} (should be 0)")
                if overlap:
                    print(f"  [ERROR] LEAKED SOURCES: {overlap}")
                    exit(1)
                label_names = {0: "Left", 1: "Centre", 2: "Right"}
                for split_name, split_data in [("Train", dataset["train"]), ("Eval", dataset["test"])]:
                    labels = split_data["label"]
                    total = len(labels)
                    print(f"  {split_name} class balance: " + "  ".join(
                        f"{label_names[c]}={labels.count(c)/total:.1%}" for c in [0, 1, 2]))
            else:
                train_test_split(test_size=TEST_SPLIT, seed=42, shuffle=True)
                print(f"Training set size: {len(dataset['train'])}")

            # TOKENISATION
            # https://www.youtube.com/watch?v=nvBXf7s7vTI
            # Batch Tokenisation    
            # BERT - WordPiece, Subword 30k vocab
            # Debertav3 - SentencePiece tokeniser, - subword 128k vocab. 
            # ModernBERT - OLMo tokenizer, byte-level subword 50k vocab
            tokeniser: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True) 
            print(f"Loading tokeniser for {model_repo}...")

            def tokenize_function(batch):
                """Passed to map to process sample batches."""
                return tokeniser(batch["text"], truncation=True, max_length=seq_len)

            # Cache tokeised dataset
            if os.path.exists(processed_data_path) and not FRESH_TOKENISATION:
                print("Loading tokenized data from disk...")
                tokenised_dataset = load_from_disk(processed_data_path)
            else:
                print(f"Tokenising datasets with max length {seq_len}...")
                tokenised_dataset = dataset.map(tokenize_function, batched=True)
                tokenised_dataset.save_to_disk(processed_data_path)

            # Remove non-feature columns before training
            for drop_col in ["text", "source"]:
                if drop_col in tokenised_dataset["train"].column_names:
                    tokenised_dataset = tokenised_dataset.remove_columns([drop_col])

            if multi_task:
                def merge_labels(batch):
                    batch["label"] = [[p, s] for p, s in zip(batch["label"], batch["sentiment_label"])]
                    return batch
                tokenised_dataset = tokenised_dataset.map(merge_labels, batched=True)
                tokenised_dataset = tokenised_dataset.remove_columns(["sentiment_label"])
            else:
                cols = tokenised_dataset["train"].column_names
                if "sentiment_label" in cols:
                    tokenised_dataset = tokenised_dataset.remove_columns(["sentiment_label"])

            # Create data collator 
            data_collator = DataCollatorWithPadding(tokenizer=tokeniser)


            # Define weighted cross entropy to handle class imbalances
            if multi_task:
                train_labels = [labels[0] for labels in tokenised_dataset["train"]["label"]]
            else:
                train_labels = tokenised_dataset["train"]["label"]

            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

            # Convert to tensor and move to GPU
            w_dtype = torch.bfloat16 if native_bf16 else torch.float32
            if multi_task:
                weights_tensor = torch.tensor(class_weights, dtype=w_dtype)
            else:
                weights_tensor = torch.tensor(class_weights, dtype=w_dtype)
                
            def weighted_cross_entropy(outputs, labels, num_items_in_batch=None):
                """Custom loss function to handle class imbalance."""
                if labels is None:
                    print("[DEBUG] labels is None during eval — returning dummy loss")
                    return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=False)
                # print(f"[DEBUG LOSS] dtype={labels.dtype}, shape={labels.shape}")
                if multi_task:
                    political_logits, sentiment_logits = outputs.logits
                    political_labels = labels[:, 0].long()
                    sentiment_labels = labels[:, 1].long()
                    weights = weights_tensor.to(political_logits.device, dtype=political_logits.dtype)
                    l_political = torch.nn.functional.cross_entropy(political_logits, political_labels, weight=weights)
                    l_sentiment = torch.nn.functional.cross_entropy(sentiment_logits, sentiment_labels)
                    # weigh
                    loss = LAMBDA * l_political + (1 - LAMBDA) * l_sentiment
                else:
                    logits = outputs.get("logits")
                    final_weights = weights_tensor.to(logits.device, dtype=logits.dtype)  
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, model.config.num_labels), 
                        labels.view(-1), 
                        weight=final_weights
                    )
                return loss
            


            # Create instance of model, dependent on task type
            # https://www.baeldung.com/cs/learning-rate-batch-size
            if multi_task:
                model = MultiTaskModel(model_repo, num_political=3, num_sentiment=5)
                # Custom MTL does not have config generated
                config_dict = model.encoder.config.to_dict()
                config_path = os.path.join(project_root, "models", "final", f"{log_name}")
                os.makedirs(config_path, exist_ok=True)
                with open(os.path.join(config_path, f"config.json"), "w") as f:
                    json.dump(config_dict, f, indent=4) # ind
                print(f"Wrote JSON to: {os.path.join(config_path, f'config.json')}")
            else:
                if native_bf16:
                    # Native bf16 cast (like Colab's model.bfloat16()) — avoids autocast overhead
                    # Required for DeBERTav3
                    print("Using native BF16 precision for model and weights")
                    model = AutoModelForSequenceClassification.from_pretrained(model_repo, num_labels=3).bfloat16()
                else:
                    print("Using standard FP32 (BF16 if cast bf16=true) precision for model and weights")
                    model = AutoModelForSequenceClassification.from_pretrained(model_repo, num_labels=3).float()
            print(f"Selected model: {experiment} ({model_repo})")
        

            def freeze_weights(model, freeze_all=False, freeze_n_layers=0):
                """Freeze encoder layers for transfer learning.

                Args:
                    freeze_all: If True, freeze entire backbone except classifier head
                    freeze_n_layers: If >0, freeze first N encoder layers (e.g., 9 out of 12)
                """
                if freeze_all:
                    # Original behavior: freeze entire backbone
                    backbone = list(model.named_children())[0][0]
                    backbone = getattr(model, backbone)
                    for param in backbone.parameters():
                        param.requires_grad = False
                    for name, param in model.named_parameters():
                        if "classifier" in name:
                            param.requires_grad = True
                    print(f"Froze entire backbone: '{backbone}'")
                elif freeze_n_layers > 0:
                    # Find the backbone: BERT/ELECTRA have 'encoder', DeBERTa has 'deberta'
                    if hasattr(model, 'encoder'):
                        # BERT, ELECTRA, etc.
                        encoder = model.encoder
                        embeddings = encoder.embeddings
                        layer_list = encoder.layer
                    elif hasattr(model, 'deberta'):
                        # DeBERTa: embeddings are at model.deberta.embeddings
                        encoder = model.deberta.encoder
                        embeddings = model.deberta.embeddings
                        layer_list = encoder.layer
                    else:
                        print(f"[WARN] Could not find encoder backbone, skipping freeze_layers")
                        return

                    # Freeze embeddings
                    for param in embeddings.parameters():
                        param.requires_grad = False

                    # Freeze first N layers
                    for param in layer_list[:freeze_n_layers].parameters():
                        param.requires_grad = False
                    print(f"Froze embeddings + first {freeze_n_layers} encoder layers")

                # Always print trainable params
                trainable = [name for name, param in model.named_parameters() if param.requires_grad]
                print(f"Trainable parameters ({len(trainable)}):")
                for name in trainable[:10]:  # Print first 10
                    print(f"  {name}")
                if len(trainable) > 10:
                    print(f"  ... and {len(trainable) - 10} more")

            if FREEZE:
                freeze_weights(model, freeze_all=True)
            elif FREEZE_LAYERS > 0:
                freeze_weights(model, freeze_n_layers=FREEZE_LAYERS)

            # if WARMUP_RATIO:
            #     num_rows = tokenised_dataset.num_rows['train']
            #     warmup_steps = int(DATASET_PERCENTAGE * WARMUP_RATIO * (num_rows / effective_batch_size) * EPOCHS)
            # else:
            warmup_steps = WARMUP_STEPS

            # Total training steps = (num_samples / (batch_size * gradient_accumulation_steps)) * num_epochs
            total_train_steps = int(((len(dataset['train']) * DATASET_PERCENTAGE) / effective_batch_size) * EPOCHS)
            total_log_steps = max(1, int(total_train_steps / LOG_NUM))

            def compute_metrics(eval_pred): 
                logits, labels = eval_pred
                print(f"[DEBUG] logits type={type(logits)}, shape[0]={np.array(logits[0]).shape}")
                if multi_task:
                    political_preds = np.argmax(logits[0], axis=-1)
                    sentiment_preds = np.argmax(logits[1], axis=-1)
                    political_labels = labels[:, 0]
                    sentiment_labels = labels[:, 1]
                    per_class = f1_score(political_labels, political_preds, average=None)
                    return {
                            'political_f1':  f1_score(political_labels, political_preds, average='macro'),
                            'political_acc': accuracy_score(political_labels, political_preds),
                            'political_f1_left':   per_class[0],
                            'political_f1_centre': per_class[1],
                            'political_f1_right':  per_class[2],
                            'sentiment_f1':  f1_score(sentiment_labels, sentiment_preds, average='macro'),
                        }
                else:
                    political_preds = np.argmax(logits, axis=-1)
                    per_class = f1_score(labels, political_preds, average=None)
                    return {
                        'political_f1':        f1_score(labels, political_preds, average='macro'),
                            'political_acc':       accuracy_score(labels, political_preds),
                            'political_f1_left':   per_class[0],
                            'political_f1_centre': per_class[1],
                            'political_f1_right':  per_class[2],
                        }


            train_dataset = tokenised_dataset["train"].shuffle(seed=42).select(range(int(tokenised_dataset.num_rows['train'] * DATASET_PERCENTAGE)))
            eval_dataset = tokenised_dataset["test"].shuffle(seed=42).select(range(int(tokenised_dataset.num_rows['test'] * DATASET_PERCENTAGE)))
            training_args = TrainingArguments(
                output_dir=f"/home/jaime/DSP/Project/models/final/{RESUME_FROM_CHECKPOINT or log_name}",
                eval_strategy="steps",
                eval_steps=total_log_steps,
                save_strategy="steps",
                save_total_limit=3,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                dataloader_num_workers=MAX_CORES // 2,
                dataloader_pin_memory=True,
                learning_rate=learning_rate,
                adam_epsilon=adam_epsilon,
                bf16=use_bf16 and not native_bf16,
                seed=42,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                max_grad_norm=1,
                logging_steps=total_log_steps,
                save_steps=total_log_steps,
                load_best_model_at_end=True,
                # Eval loss for multi-task to avoid overfitting to political head at expense of sentiment head. 
                # Political F1 for single-task as normal.
                metric_for_best_model="eval_loss" if multi_task and not MTL_EVAL_NO_LOSS else "eval_political_f1",
                greater_is_better=False if multi_task else True,
                report_to="wandb",
                run_name=log_name,
                label_names=["labels"],
                lr_scheduler_type=lr_scheduler_type
            )
                
            class MultiTaskTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    # inline weighted cross-entropy for logging
                    # labels = inputs.pop("labels")
                    # outputs = model(**inputs)
                    # loss = weighted_cross_entropy(outputs, labels, num_items_in_batch)

                    # for logging - weighted
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)

                    political_logits, sentiment_logits = outputs.logits
                    political_labels = labels[:, 0].long()
                    sentiment_labels = labels[:, 1].long()
                    weights = weights_tensor.to(political_logits.device, dtype=political_logits.dtype)

                    l_political = torch.nn.functional.cross_entropy(political_logits, political_labels, weight=weights)
                    l_sentiment = torch.nn.functional.cross_entropy(sentiment_logits, sentiment_labels)
                    loss = LAMBDA * l_political + (1 - LAMBDA) * l_sentiment

                    if model.training and self.state.global_step % self.args.logging_steps == 0:
                        self.log({"loss_political": l_political.item(),
                                "loss_sentiment": l_sentiment.item(),
                                "loss_combined":  loss.item()})

                    return (loss, outputs) if return_outputs else loss

            class SingleTaskTrainer(Trainer):
                "Remove gradient accumulation multipler"
                def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    loss = weighted_cross_entropy(outputs, labels)
                    return (loss, outputs) if return_outputs else loss

            print(f"TRAINING ARGS\n{'='*10}\n{training_args}")
            class NanStoppingCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and (math.isnan(logs.get("loss", 0)) or math.isinf(logs.get("loss", 0))):
                        print(f"\n[FATAL] NaN/Inf loss detected at step {state.global_step}. Stopping.")
                        control.should_training_stop = True

            callbacks = [NanStoppingCallback()]
            if early_stopping_patience:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

            if multi_task:
                trainer = MultiTaskTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            else:
                trainer = SingleTaskTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    callbacks=callbacks,
                )

            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            print("DEBUG: Decoding three random training examples...")
            for i in range(3):
                random_idx = np.random.randint(0, len(train_dataset))
                print(f"\n{tokeniser.decode(train_dataset[random_idx]['input_ids'])}")
            print("\nStarting training...")

            os.environ["WANDB_DIR"] = os.path.abspath("../results")  # 
            os.environ["WANDB_PROJECT"] = WANDB_PROJECT
            trainer.train(resume_from_checkpoint=True if RESUME_FROM_CHECKPOINT else False)
            trainer.save_model()  # saves the best model (load_best_model_at_end=True)

            log_df = pd.DataFrame(trainer.state.log_history)
            os.makedirs("/home/jaime/DSP/Project/results/logs", exist_ok=True)
            log_df.to_csv(f"/home/jaime/DSP/Project/results/logs/{log_name}.csv", index=False)
            
            # Load and evaluate golden set
            # Final evaluation
            results = trainer.predict(eval_dataset)
            if multi_task:
                    political_preds = np.argmax(results.predictions[0], axis=-1)
                    political_labels = results.label_ids[:, 0]
            else:
                    political_preds = np.argmax(results.predictions, axis=-1)
                    political_labels = results.label_ids
            print(classification_report(political_labels, political_preds,
                    target_names=["Left", "Centre", "Right"]))
            # plot cm with seaborn heatmap
            cm = confusion_matrix(political_labels, political_preds)

            # Normalise by true label (row) so each cell shows recall per class
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=["Left", "Centre", "Right"],
                        yticklabels=["Left", "Centre", "Right"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            os.makedirs('../results/figures', exist_ok=True)
            f1 = results.metrics.get('test_political_f1', float('nan'))
            plt.title(f"{log_name_short} Multi-task - Political Leaning (macro F1={f1:.3f})")
            plt.tight_layout()
            plt.savefig(f"../results/figures/cm_{log_name}.png", dpi=150)
            plt.close()
            wandb.finish()

        except Exception as e:
            print(f"\n[ERROR] Experiment {experiment} failed: {e}")
            import traceback; traceback.print_exc()
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass

        finally:
            # Free GPU and CPU memory before next experiment
            try:
                del model, trainer, tokeniser, tokenised_dataset
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[INFO] Cleaned up after {experiment}")
