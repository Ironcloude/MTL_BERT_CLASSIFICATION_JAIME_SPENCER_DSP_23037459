from transformers import AutoModel
import torch
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

models = {
    'BERT': 'bert-base-uncased',
    'DeBERTav3': 'microsoft/deberta-v3-base',
    'DeBERTav1': 'microsoft/deberta-base',
    'ELECTRA': 'google/electra-base-discriminator',
    'ModernBERT': 'answerdotai/ModernBERT-base',
    'RoBERTa': 'roberta-base'
}

for name, repo in models.items():
    try:
        m = AutoModel.from_pretrained(repo)
        dtype = next(m.parameters()).dtype
        print(f'{name}: {dtype}')
        del m
    except Exception as e:
        print(f'{name}: ERROR - {e}')