from transformers import AutoModel, AutoModelForSequenceClassification
import yaml

with open('configs/default.yaml') as f:
    default_config = yaml.safe_load(f)

for key, value in default_config.items():
    if not isinstance(value, dict) or 'repo' not in value:
        continue

    repo = value['repo']
    print(f'\n{"="*60}')
    print(f'  {key} — {repo}')
    print(f'{"="*60}')

    print('\n--- AutoModel (encoder only) ---')
    m = AutoModel.from_pretrained(repo)
    for name, mod in m.named_children():
        print(f'\n{name}: {type(mod).__name__}')
        for sub_name, sub_mod in mod.named_children():
            print(f'  {sub_name}: {sub_mod.__class__.__name__}', end='')
            if hasattr(sub_mod, 'weight'):
                print(f'  weight={tuple(sub_mod.weight.shape)}', end='')
            if hasattr(sub_mod, 'num_embeddings'):
                print(f'  vocab={sub_mod.num_embeddings}, dim={sub_mod.embedding_dim}', end='')
            print()

    print('\n--- ForSequenceClassification (encoder + head) ---')
    m2 = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=3)
    for name, mod in m2.named_children():
        print(f'\n{name}: {type(mod).__name__}')
        for sub_name, sub_mod in mod.named_children():
            print(f'  {sub_name}: {type(sub_mod).__name__}', end='')
            if hasattr(sub_mod, 'weight'):
                print(f'  weight={tuple(sub_mod.weight.shape)}', end='')
            print()
            for sub2_name, sub2_mod in sub_mod.named_children():
                print(f'    {sub2_name}: {type(sub2_mod).__name__}', end='')
                if hasattr(sub2_mod, 'weight'):
                    print(f'  weight={tuple(sub2_mod.weight.shape)}', end='')
                print()

    del m, m2
