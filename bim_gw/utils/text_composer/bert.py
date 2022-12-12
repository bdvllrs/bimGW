from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_bert_latents(data, bert_path, bert_latents, path, device):
    transformer = BertModel.from_pretrained(bert_path)
    transformer.eval()
    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    data_loaders = [
        ("train", data.train_dataloader(shuffle=False)),
        ("val", data.val_dataloader()[0]),  # only keep in dist dataloaders
        ("test", data.test_dataloader()[0])
    ]
    path = Path(path)
    for name, data_loader in data_loaders:
        latents = []
        print(f"Fetching {name} data.")
        for idx, (batch) in tqdm(enumerate(data_loader),
                                 total=int(len(data_loader.dataset) / data_loader.batch_size)):
            sentences = batch["t"][2]
            tokens = tokenizer(sentences, return_tensors='pt', padding=True).to(device)
            x = transformer(**tokens)["last_hidden_state"][:, 0]
            latents.append(x.cpu().numpy())
        np.save(str(path / f"{name}_{bert_latents}"), np.concatenate(latents, axis=0))
