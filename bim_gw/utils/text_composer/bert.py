from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_language_model(
    device: torch.device, bert_path: str
) -> Tuple[BertTokenizer, BertModel]:
    transformer = cast(BertModel, BertModel.from_pretrained(bert_path))
    transformer.eval()
    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad_(False)
    tokenizer = cast(
        BertTokenizer,
        BertTokenizer.from_pretrained(bert_path),
    )
    return tokenizer, transformer


def embed_text(
    tokenizer: BertTokenizer,
    transformer: BertModel,
    sentences: List[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    return transformer(**tokens)["last_hidden_state"][:, 0]


def save_bert_latents(data, bert_path, bert_latents, path, device):
    tokenizer, transformer = get_language_model(device, bert_path)
    data_loaders = [
        ("train", data.train_dataloader(shuffle=False)),
        ("val", data.val_dataloader()[0]),  # only keep in dist dataloaders
        ("test", data.test_dataloader()[0]),
    ]
    path = Path(path)
    for name, data_loader in data_loaders:
        latents = []
        print(f"Fetching {name} data.")
        for idx, (batch) in tqdm(
            enumerate(data_loader),
            total=int(len(data_loader.dataset) / data_loader.batch_size),
        ):
            sentences = batch["t"]["text"]
            x = embed_text(tokenizer, transformer, sentences, device)
            latents.append(x.cpu().numpy())
        all_latents = np.concatenate(latents, axis=0)
        np.save(str(path / f"{name}_{bert_latents}"), all_latents)

        # Save latent statistics
        if name == "train":
            mean = all_latents.mean(axis=0)
            std = all_latents.std(axis=0)
            np.save(str(path / f"mean_{bert_latents}"), mean)
            np.save(str(path / f"std_{bert_latents}"), std)
