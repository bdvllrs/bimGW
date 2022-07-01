import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, BertTokenizer

from bim_gw.datasets import load_dataset
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    assert args.global_workspace.load_pre_saved_latents is not None, "Pre-saved latent path should be defined."

    args.seed = 0
    bert_latents = args.fetchers.t.bert_latents
    args.fetchers.t.bert_latents = None
    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = {"t": "t"}

    data = load_dataset(args, args.global_workspace, with_actions=True, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    transformer = BertModel.from_pretrained(args.global_workspace.bert_path)
    transformer.eval()
    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(args.global_workspace.bert_path)

    data_loaders = [
        ("train", data.train_dataloader(shuffle=False)),
        ("val", data.val_dataloader()[0]),  # only keep in dist dataloaders
        ("test", data.test_dataloader()[0])
    ]

    path = Path(args.simple_shapes_path) / "saved_latents"
    path.mkdir(exist_ok=True)

    for name, data_loader in data_loaders:
        latents = []
        print(f"Fetching {name} data.")
        for idx, (batch, target) in tqdm(enumerate(data_loader),
                                         total=int(len(data_loader.dataset) / data_loader.batch_size)):

            sentences = batch["t"][0][2]
            tokens = tokenizer(sentences, return_tensors='pt', padding=True).to(device)
            x = transformer(**tokens)["last_hidden_state"][:, 0]
            latents.append(x.cpu().numpy())
        (path / name).mkdir(exist_ok=True)
        np.save(str(path / name / bert_latents), np.concatenate(latents, axis=0))
