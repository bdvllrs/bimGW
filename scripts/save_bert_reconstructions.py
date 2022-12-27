import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.gw import split_domains_available_domains
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

    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")
    bert_path = args.global_workspace.bert_path
    path = args.simple_shapes_path

    global_workspace = GlobalWorkspace.load_from_checkpoint(args.checkpoint,
                                                            domain_mods=get_domains(args, len(data.classes),
                                                                                    data.img_size), strict=False)
    global_workspace.eval()

    attr_model = global_workspace.domain_mods["attr"]
    text_model = global_workspace.domain_mods["t"]

    data_loaders = [
        ("train", data.train_dataloader(shuffle=False)),
        ("val", data.val_dataloader()[0]),  # only keep in dist dataloaders
        ("test", data.test_dataloader()[0])
    ]
    path = Path(path)
    for name, data_loader in data_loaders:
        all_latents = []
        print(f"Fetching {name} data.")
        for idx, (batch) in tqdm(enumerate(data_loader),
                                 total=int(len(data_loader.dataset) / data_loader.batch_size)):
            available_domains, domains = split_domains_available_domains(batch)
            latents = global_workspace.encode_uni_modal(domains)
            predictions = global_workspace.adapt(global_workspace.predict(global_workspace.project(latents, ["attr"])))
            all_latents.append(predictions["t"][0].detach().numpy())

        np.save(str(path / f"{name}_predicted_{bert_latents}"), np.concatenate(all_latents, axis=0))

