import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    assert args.global_workspace.load_pre_saved_latents is not None, "Pre-saved latent path should be defined."

    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.sync_uses_whole_dataset = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    domains = get_domains(args, data)
    for domain in domains.values():
        domain.to(device)
        domain.eval()

    data_loaders = {
        "train": data.train_dataloader(shuffle=False)["sync_"],
        "val": data.val_dataloader()[0],  # only keep in dist dataloaders
        "test": data.test_dataloader()[0]
    }

    for domain_key in domains.keys():
        domain_name = args.global_workspace.selected_domains[domain_key]
        assert domain_name in args.global_workspace.load_pre_saved_latents, f"Path for domain {domain_name} is not provided."

    path = Path(args.simple_shapes_path) / "saved_latents"
    path.mkdir(exist_ok=True)

    for name, data_loader in data_loaders.items():
        latents = {
            args.global_workspace.selected_domains[domain_key]: [] for domain_key in domains.keys()
        }
        print(f"Fetching {name} data.")
        for idx, batch in tqdm(enumerate(data_loader),
                               total=int(len(data_loader.dataset) / data_loader.batch_size)):
            for domain_key, data in batch.items():
                domain_name = args.global_workspace.selected_domains[domain_key]
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                latents[domain_name].append(domains[domain_key].encode(data).cpu().detach().numpy())
        for domain_name, l in latents.items():
            (path / name).mkdir(exist_ok=True)
            np.save(str(path / name / args.global_workspace.load_pre_saved_latents[domain_name]), np.concatenate(l))
