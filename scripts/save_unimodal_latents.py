import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    assert args.global_workspace.load_pre_saved_latents is not None, "Pre-saved latent path should be defined."

    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = OmegaConf.create([domain for domain in args.global_workspace.load_pre_saved_latents.keys()])

    data = load_dataset(args, args.global_workspace, with_actions=True)
    data.prepare_data()
    data.setup(stage="fit")

    domains = get_domains(args, len(data.classes), data.img_size)
    for domain in domains.values():
        domain.to(device)
        domain.eval()

    data_loaders = {
        "train": data.train_dataloader(shuffle=False),
        "val": data.val_dataloader()[0],  # only keep in dist dataloaders
        "test": data.test_dataloader()[0]
    }

    for domain_key in domains.keys():
        assert domain_key in args.global_workspace.load_pre_saved_latents, f"Path for domain {domain_key} is not provided."

    path = Path(args.simple_shapes_path) / "saved_latents"
    path.mkdir(exist_ok=True)

    for name, data_loader in data_loaders.items():
        latents = {
            domain_key: None for domain_key in domains.keys()
        }
        print(f"Fetching {name} data.")
        for idx, (batch, target) in tqdm(enumerate(data_loader),
                                         total=int(len(data_loader.dataset) / data_loader.batch_size)):
            for domain_key in domains.keys():
                l = None
                for t in range(len(batch[domain_key])):
                    data = batch[domain_key][t][1:]
                    for k in range(len(data)):
                        if isinstance(data[k], torch.Tensor):
                            data[k] = data[k].to(device)
                    encoded = domains[domain_key].encode(data)
                    if l is None:
                        l = [[] for k in range(len(encoded))]
                    for k in range(len(encoded)):
                        l[k].append(encoded[k].cpu().detach().numpy())
                if latents[domain_key] is None:
                    latents[domain_key] = [[] for k in range(len(l))]
                for k in range(len(l)):
                    latents[domain_key][k].append(np.stack(l[k], axis=1))
        for domain_name, l in latents.items():
            (path / name).mkdir(exist_ok=True)
            paths = []
            for k in range(len(l)):
                x = np.concatenate(l[k])
                p = path / name / args.global_workspace.load_pre_saved_latents[domain_name]
                p = p.with_stem(p.stem + f"_part_{k}")
                paths.append(p.name)
                np.save(str(p), x)
            np.save(str(path / name / args.global_workspace.load_pre_saved_latents[domain_name]), np.array(paths))
