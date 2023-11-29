import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.utils import get_args
from bim_gw.utils.errors import ConfigError
from bim_gw.utils.scripts import get_domains

domain_item_name_mapping = {
    "v": ["z_img"],
    "attr": ["z_cls", "z_attr"],
    "t": ["z"],
}

if __name__ == "__main__":
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))
    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.0
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True
    args.global_workspace.ood_idx_domain = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = OmegaConf.create(
        [
            domain
            for domain in args.global_workspace.load_pre_saved_latents.keys()
        ]
    )

    root_path = Path(args.simple_shapes_path)

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    domains = get_domains(args, data.img_size)
    for domain in domains.values():
        domain.to(device)
        domain.eval()

    data_loaders = {
        "val": data.val_dataloader()[0],  # only keep in dist dataloaders
        "test": data.test_dataloader()[0],
        "train": data.train_dataloader(shuffle=False),
    }

    for domain_key in domains.keys():
        if domain_key not in args.global_workspace.load_pre_saved_latents:
            raise ConfigError(
                "global_workspace.load_pre_saved_latents",
                f"Domain {domain_key} is not provided.",
            )

    path = root_path / "saved_latents"
    path.mkdir(exist_ok=True)

    for name, data_loader in data_loaders.items():
        latents = {domain_key: None for domain_key in domains.keys()}
        print(f"Fetching {name} data.")
        for idx, batch in tqdm(
            enumerate(data_loader),
            total=int(len(data_loader.dataset) / data_loader.batch_size),
        ):
            for domain_key in domains.keys():
                batch[domain_key].to_device(device)
                encoded = domains[domain_key].encode(
                    batch[domain_key].sub_parts
                )
                encoded = [
                    encoded[key].cpu().detach().numpy()
                    for key in domain_item_name_mapping[domain_key]
                ]
                if latents[domain_key] is None:
                    latents[domain_key] = [[] for _ in range(len(encoded))]
                for k, e in enumerate(encoded):
                    latents[domain_key][k].append(e)
        for domain_name, latent_list in latents.items():
            (path / name).mkdir(exist_ok=True)
            paths = []
            for k in range(len(latent_list)):
                x = np.concatenate(latent_list[k], axis=0)
                x = np.expand_dims(x, axis=1)
                p = path / name
                p /= args.global_workspace.load_pre_saved_latents[domain_name]
                p = p.parent / (p.stem + f"_part_{k}" + p.suffix)
                paths.append(p.name)
                np.save(str(p), x)
            save_path = path / name
            save_path /= args.global_workspace.load_pre_saved_latents[
                domain_name
            ]
            np.save(str(save_path), np.array(paths))
