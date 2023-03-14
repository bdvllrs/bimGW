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

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    if args.global_workspace.load_pre_saved_latents is None:
        raise ConfigError(
            "global_workspace.load_pre_saved_latents",
            "This should not be None."
        )

    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = OmegaConf.create(
        [domain for domain in
         args.global_workspace.load_pre_saved_latents.keys()]
    )

    data = load_dataset(args, args.global_workspace, with_actions=True)
    data.prepare_data()
    data.setup(stage="fit")

    domains = get_domains(args, data.img_size)
    for domain in domains.values():
        domain.to(device)
        domain.eval()

    data_loaders = {
        "train": data.train_dataloader(shuffle=False),
        "val": data.val_dataloader()[0],  # only keep in dist dataloaders
        "test": data.test_dataloader()[0]
    }

    for domain_key in domains.keys():
        if domain_key not in args.global_workspace.load_pre_saved_latents:
            raise ConfigError(
                "global_workspace.load_pre_saved_latents",
                f"Domain {domain_key} is not provided."
            )

    path = Path(args.simple_shapes_path) / "saved_latents"
    path.mkdir(exist_ok=True)

    for name, data_loader in data_loaders.items():
        latents = {
            domain_key: None for domain_key in domains.keys()
        }
        print(f"Fetching {name} data.")
        for idx, (batch, target) in tqdm(
                enumerate(data_loader),
                total=int(len(data_loader.dataset) / data_loader.batch_size)
        ):
            for domain_key in domains.keys():
                latent_list = None
                for t in range(len(batch[domain_key])):
                    data = batch[domain_key][t][1:]
                    for k in range(len(data)):
                        if isinstance(data[k], torch.Tensor):
                            data[k] = data[k].to(device)
                    encoded = domains[domain_key].encode(data)
                    if latent_list is None:
                        latent_list = [[] for k in range(len(encoded))]
                    for k in range(len(encoded)):
                        latent_list[k].append(
                            encoded[k].cpu().detach().numpy()
                        )
                if latents[domain_key] is None:
                    latents[domain_key] = [[] for k in range(len(latent_list))]
                for k in range(len(latent_list)):
                    latents[domain_key][k].append(
                        np.stack(latent_list[k], axis=1)
                    )
        for domain_name, latent_list in latents.items():
            (path / name).mkdir(exist_ok=True)
            paths = []
            for k in range(len(latent_list)):
                x = np.concatenate(latent_list[k])
                p = path / name
                p /= args.global_workspace.load_pre_saved_latents[domain_name]
                p = p.with_stem(p.stem + f"_part_{k}")
                paths.append(p.name)
                np.save(str(p), x)
            save_path = path / name
            save_path /= args.global_workspace.load_pre_saved_latents[
                domain_name]
            np.save(
                str(
                    save_path
                ), np.array(paths)
            )
