import numpy as np


def load_pre_saved_latent(root_path, split, pre_saved_latent_path, domain_key, ids=None):
    if ids is None:
        ids = slice(None)
    root_path = root_path / "saved_latents" / split / pre_saved_latent_path[domain_key]
    data = np.load(str(root_path))
    if data.ndim == 1 and isinstance(data[0], np.str):
        d = []
        for path in data:
            d.append(np.load(str(root_path / "saved_latents" / split / path))[ids])
        return d
    else:
        return [data[ids]]
