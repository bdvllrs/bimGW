import pathlib
from typing import Dict, List, Optional, SupportsIndex

import numpy as np

from bim_gw.datasets.simple_shapes.types import AvailableDomainsType


def load_pre_saved_latent(
    root_path: pathlib.Path, split: str,
    pre_saved_latent_path: Dict[AvailableDomainsType, str],
    domain_key: AvailableDomainsType,
    ids: Optional[SupportsIndex] = None
) -> List[np.ndarray]:
    kept_idx = ids if ids is None else slice(None)
    latent_dir = root_path / "saved_latents" / split
    path = latent_dir / pre_saved_latent_path[domain_key]
    data = np.load(str(path))
    if data.ndim == 1 and isinstance(data[0], str):
        d = []
        for path in data:
            d.append(np.load(str(latent_dir / path))[kept_idx])
        return d
    else:
        return [data[kept_idx]]
