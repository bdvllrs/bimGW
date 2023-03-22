import shutil
from pathlib import Path

import numpy as np

from bim_gw.utils import get_args
from scripts.create_shape_dataset import main as create_shape_dataset


def main():
    root_dir = Path(__file__).absolute().parent.parent
    data_dir = root_dir / "data"
    dataset_dir = data_dir / "shapes"
    tests_dir = root_dir / "tests"
    test_config_dir = tests_dir / "configs"
    config_dir = root_dir / "config"

    shutil.copy(
        test_config_dir / "test_base.yaml", config_dir / "local.yaml"
    )

    args = get_args(cli=False)

    data_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    create_shape_dataset()
    bert_latents_file = args.fetchers.t.bert_latents

    (dataset_dir / "saved_latents").mkdir(exist_ok=True)
    for split in ["train", "test", "val"]:
        latent_dir = dataset_dir / "saved_latents" / split
        latent_dir.mkdir(exist_ok=True)
        n_examples = args.datasets.shapes[f"n_{split}_examples"]

        latent = np.randn(n_examples, args.vae.z_size)
        np.save(latent_dir / "fake_latents.npy", latent)

        bert_latents = np.randn(n_examples, 768)
        np.save(dataset_dir / f"{split}_{bert_latents_file}", bert_latents)

    mean = np.zeros((768,))
    np.save(dataset_dir / f"mean_{bert_latents_file}", mean)
    std = np.ones((768,))
    np.save(dataset_dir / f"std_{bert_latents_file}", std)


if __name__ == "__main__":
    main()
