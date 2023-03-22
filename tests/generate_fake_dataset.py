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

    if dataset_dir.is_dir():
        dataset_dir.rmdir()

    args = get_args(
        cli=False,
        use_local=False,
        additional_config_files=[test_config_dir / "test_base.yaml"],
    )

    data_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    args.simple_shapes_path = str(dataset_dir.resolve())

    create_shape_dataset(args)

    (dataset_dir / "saved_latents").mkdir(exist_ok=True)
    for split in ["train", "test", "val"]:
        latent_dir = dataset_dir / "saved_latents" / split
        latent_dir.mkdir(exist_ok=True)
        n_examples = args.datasets.shapes[f"n_{split}_examples"]

        latent = np.random.randn(n_examples, args.vae.z_size)
        np.save(latent_dir / "fake_latents.npy", latent)


if __name__ == "__main__":
    main()
