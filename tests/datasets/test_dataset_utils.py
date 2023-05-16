from pathlib import Path

from bim_gw.datasets.utils import load_simple_shapes_dataset
from bim_gw.utils import get_args

tests_folder = Path(__file__).absolute().parent.parent
dataset_dir = tests_folder.parent / "data" / "shapes"


def test_load_simple_shapes_dataset_vae_local():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[tests_folder / "configs/test_base.yaml"],
    )
    args.simple_shapes_path = str(dataset_dir.resolve())
    load_simple_shapes_dataset(args, args.vae, selected_domains=["v"])


def test_load_simple_shapes_dataset_lm_local():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[tests_folder / "configs/test_base.yaml"],
    )
    args.simple_shapes_path = str(dataset_dir.resolve())
    load_simple_shapes_dataset(
        args, args.lm, add_unimodal=False, selected_domains=["t", "attr"]
    )


def test_load_simple_shapes_dataset_gw_local():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[tests_folder / "configs/test_base.yaml"],
    )
    args.simple_shapes_path = str(dataset_dir.resolve())
    load_simple_shapes_dataset(args, args.global_workspace)
