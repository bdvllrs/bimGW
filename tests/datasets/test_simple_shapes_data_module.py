from pathlib import Path

from bim_gw.datasets.utils import load_simple_shapes_dataset
from bim_gw.utils import get_args

tests_folder = Path(__file__).absolute().parent.parent
dataset_dir = tests_folder.parent / "data" / "shapes"


def get_test_args():
    return get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml"
        ]
    )


def get_datamodule(args):
    args.simple_shapes_path = str(dataset_dir.resolve())
    return load_simple_shapes_dataset(
        args, args.global_workspace
    )


def test_data_module():
    args = get_test_args()
    datamodule = get_datamodule(args)
    datamodule.setup(stage="fit")


def test_filter_sync_domains():
    args = get_test_args()
    datamodule = get_datamodule(args)
    datamodule.setup(stage="fit")
    allowed_indices = list(range(args.datasets.shapes.n_train_examples))
    mapping, domain_mapping = datamodule.filter_sync_domains(allowed_indices)
    assert mapping is None
    assert domain_mapping is None


def test_filter_sync_domains_low_prop_labelled_images():
    args = get_test_args()
    args.global_workspace.prop_labelled_images = 0.1
    datamodule = get_datamodule(args)
    allowed_indices = list(range(args.datasets.shapes.n_train_examples))
    mapping, domain_mapping = datamodule.filter_sync_domains(allowed_indices)
    assert mapping is not None
    assert domain_mapping is not None
