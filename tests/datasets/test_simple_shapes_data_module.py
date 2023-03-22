from pathlib import Path

import pytest

from bim_gw.datasets.utils import load_simple_shapes_dataset
from bim_gw.utils import get_args

tests_folder = Path(__file__).absolute().parent.parent
dataset_dir = tests_folder.parent / "data" / "shapes"


def get_test_args():
    return get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/test_gw_with_text.yaml",
        ]
    )


def get_datamodule(args):
    args.simple_shapes_path = str(dataset_dir.resolve())
    return load_simple_shapes_dataset(
        args, args.global_workspace
    )


def compute_counts(domain_mapping):
    counts = {}
    for key in domain_mapping:
        key = "".join(sorted(key))
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
    return counts


def compute_uniques(mapping, domain_mapping):
    uniques = {}
    for key, val in zip(domain_mapping, mapping):
        key = "".join(sorted(key))
        if key not in uniques:
            uniques[key] = set()
        uniques[key].add(val)
    return {key: len(val) for key, val in uniques.items()}


def check_domain_mapping(domain_mapping, expected_counts):
    effective_counts = compute_counts(domain_mapping)
    for key, count in expected_counts.items():
        assert effective_counts[key] == count


def check_mapping(mapping, domain_mapping, expected_unique):
    effective_uniques = compute_uniques(mapping, domain_mapping)
    for key, n_unique in expected_unique.items():
        assert effective_uniques[key] == n_unique


def test_data_module():
    args = get_test_args()
    datamodule = get_datamodule(args)
    datamodule.setup(stage="fit")


def test_data_module_without_selected_domains():
    args = get_test_args()
    args.global_workspace.selected_domains = []
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


def test_filter_sync_domains_nonzero_prop_labelled_images():
    args = get_test_args()
    args.global_workspace.prop_labelled_images = 0.1
    datamodule = get_datamodule(args)
    allowed_indices = list(range(args.datasets.shapes.n_train_examples))
    mapping, domain_mapping = datamodule.filter_sync_domains(allowed_indices)

    n_train_examples = len(allowed_indices)
    n_sync_examples = int(
        args.global_workspace.prop_labelled_images
        * n_train_examples
    )
    expected_counts = {
        "v": n_train_examples,
        "t": n_train_examples,
        "tv": 2 * n_train_examples,
    }
    expected_unique = {
        "v": n_train_examples,
        "t": n_train_examples,
        "tv": n_sync_examples,
    }
    check_domain_mapping(domain_mapping, expected_counts)
    check_mapping(mapping, domain_mapping, expected_unique)


def test_filter_sync_domains_nonzero_prop_available_images_fail():
    with pytest.raises(ValueError):
        args = get_test_args()
        args.global_workspace.prop_available_images = 0.5
        args.global_workspace.prop_labelled_images = 1
        datamodule = get_datamodule(args)
        datamodule.setup(stage="fit")


def test_filter_sync_domains_nonzero_prop_available_images():
    args = get_test_args()
    args.global_workspace.prop_available_images = 0.4
    args.global_workspace.prop_labelled_images = 0.1
    datamodule = get_datamodule(args)
    allowed_indices = list(range(args.datasets.shapes.n_train_examples))
    mapping, domain_mapping = datamodule.filter_sync_domains(allowed_indices)

    n_train_examples = int(
        args.global_workspace.prop_available_images * len(allowed_indices)
    )
    n_sync_examples = int(
        args.global_workspace.prop_labelled_images
        * len(allowed_indices)
    )
    expected_counts = {
        "v": n_train_examples,
        "t": n_train_examples,
        "tv": 2 * n_train_examples,
    }
    expected_unique = {
        "v": n_train_examples,
        "t": n_train_examples,
        "tv": n_sync_examples,
    }
    check_domain_mapping(domain_mapping, expected_counts)
    check_mapping(mapping, domain_mapping, expected_unique)
