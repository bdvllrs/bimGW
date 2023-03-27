import logging
from typing import List, Tuple

import numpy as np
import torch

from bim_gw.datasets.domain import collate_fn
from bim_gw.utils import registries


@registries.register_dataset("shapes")
def load_simple_shapes_dataset(args, local_args, **kwargs):
    from .simple_shapes.data_modules import SimpleShapesDataModule

    print("Loading Shapes.")
    pre_saved_latent_paths = None
    sync_uses_whole_dataset = False
    if local_args.get("use_pre_saved", False):
        pre_saved_latent_paths = args.global_workspace.load_pre_saved_latents
    if local_args.get("sync_uses_whole_dataset", False):
        sync_uses_whole_dataset = True
    selected_domains = local_args.get("selected_domains", None) or kwargs.get(
        "selected_domains", None
    )
    if "selected_domains" in kwargs:
        del kwargs["selected_domains"]
    return SimpleShapesDataModule(
        args.simple_shapes_path, local_args.batch_size,
        args.dataloader.num_workers,
        local_args.get("prop_labelled_images", 1.),
        local_args.get("prop_available_images", 1.),
        local_args.get("remove_sync_domains", None),
        args.n_validation_examples, local_args.get("split_ood", False),
        selected_domains,
        pre_saved_latent_paths,
        sync_uses_whole_dataset,
        domain_loader_params=args.domain_loader,
        len_train_dataset=args.datasets.shapes.n_train_examples,
        **kwargs
    )


# @registries.register_dataset("cmu_mosei")
def load_cmu_mosei_dataset(args, local_args, **kwargs):
    from bim_gw.datasets.cmu_mosei.data_module import CMUMOSEIDataModule

    # TODO: finish cmu_mosei. But how to handle sequences?
    print("Loading CMU MOSEI.")
    return CMUMOSEIDataModule(
        args.cmu_mosei.path, local_args.batch_size,
        args.dataloader.num_workers,
        local_args.selected_domains, args.cmu_mosei.validate,
        args.cmu_mosei.seq_length
    )


def load_dataset(args, local_args, **kwargs):
    try:
        dataset = registries.get_dataset(args.current_dataset)
    except KeyError:
        raise ValueError("The requested dataset is not implemented.")
    return dataset(args, local_args, **kwargs)


def get_lm(args, data, **kwargs):
    raise NotImplementedError("Use get_domains instead.")


def filter_sync_domains(
    domains: List[str],
    allowed_indices: List[int],
    prop_labelled_images: float,
    prop_available_images: float,
) -> Tuple[List[int], List[List[str]]]:
    # permute for number of couples of domains
    permuted_indices = np.random.permutation(allowed_indices)
    logging.debug(f"Loaded {len(allowed_indices)} examples in train set.")

    prop_2_domains = prop_labelled_images / prop_available_images
    original_size = int(
        len(allowed_indices) * prop_available_images
    )

    if prop_2_domains == 1 and prop_available_images == 1:
        return None, None

    sync_split = int(prop_2_domains * original_size)
    sync_items = permuted_indices[:sync_split]
    rest = permuted_indices[sync_split:]

    mapping = []
    domain_mapping = []
    if prop_2_domains < 1:
        labelled_size = int(original_size * prop_2_domains)
        n_repeats = ((len(domains) * original_size) // labelled_size +
                     int(original_size % labelled_size > 0))

        domain_items = np.tile(sync_items, n_repeats)
        mapping.extend(domain_items)
        domain_mapping.extend(
            [domains] * len(domain_items)
        )

    unsync_domain_items = permuted_indices
    if prop_available_images < 1:
        n_unsync = int(
            prop_available_images * len(allowed_indices)
        ) - sync_split
        unsync_items = rest[:n_unsync]
        unsync_domain_items = np.concatenate((unsync_items, sync_items))
    mapping.extend(unsync_domain_items)
    domain_mapping.extend([[domains[0]]] * len(unsync_domain_items))
    mapping.extend(unsync_domain_items)
    domain_mapping.extend([[domains[1]]] * len(unsync_domain_items))

    return mapping, domain_mapping


def get_validation_examples(
    train_set, val_set, test_set,
    n_domain_examples,
):
    reconstruction_indices = {
        "train": {
            "in_dist": torch.randint(
                len(train_set), size=(n_domain_examples,)
            ),
        },
        "val": {
            "in_dist": torch.randint(
                len(val_set["in_dist"]), size=(n_domain_examples,)
            ),

        },
        "test": {
            "in_dist": torch.randint(
                len(test_set["in_dist"]), size=(n_domain_examples,)
            ),
        }
    }

    if val_set["ood"] is not None:
        reconstruction_indices["val"]["ood"] = torch.randint(
            len(val_set["ood"]),
            size=(n_domain_examples,)
        )
    if test_set["ood"] is not None:
        reconstruction_indices["test"]["ood"] = torch.randint(
            len(test_set["ood"]),
            size=(n_domain_examples,)
        )

    domain_examples = {}

    all_sets = [
        ("train", {"in_dist": train_set}),
        ("val", val_set),
        ("test", test_set)
    ]

    for set_name, used_set in all_sets:
        domain_examples[set_name] = {}
        for used_dist in reconstruction_indices[set_name].keys():
            dist_indices = reconstruction_indices[set_name][used_dist]
            examples = collate_fn(
                [used_set[used_dist][i] for i in dist_indices]
            )
            domain_examples[set_name][used_dist] = examples

    return domain_examples
