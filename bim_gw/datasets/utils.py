import logging
from math import ceil
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bim_gw.datasets.domain import collate_fn, DomainItems
from bim_gw.utils import registries
from bim_gw.utils.types import AvailableDomains, DistLiteral, SplitLiteral


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
        args.simple_shapes_path,
        local_args.batch_size,
        args.dataloader.num_workers,
        local_args.get("prop_labelled_images", 1.0),
        local_args.get("prop_available_images", 1.0),
        local_args.get("remove_sync_domains", None),
        args.n_validation_examples,
        local_args.get("split_ood", False),
        selected_domains,
        pre_saved_latent_paths,
        sync_uses_whole_dataset,
        domain_loader_params=args.domain_loader,
        len_train_dataset=args.datasets.shapes.n_train_examples,
        **kwargs,
    )


# @registries.register_dataset("cmu_mosei")
def load_cmu_mosei_dataset(args, local_args, **kwargs):
    from bim_gw.datasets.cmu_mosei.data_module import CMUMOSEIDataModule

    # TODO: finish cmu_mosei. But how to handle sequences?
    print("Loading CMU MOSEI.")
    return CMUMOSEIDataModule(
        args.cmu_mosei.path,
        local_args.batch_size,
        args.dataloader.num_workers,
        local_args.selected_domains,
        args.cmu_mosei.validate,
        args.cmu_mosei.seq_length,
    )


def load_dataset(args, local_args, **kwargs):
    try:
        dataset = registries.get_dataset(args.current_dataset)
    except KeyError:
        raise ValueError("The requested dataset is not implemented.")
    return dataset(args, local_args, **kwargs)


def get_lm(args, data, **kwargs):
    raise NotImplementedError("Use get_domains instead.")


DomainMappingType = Sequence[Sequence[AvailableDomains]]


def filter_sync_domains(
    domains: Sequence[AvailableDomains],
    allowed_indices: List[int],
    prop_labelled_images: float,
    prop_available_images: float,
) -> Tuple[Optional[List[int]], Optional[DomainMappingType]]:
    # permute for number of couples of domains
    permuted_indices = np.random.permutation(allowed_indices)
    logging.debug(f"Loaded {len(allowed_indices)} examples in train set.")

    prop_2_domains = prop_labelled_images / prop_available_images
    original_size = int(len(allowed_indices) * prop_available_images)

    if prop_2_domains == 1 and prop_available_images == 1:
        return None, None

    sync_split = int(prop_2_domains * original_size)
    sync_items = permuted_indices[:sync_split]
    rest = permuted_indices[sync_split:]

    mapping: List[int] = []
    domain_mapping = []
    if prop_2_domains < 1:
        labelled_size = int(original_size * prop_2_domains)
        n_repeats = ceil((len(domains) * len(allowed_indices))
                              / labelled_size)

        domain_items = np.tile(sync_items, n_repeats)
        mapping.extend(domain_items)
        domain_mapping.extend([domains] * len(domain_items))

    unsync_domain_items = permuted_indices
    if prop_available_images < 1:
        n_unsync = (
            int(prop_available_images * len(allowed_indices)) - sync_split
        )
        unsync_items = rest[:n_unsync]
        unsync_domain_items = np.concatenate((unsync_items, sync_items))
        n_repeats = ceil(len(allowed_indices) / len(unsync_domain_items))
        unsync_domain_items = np.tile(unsync_domain_items, n_repeats)
    mapping.extend(unsync_domain_items)
    domain_mapping.extend([[domains[0]]] * len(unsync_domain_items))
    mapping.extend(unsync_domain_items)
    domain_mapping.extend([[domains[1]]] * len(unsync_domain_items))

    return mapping, domain_mapping


DomainExamplesType = Mapping[
    SplitLiteral, Mapping[DistLiteral, Mapping[AvailableDomains, DomainItems]]
]


def get_validation_examples(
    datasets: Mapping[SplitLiteral, Mapping[DistLiteral, Dataset]],
    n_domain_examples: int,
) -> DomainExamplesType:
    reconstruction_indices = {
        split: {
            dist: torch.randint(len(dataset[dist]), size=(n_domain_examples,)),
        }
        for split, dataset in datasets.items()
        for dist in dataset.keys()
        if dataset[dist] is not None
    }

    domain_examples: Dict[
        SplitLiteral, Dict[DistLiteral, Dict[AvailableDomains, DomainItems]]
    ] = {}

    all_sets = [(split, dataset) for split, dataset in datasets.items()]

    for set_name, used_set in all_sets:
        domain_examples[set_name] = {}
        for used_dist in reconstruction_indices[set_name].keys():
            dist_indices = reconstruction_indices[set_name][used_dist]
            examples = collate_fn(
                [used_set[used_dist][i] for i in dist_indices]
            )
            domain_examples[set_name][used_dist] = examples

    return domain_examples
