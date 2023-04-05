import random

import torch

from bim_gw.datasets.domain import collate_fn, DomainItems


def test_collate_domain_items():
    batch_size = 16

    items = [
        DomainItems.singular(
            attr1=torch.randn(3),
            attr2=torch.randn(8),
            is_available=random.randint(0, 1)
        )

        for _ in range(batch_size)
    ]

    domain_collate_fn = DomainItems.get_collate_fn()
    collated_items = domain_collate_fn(items)
    assert isinstance(collated_items, DomainItems)
    assert collated_items.is_batch()
    assert collated_items.available_masks.ndim == 1
    assert collated_items.available_masks.size(0) == batch_size

    assert len(list(collated_items.items())) == 2
    assert "attr1" in collated_items.keys()
    assert "attr2" in collated_items.keys()
    assert collated_items['attr1'].ndim == 2
    assert collated_items['attr1'].size(0) == batch_size
    assert collated_items['attr1'].size(1) == 3
    assert collated_items['attr2'].ndim == 2
    assert collated_items['attr2'].size(0) == batch_size
    assert collated_items['attr2'].size(1) == 8


def test_collate_fn():
    batch_size = 16

    items = [
        {
            "d1": DomainItems.singular(
                attr1=torch.randn(3),
                attr2=torch.randn(8),
                is_available=random.randint(0, 1)
            ),
            "d2": DomainItems.singular(
                attr3=torch.randn(5),
                is_available=random.randint(0, 1)
            ),
        }
        for _ in range(batch_size)
    ]

    collated_items = collate_fn(items)

    assert isinstance(collated_items, dict)
    assert "d1" in collated_items.keys()
    assert "d2" in collated_items.keys()
    assert isinstance(collated_items["d1"], DomainItems)
    assert isinstance(collated_items["d2"], DomainItems)
