import torch
from torch.utils.data.dataloader import default_collate


class DomainItem:
    def __init__(self, is_available=True, **items):
        self.available_mask = torch.tensor(is_available).float()
        self.items = items


class DomainItems:
    def __init__(self, available_masks, items):
        self.available_masks = available_masks
        self._items = items

    def __getattr__(self, key):
        return self._items[key]

    def __setattr__(self, key, value):
        self._items[key] = value

    def items(self):
        yield from self._items.items()

    def keys(self):
        yield from self._items.keys()

    def values(self):
        yield from self._items.values()

    @staticmethod
    def get_collate_fn():
        return domain_collate_fn


def domain_collate_fn(batch):
    items = [(item.available_mask, item.items) for item in batch]
    stacked_items = default_collate(items)
    return DomainItems(stacked_items[0], stacked_items[1])


def collate_fn(batch):
    items = {domain_name: [] for domain_name in batch[0].keys()}
    for item in batch:
        for domain_name, domain_item in item.items():
            items[domain_name].append(domain_item)
    return {
        domain_name: domain_collate_fn(items[domain_name])
        for domain_name in items.keys()
    }
