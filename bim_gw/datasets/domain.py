import torch
from torch.utils.data.dataloader import default_collate


class DomainItem:
    def __init__(self, is_available=True, **sub_parts):
        self.available_mask = torch.tensor(is_available).float()
        self._items = sub_parts

    def __getattr__(self, key):
        return self._items[key]

    def __getitem__(self, item):
        return self._items[item]

    def update(self, key, other):
        self._items[key] = other

    def get_sub_parts(self):
        return self._items


class DomainItems:
    def __init__(self, batched_available_masks, batched_sub_parts):
        self.available_masks = batched_available_masks
        self._sub_parts = batched_sub_parts

    def items(self):
        yield from self._sub_parts.items()

    def keys(self):
        yield from self._sub_parts.keys()

    def values(self):
        yield from self._sub_parts.values()

    def __getattr__(self, key):
        return self._sub_parts[key]

    def __getitem__(self, item):
        return self._sub_parts[item]

    @staticmethod
    def get_collate_fn():
        return domain_collate_fn


def domain_collate_fn(batch):
    items = [(item.available_mask, item.get_sub_parts()) for item in batch]
    batched_items = default_collate(items)
    return DomainItems(batched_items[0], batched_items[1])


def collate_fn(batch):
    items = {domain_name: [] for domain_name in batch[0].keys()}
    for item in batch:
        for domain_name, domain_item in item.items():
            items[domain_name].append(domain_item)
    return {
        domain_name: domain_collate_fn(items[domain_name])
        for domain_name in items.keys()
    }
