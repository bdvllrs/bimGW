from typing import Any, Dict, Iterator, TypeVar

import torch
from torch import nn

from bim_gw.datasets.domain import DomainItems

_DomainExamplesType = Dict[str, Dict[str, Dict[str, DomainItems]]]


class DomainBuffer(nn.Module):
    def __init__(
        self,
        domain_examples: DomainItems,
        persistent: bool = True
    ):
        super().__init__()

        self.register_buffer(
            "available_masks",
            domain_examples.available_masks,
            persistent
        )
        self.sub_parts = DictBuffer(domain_examples.sub_parts, persistent)

    def __getitem__(self, item: str) -> Any:
        return self.sub_parts[item]

    def __len__(self) -> int:
        return len(self.sub_parts)

    def __iter__(self) -> Iterator[Any]:
        return self.items()

    def items(self):
        yield from self.sub_parts.items()

    def keys(self):
        yield from self.sub_parts.keys()

    def values(self):
        yield from self.sub_parts.values()


T = TypeVar("T")


class DictBuffer(nn.Module):
    def __init__(
        self,
        buffer_dict: Dict[str, T],
        persistent: bool = True
    ):
        super().__init__()

        buffers: Dict[str, T] = {}
        self._buffer_keys = set()
        self._item_keys = set()
        for key, value in buffer_dict.items():
            if isinstance(value, dict):
                buffers[f"buffer_{key}"] = DictBuffer(value, persistent)
                self._item_keys.add(key)
            elif isinstance(value, DomainItems):
                buffers[f"buffer_{key}"] = DomainBuffer(value, persistent)
                self._item_keys.add(key)
            elif isinstance(value, torch.Tensor):
                self._buffer_keys.add(key)
                self.register_buffer(
                    f"buffer_{key}",
                    value,
                    persistent
                )
            else:
                self._item_keys.add(key)
                setattr(self, f"buffer_{key}", value)
        self._buffer_dict = nn.ModuleDict(buffers)

    def __getitem__(self, item):
        if item in self._buffer_keys:
            return getattr(self, f"buffer_{item}")
        return self._buffer_dict[f"buffer_{item}"]

    def __len__(self) -> int:
        return len(self._buffer_keys) + len(self._item_keys)

    def __iter__(self) -> Iterator[T]:
        return self.items()

    def items(self):
        for key in self._buffer_keys:
            yield key, getattr(self, f"buffer_{key}")
        for key in self._item_keys:
            yield key, self._buffer_dict[f"buffer_{key}"]

    def keys(self):
        yield from iter(self._buffer_keys)
        yield from iter(self._item_keys)

    def values(self):
        for key in self._buffer_keys:
            yield getattr(self, f"buffer_{key}")
        for key in self._item_keys:
            yield self._buffer_dict[f"buffer_{key}"]
