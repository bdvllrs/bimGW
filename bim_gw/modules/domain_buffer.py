from typing import Dict, Iterator, TypeVar, Union

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

        self.sub_parts_keys = []

        self.register_buffer(
            f"available_masks",
            domain_examples.available_masks,
            persistent
        )
        for key, value in domain_examples.sub_parts.items():
            self.sub_parts_keys.append(key)
            self.register_buffer(
                f"sub_parts_{key}",
                value,
                persistent
            )

    @property
    def sub_parts(self) -> Dict[str, torch.Tensor]:
        return {
            key: getattr(self, f"sub_parts_{key}")
            for key in self.sub_parts_keys
        }

    def __getitem__(self, item: str) -> torch.Tensor:
        return getattr(self, f"sub_parts_{item}")

    def __len__(self) -> int:
        return len(self.sub_parts_keys)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.items()

    def items(self):
        for key in self.sub_parts_keys:
            yield key, getattr(self, f"sub_parts_{key}")

    def keys(self):
        yield from iter(self.sub_parts_keys)

    def values(self):
        for key in self.sub_parts_keys:
            yield getattr(self, f"sub_parts_{key}")


T = TypeVar("T")


class DictBuffer(nn.Module):
    def __init__(
        self,
        domain_examples: Dict[str, T],
        persistent: bool = True
    ):
        super().__init__()

        examples: Dict[str, Union[DictBuffer, DomainBuffer]] = {}
        self.buffer_keys = []
        for key, value in domain_examples.items():
            if isinstance(value, dict):
                examples[f"buffer_{key}"] = DictBuffer(value, persistent)
                self.buffer_keys.append(key)
            elif isinstance(value, DomainItems):
                examples[f"buffer_{key}"] = DomainBuffer(value, persistent)
                self.buffer_keys.append(key)
        self.domain_examples = nn.ModuleDict(examples)

    def __getitem__(self, item):
        return self.domain_examples[f"buffer_{item}"]

    def __len__(self) -> int:
        return len(self.domain_examples)

    def __iter__(self) -> Iterator[T]:
        return self.items()

    def items(self):
        for key in self.buffer_keys:
            yield key, self.domain_examples[f"buffer_{key}"]

    def keys(self):
        yield from iter(self.buffer_keys)

    def values(self):
        for key in self.buffer_keys:
            yield self.domain_examples[f"buffer_{key}"]
