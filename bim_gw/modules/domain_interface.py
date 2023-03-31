import collections.abc
from typing import Any, Dict, Iterator

from torch import nn

from bim_gw.datasets.domain import DomainItems
from bim_gw.modules.domain_modules import DomainModule


class DomainInterface(nn.Module, collections.abc.Mapping):
    def __init__(
        self, domain_mods: Dict[str, DomainModule]
    ):
        super(DomainInterface, self).__init__()

        self._domain_modules = nn.ModuleDict(domain_mods)
        self._domain_modules.freeze()  # insures that all modules are frozen
        self.names = list(domain_mods.keys())

    def get_specs(self):
        for key, mod in self._domain_modules.items():
            yield key, mod.domain_specs

    def encode(
        self, domains: Dict[str, DomainItems]
    ) -> Dict[str, DomainItems]:
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            out[domain_name] = DomainItems(
                x.available_masks,
                **self._domain_modules[domain_name].encode(
                    x.sub_parts
                )
            )
        return out

    def decode(
        self, domains: Dict[str, DomainItems]
    ) -> Dict[str, DomainItems]:
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            out[domain_name] = DomainItems(
                x.available_masks,
                **self._domain_modules[domain_name].decode(
                    x.sub_parts
                )
            )
        return out

    def adapt(
        self, latents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return {
            domain: self._domain_modules[domain].adapt(latent)
            for domain, latent in latents.items()
        }

    def __getitem__(self, item: str) -> nn.Module:
        return self._domain_modules[item]

    def __len__(self) -> int:
        return len(self._domain_modules)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._domain_modules)
