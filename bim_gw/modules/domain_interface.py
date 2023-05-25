from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple

from torch import nn

from bim_gw.datasets.domain import DomainItems
from bim_gw.modules.domain_modules import DomainModule, PassThroughWM
from bim_gw.modules.domain_modules.domain_module import DomainSpecs


class DomainInterface(nn.Module):
    def __init__(self, domain_mods: Dict[str, DomainModule]):
        super(DomainInterface, self).__init__()

        self._domain_modules = nn.ModuleDict(
            domain_mods  # consider it a mapping for the editor # type: ignore
        )
        for param in self._domain_modules.parameters():
            param.requires_grad = False
        self._domain_modules.eval()
        self.names = list(domain_mods.keys())

    def get_specs(self) -> Iterator[Tuple[str, DomainSpecs]]:
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
                **self._domain_modules[domain_name].encode(x.sub_parts)
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
                **self._domain_modules[domain_name].decode(x.sub_parts)
            )
        return out

    def adapt(
        self, latents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return {
            domain: self._domain_modules[domain].adapt(latent)
            for domain, latent in latents.items()
        }

    def set_pass_through(self, mode: bool = True):
        for domain_mod in self._domain_modules.values():
            if isinstance(domain_mod, PassThroughWM):
                domain_mod.pass_through(mode)

    @contextmanager
    def pass_through(self, mode: bool = True):
        self.set_pass_through(mode)
        yield
        self.set_pass_through(not mode)

    def __getitem__(self, item: str) -> DomainModule:
        return self._domain_modules[item]

    def __len__(self) -> int:
        return len(self._domain_modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._domain_modules)
