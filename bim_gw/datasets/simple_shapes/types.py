from enum import auto
from typing import Dict

from bim_gw.datasets.domain import DomainItems
from bim_gw.utils.types import AvailableDomains


class ShapesAvailableDomains(AvailableDomains):
    v = auto()
    t = auto()
    attr = auto()


SelectedDomainType = Dict[ShapesAvailableDomains, DomainItems]
