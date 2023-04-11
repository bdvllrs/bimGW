from typing import Callable, Dict, Literal, Tuple, Union

import torch
from PIL import Image

from bim_gw.datasets.domain import DomainItems

AvailableDomainsType = Literal["v", "attr", "t"]
VisualDataType = Tuple[torch.FloatTensor, Image.Image]
AttributesDataType = Tuple[torch.FloatTensor, int, torch.FloatTensor]
TextDataType = Tuple[torch.FloatTensor, torch.LongTensor, str, Dict[str, int]]
SelectedDomainType = Dict[
    str, Union[VisualDataType, AttributesDataType, TextDataType]]
TransformType = Callable[[DomainItems], DomainItems]
