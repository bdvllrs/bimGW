from dataclasses import dataclass, field
from typing import Dict

import pytest
from omegaconf import MISSING, OmegaConf

from bim_gw.utils.cli import parse_argv_from_structure


@dataclass
class NestedConfig:
    param4_1: Dict = field(default_factory=dict)


@dataclass
class StructuredConfig:
    param1: int = 1
    param2: bool = True
    param3: str = MISSING
    param4: NestedConfig = field(default_factory=NestedConfig)


def test_parse_argv_from_structure():
    structure = OmegaConf.structured(StructuredConfig)
    dotlist = parse_argv_from_structure(
        structure,
        ["param1=2", "param2=True"]
    )
    assert dotlist == ["param1=2", "param2=True"]


def test_parse_argv_from_structure_with_flag():
    structure = OmegaConf.structured(StructuredConfig)
    dotlist = parse_argv_from_structure(
        structure,
        ["param2"]
    )
    assert dotlist == ["param2=True"]


def test_parse_argv_from_structure_with_complex_type():
    structure = OmegaConf.structured(StructuredConfig)
    dotlist = parse_argv_from_structure(
        structure,
        ["param4={param4_1: {}}"]
    )
    assert dotlist == ["param4={param4_1: {}}"]


def test_parse_argv_from_structure_wrong_type():
    structure = OmegaConf.structured(StructuredConfig)
    with pytest.raises(ValueError):
        parse_argv_from_structure(
            structure,
            ["param1='ok'"]
        )


def test_parse_argv_from_structure_wrong_type_complex_type():
    structure = OmegaConf.structured(StructuredConfig)
    with pytest.raises(ValueError):
        parse_argv_from_structure(
            structure,
            ["param4={param4_1: 'ok'}"]
        )
