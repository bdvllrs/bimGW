from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pytest
from omegaconf import MISSING

from bim_gw.utils.cli import parse_argv_from_dataclass


@dataclass
class NestedConfig:
    param4_1: Dict = field(default_factory=dict)


@dataclass
class StructuredConfig:
    param1: int = 1
    param2: bool = True
    param3: str = MISSING
    param4: NestedConfig = field(default_factory=NestedConfig)
    param5: str = field(default="nok", metadata={"cli_names": ["--param5"]})
    param6: List[NestedConfig] = field(
        default_factory=lambda: [NestedConfig(param4_1={})]
    )
    param7: Any = "ok"
    param8: Optional[int] = None
    param9: Union[float, int] = field(default=42)


def test_parse_argv_from_structure():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param1=2", "param2=True"]
    )
    assert dotlist == ["param1=2", "param2=True"]


def test_parse_argv_from_structure_with_flag():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param2", "param1=2"]
    )
    assert dotlist == ["param2=True", "param1=2"]


def test_parse_argv_from_structure_with_flag_last():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param2"]
    )
    assert dotlist == ["param2=True"]


def test_parse_argv_from_structure_with_complex_type():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param4={param4_1: {}}"]
    )
    assert dotlist == ["param4={param4_1: {}}"]


def test_parse_argv_from_structure_with_complex_type_list():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param6=[{param4_1: {}}]"]
    )
    assert dotlist == ["param6=[{param4_1: {}}]"]


def test_parse_argv_from_structure_with_complex_type_list_fail():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param6=[{param4_1: nok}]"]
        )


def test_parse_argv_from_structure_with_dot_notation():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param4.param4_1={}"]
    )
    assert dotlist == ["param4.param4_1={}"]


def test_parse_argv_from_structure_with_dot_notation_fail():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param4.param4_1=nok"]
        )


def test_parse_argv_from_structure_wrong_type():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param1='ok'"]
        )


def test_parse_argv_from_structure_wrong_type_complex_type():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param4={param4_1: 'ok'}"]
        )


def test_parse_argv_from_structure_with_metadata():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["--param5", "ok"]
    )
    assert dotlist == ["--param5=ok"]


def test_parse_argv_from_structure_with_optional():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param8=1"]
    )
    assert dotlist == ["param8=1"]


def test_parse_argv_from_structure_with_optional_null():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param8=null"]
    )
    assert dotlist == ["param8=null"]


def test_parse_argv_from_structure_with_optional_null_fail():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param8=nok"]
        )


def test_parse_argv_from_structure_with_union_type_1():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param9=0.3"]
    )
    assert dotlist == ["param9=0.3"]


def test_parse_argv_from_structure_with_union_type_2():
    dotlist = parse_argv_from_dataclass(
        StructuredConfig,
        ["param9=3"]
    )
    assert dotlist == ["param9=3"]


def test_parse_argv_from_structure_with_union_fail():
    with pytest.raises(ValueError):
        parse_argv_from_dataclass(
            StructuredConfig,
            ["param9=nok"]
        )
