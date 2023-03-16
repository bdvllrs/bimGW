import sys
from typing import List, Optional

from omegaconf import (
    BooleanNode, Container, ListConfig,
    OmegaConf, ValidationError, ValueNode
)

_cached_nodes = {}


def select_node(
    structure: Container,
    key: str
):
    if key in _cached_nodes.keys():
        return _cached_nodes[key]

    _node = None

    def gather(_cfg: Container) -> None:
        nonlocal _node
        if isinstance(_cfg, ListConfig):
            itr = range(len(_cfg))
        else:
            itr = _cfg

        for k in itr:
            _cached_nodes[_cfg._get_full_key(k)] = _cfg._get_node(k)

            if OmegaConf.is_config(_cfg._get_node(k)):
                gather(_cfg[k])
            elif _cfg._get_full_key(k) == key:
                _node = _cfg._get_node(k)

    gather(structure)
    return _node


def validate_node(node, value):
    if isinstance(node, ValueNode):
        return node.validate_and_convert(value)

    return OmegaConf.merge(node, OmegaConf.create(value))


def parse_argv_from_structure(
    structure: Container,
    argv: Optional[List[str]] = None
):
    if argv is None:
        argv = sys.argv[1:]
    dotlist = []
    last_key = None
    last_flag = False
    for k in range(len(argv) + 1):
        key, val = None, None
        if k < len(argv):
            arg = argv[k].strip(" ")
            if "=" in arg:
                key, val = arg.split("=")
            elif last_key is not None:
                val = arg
            else:
                key = arg
        elif last_key is None:
            break

        if (last_flag
                and last_key is not None
                and key is None and val is not None):
            node = select_node(structure, last_key)
            try:
                validate_node(node, val)
            except ValidationError:
                raise ValueError(f"Invalid value for {last_key}: {val}")
            dotlist.append(f"{last_key}={val}")
            last_flag = False
            last_key = None
            continue
        elif last_flag and last_key is not None and val is None:
            dotlist.append(f"{last_key}=True")
            last_flag = False
            last_key = None
            continue

        if key is None and last_key is not None:
            key = last_key

        try:
            OmegaConf.select(
                structure, key,
                throw_on_missing=False,
                throw_on_resolution_failure=True
            )
        except Exception:
            raise ValueError(f"Invalid argument: {key}")

        node = select_node(structure, key)

        if val is not None:
            try:
                validate_node(node, val)
            except ValidationError:
                raise ValueError(f"Invalid value for {key}: {val}")
            dotlist.append(f"{key}={val}")
            last_key = None
            continue

        if isinstance(node, BooleanNode) and val is None:
            last_flag = True
            last_key = key
            continue

        last_key = key

    return dotlist
