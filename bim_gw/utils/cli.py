import dataclasses
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def _get_real_field(fields, name):
    for field in fields.values():
        if name in field.metadata.get("cli_names", [field.name]):
            return field
    return None


def _isinstance(obj, type_):
    if type_ is Any:
        return True
    try:
        subtypes = type_.__args__
        for subtype in subtypes:
            if subtype is float:
                subtype = (float, int)
            if isinstance(obj, subtype):
                return True
        return False
    except AttributeError:
        if type_ is float:
            type_ = (float, int)
        return isinstance(obj, type_)


def _is_valid_dataclass_type(dataclass_type, loaded_value):
    sub_fields = _get_fields(dataclass_type)
    for val_name, sub_val in loaded_value.items():
        real_field = _get_real_field(sub_fields, val_name)
        if real_field is None:
            return False
        if not _is_valid_field_from_type(real_field.type, sub_val):
            return False
    return True


def _is_valid_list(field_type, loaded_value):
    if not isinstance(loaded_value, (list, tuple)):
        return False
    for val in loaded_value:
        field_subtype = field_type.__args__[0]
        if not _is_valid_field_from_type(field_subtype, val):
            return False
    return True


def _is_valid_enum(field_type, loaded_value):
    if not issubclass(field_type, Enum):
        return False
    return loaded_value in field_type.__members__.keys()


def _is_valid_field_from_type(field_type, loaded_value):
    if isinstance(loaded_value, (list, tuple)):
        return _is_valid_list(field_type, loaded_value)
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return _is_valid_enum(field_type, loaded_value)
    if _is_dataclass(field_type):
        return _is_valid_dataclass_type(field_type, loaded_value)
    return _isinstance(loaded_value, field_type)


def _is_valid_field(field, value):
    loaded_value = yaml.load(value)
    return _is_valid_field_from_type(field.type, loaded_value)


def _is_union_type(type_):
    return hasattr(type_, "__origin__") and type_.__origin__ is Union


# from https://github.com/omry/omegaconf/blob
# /deeee0475759c385583b6876d963dd9a8f47a108/omegaconf/_utils.py#LL630-L641C44
def _is_dict(type_: Any) -> bool:
    if type_ in (dict, Dict):
        return True
    origin = getattr(type_, "__origin__", None)
    # type_dict is a bit hard to detect.
    # this support is tentative, if it eventually causes issues in other
    # areas it may be dropped.
    if sys.version_info < (3, 7, 0):  # pragma: no cover
        typed_dict = hasattr(type_, "__base__") and type_.__base__ == Dict
        return origin is Dict or type_ is Dict or typed_dict
    else:  # pragma: no cover
        typed_dict = hasattr(type_, "__base__") and type_.__base__ == dict
        return origin is dict or typed_dict


def _is_type(type_, type_check):
    if type_check is Any or type_ is Any:
        return True
    if hasattr(type_, "__args__") and _is_union_type(type_):
        for t in type_.__args__:
            if _is_type(t, type_check):
                return True
        return False
    return type_ is type_check


def _is_field_flag(field):
    return _is_type(field.type, bool)


def _is_dataclass(structure):
    return dataclasses.is_dataclass(structure) and isinstance(structure, type)


def _get_fields(structure):
    fields = getattr(structure, "__dataclass_fields__")
    return fields


def _get_dict_type(type_):
    if hasattr(type_, "__args__"):
        return type_.__args__[1]
    return Any


def _split_argv(argv: List[str]) -> List[str]:
    new_argv = []
    for arg in argv:
        if "=" in arg:
            new_argv.extend(arg.split("="))
        else:
            new_argv.append(arg)
    return new_argv


def _get_field(structure, key: str):
    key_parts = key.split(".")
    renamed_key = []
    fields = _get_fields(structure)
    is_dict = False
    dict_type = None
    field = None
    for k, key in enumerate(key_parts):
        if is_dict:
            field = dataclasses.field()
            field.name = key
            field.type = dict_type
            is_dict = False
            dict_type = None
        else:
            field = _get_real_field(fields, key)

        if field is None:
            return None, None
        renamed_key.append(field.name)

        if _is_dataclass(field.type):
            structure = field.type
            fields = _get_fields(structure)
            continue

        if _is_dict(field.type) or field.type == Any:
            is_dict = True
            dict_type = _get_dict_type(field.type)
            continue

    return field, ".".join(renamed_key)


def parse_argv_from_dataclass(
    structure,
    argv: Optional[List[str]] = None,
):
    if not _is_dataclass(structure):
        raise TypeError("Structure must be a dataclass")

    dotlist = []
    # remove all = and split keys and values
    argv = _split_argv(argv or sys.argv[1:])

    last_key = None
    last_field = None
    for arg in argv:
        arg_field, renamed_key = _get_field(structure, arg)
        is_arg_valid_key = arg_field is not None

        # Last key was a flag
        if (
            is_arg_valid_key
            and last_field is not None
            and _is_field_flag(last_field)
        ):
            dotlist.append(f"{last_key}=True")
            last_key, last_field = None, None

        # This arg is a value
        if last_key is not None and _is_valid_field(last_field, arg):
            dotlist.append(f"{last_key}={arg}")
            last_key, last_field = None, None
            continue

        # This arg is a valid key
        if last_key is None and is_arg_valid_key:
            last_key, last_field = renamed_key, arg_field
            continue

        raise ValueError("Invalid key: " + arg)

    # Last key was a flag
    if last_field is not None and _is_field_flag(last_field):
        dotlist.append(f"{last_key}=True")
    elif last_key is not None:
        raise ValueError("Invalid key: " + last_key)

    return dotlist


def parse_args(cls, argv=None):
    cfg = OmegaConf.from_dotlist(parse_argv_from_dataclass(cls, argv))
    schema = OmegaConf.structured(cls)
    return OmegaConf.merge(schema, cfg)
