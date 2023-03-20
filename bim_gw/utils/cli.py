import dataclasses
import sys
from enum import Enum
from typing import List, Optional

import yaml
from omegaconf import OmegaConf


def _get_real_field(fields, name):
    for field in fields.values():
        if name in field.metadata.get("cli_names", [field.name]):
            return field
    return None


def _isinstance(obj, type_):
    try:
        subtypes = type_.__args__
        for subtype in subtypes:
            if isinstance(obj, subtype):
                return True
        return False
    except AttributeError:
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
        field_type = field_type.__args__[0]
        if not _is_valid_field_from_type(field_type, val):
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
    loaded_value = yaml.load(value, Loader=yaml.SafeLoader)
    return _is_valid_field_from_type(field.type, loaded_value)


def _is_field_flag(field):
    if hasattr(field.type, "__args__"):
        return bool in field.type.__args__
    return field.type is bool


def _is_dataclass(structure):
    return (
            dataclasses.is_dataclass(structure)
            and isinstance(structure, type)
    )


def _get_fields(structure):
    fields = getattr(structure, "__dataclass_fields__")
    return fields


def _split_argv(argv):
    new_argv = []
    for arg in argv:
        if "=" in arg:
            new_argv.extend(arg.split("="))
        else:
            new_argv.append(arg)
    return new_argv


def _get_field(structure, key: str):
    key_parts = key.split(".")
    fields = _get_fields(structure)
    field = None
    for key in key_parts:
        field = _get_real_field(fields, key)
        if field is None:
            return None

        if _is_dataclass(field.type):
            structure = field.type
            fields = _get_fields(structure)
    return field


def parse_argv_from_dataclass(
    structure,
    argv: Optional[List[str]] = None,
):
    if not _is_dataclass(structure):
        raise TypeError("Structure must be a dataclass")

    if argv is None:
        argv = sys.argv[1:]
    dotlist = []
    # remove all = and split keys and values
    argv = _split_argv(argv)

    last_key = None
    last_field = None
    for arg in argv:
        arg_field = _get_field(structure, arg)
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
            last_key, last_field = arg, arg_field
            continue

        raise ValueError("Invalid key: " + arg)

    # Last key was a flag
    if (last_field is not None
            and _is_field_flag(last_field)):
        dotlist.append(f"{last_key}=True")
    elif last_key is not None:
        raise ValueError("Invalid key: " + last_key)

    return dotlist


def parse_args(cls, argv=None):
    cfg = OmegaConf.from_dotlist(parse_argv_from_dataclass(cls, argv))
    schema = OmegaConf.structured(cls)
    return OmegaConf.merge(schema, cfg)
