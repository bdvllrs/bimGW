import dataclasses
import sys
from typing import List, Optional

import yaml


def _get_real_field(fields, name):
    if name in fields.keys():
        return fields[name]
    for field in fields.values():
        if field.metadata.get("cli", field.name) == name:
            return field
    return None


def _is_valid_dataclass_type(dataclass_type, loaded_value):
    sub_fields = {
        field.name: field
        for field in dataclasses.fields(dataclass_type)
    }
    for val_name, sub_val in loaded_value.items():
        real_field = _get_real_field(sub_fields, val_name)
        if real_field is None:
            return False
        if not _is_valid_field(real_field.type, sub_val):
            return False
    return True


def _is_valid_list(field_type, loaded_value):
    if not isinstance(loaded_value, (list, tuple)):
        return False
    for val in loaded_value:
        field_type = field_type.__args__[0]
        if not _is_valid_field(field_type, val):
            return False
    return True


def _is_valid_field(field_type, loaded_value):
    if isinstance(loaded_value, (list, tuple)):
        return _is_valid_list(field_type, loaded_value)
    if is_dataclass(field_type):
        return _is_valid_dataclass_type(field_type, loaded_value)
    return isinstance(loaded_value, field_type)


def is_valid_field(field, value):
    loaded_value = yaml.load(value, Loader=yaml.SafeLoader)
    return _is_valid_field(field.type, loaded_value)


def is_field_flag(field):
    return field.type is bool


def is_dataclass(structure):
    return (
            dataclasses.is_dataclass(structure)
            and isinstance(structure, type)
    )


def split_argv(argv):
    new_argv = []
    for arg in argv:
        if "=" in arg:
            new_argv.extend(arg.split("="))
        else:
            new_argv.append(arg)
    return new_argv


def get_field(structure, key: str):
    key_parts = key.split(".")
    fields = {field.name: field for field in dataclasses.fields(structure)}
    field = None
    for key in key_parts:
        field = _get_real_field(fields, key)
        if field is None:
            return None

        if is_dataclass(field.type):
            structure = field.type
            fields = {field.name: field
                      for field in dataclasses.fields(structure)}
    return field


def parse_argv_from_structure(
    structure,
    argv: Optional[List[str]] = None,
):
    if not is_dataclass(structure):
        raise TypeError("Structure must be a dataclass")

    if argv is None:
        argv = sys.argv[1:]
    dotlist = []
    # remove all = and split keys and values
    argv = split_argv(argv)

    last_key = None
    last_field = None
    for arg in argv:
        arg_field = get_field(structure, arg)
        is_arg_valid_key = arg_field is not None

        # Last key was a flag
        if (
                is_arg_valid_key
                and last_field is not None
                and is_field_flag(structure, last_field)
        ):
            dotlist.append(f"{last_key}=True")
            last_key, last_field = None, None

        # This arg is a value
        if last_key is not None and is_valid_field(last_field, arg):
            dotlist.append(f"{last_key}={arg}")
            last_key, last_field = None, None
            continue

        # This arg is a valid key
        if is_arg_valid_key:
            last_key, last_field = arg, arg_field
            continue

        raise ValueError("Invalid key: " + arg)

    # Last key was a flag
    if (last_field is not None
            and is_field_flag(last_field)):
        dotlist.append(f"{last_key}=True")

    return dotlist
