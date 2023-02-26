import logging

_domain_registry = {}
_dataset_registry = {}


def register_domain(key, domain):
    if key not in _domain_registry:
        _domain_registry[key] = domain
        return
    logging.warning(f"{key} already exists in registry. Ignoring.")


def get_domain(key):
    return _domain_registry[key]


def register_dataset(key, dataset):
    if key not in _dataset_registry:
        _dataset_registry[key] = dataset
        return
    logging.warning(f"{key} already exists in registry. Ignoring.")


def get_dataset(key):
    return _dataset_registry[key]
