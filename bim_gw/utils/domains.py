import logging


class DomainRegistry:
    __instance = None
    _domain_callbacks = {}

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def add(self, key, domain):
        if key not in self._domain_callbacks:
            self._domain_callbacks[key] = domain
            return
        logging.warning(f"Domain {key} already exists in registry. Ignoring.")

    def get(self, key):
        return self._domain_callbacks[key]


