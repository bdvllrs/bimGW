import logging


class Register:
    __instance = None
    _register = {}

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls, *args, **kwargs)
        return cls.__instance

    def add(self, key, domain):
        if key not in self._register:
            self._register[key] = domain
            return
        logging.warning(f"{key} already exists in register. Ignoring.")

    def get(self, key):
        return self._register[key]


class DomainRegister(Register):
    pass


class DatasetRegister(Register):
    pass
