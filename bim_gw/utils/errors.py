class ConfigError(Exception):
    """Exception raised when an error in the configuration file is detected.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, key, message=None):
        self.message = f"Error in config key {key}."
        if message is not None:
            self.message += f" {message}"
        super().__init__(self.message)
