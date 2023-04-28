import codecs
import re
from pathlib import Path

from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements


def get_version():
    """
    Inspired from https://github.com/omry/omegaconf/blob
    /63c36b507f216f48e23ddb3c5251698cc2f51358/build_helpers/build_helpers.py
    #L164
    """
    root = Path(__file__).parent.absolute()
    with codecs.open(root / "bim_gw/version.py", "r") as fp:
        version_file = fp.read()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError(
        "Unable to find version string."
    )


extras_require = {
    "wandb": ["wandb>=0.13.9"],
    "dev": ["flake8"],
}
extras_require["all"] = list(set(sum(extras_require.values(), [])))

if __name__ == '__main__':
    setup(
        name='bim_gw',
        version=get_version(),
        install_requires=get_requirements(),
        packages=find_packages(),
        extras_require=extras_require
    )
