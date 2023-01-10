from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements


def get_version():
    with open("version.txt", "r") as f:
        version = f.read().lstrip("\n")
    return version


extras_require = {
    "wandb": ["wandb>=0.13.4"],
    "neptune": ["neptune-client>=0.16"],
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
