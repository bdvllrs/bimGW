import logging
import sys

import numpy as np
from omegaconf import OmegaConf

from bim_gw.utils import PROJECT_DIR


def has_internet_connection(host='https://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False

def split_resolver(key, value, item=None):
    if hasattr(value, "split"):
        return value.split(key)[item]
    return value

def load_extra_conf_resolver(path):
    return OmegaConf.load(str(PROJECT_DIR / "config" / path))


def get_args(debug=False, additional_config_files=None, cli=True):
    OmegaConf.register_new_resolver("split", split_resolver)
    OmegaConf.register_new_resolver("path", load_extra_conf_resolver)

    print("Cli args")
    print(sys.argv)

    # Configurations
    default_args = OmegaConf.create({
        "debug": False
    })

    config_path = PROJECT_DIR / "config"
    main_args = OmegaConf.load(str((config_path / "main.yaml").resolve()))
    if debug and (config_path / "debug.yaml").exists():
        debug_args = OmegaConf.load(str((config_path / "debug.yaml").resolve()))
    else:
        debug_args = {}

    if cli:
        cli_args = OmegaConf.from_dotlist(list(map(lambda x: x.replace("--", ""), sys.argv[1:])))
    else:
        cli_args = OmegaConf.create()
    if (config_path / "local.yaml").exists():
        local_args = OmegaConf.load(str((config_path / "local.yaml").resolve()))
    else:
        local_args = {}

    args = OmegaConf.merge(default_args, main_args, local_args, debug_args)

    if additional_config_files is not None:
        for file in additional_config_files:
            if file.exists():
                args = OmegaConf.merge(args, OmegaConf.load(str(file.resolve())))

    args = OmegaConf.merge(args, cli_args)

    print(OmegaConf.to_yaml(cli_args))
    print("Complete args")
    print(OmegaConf.to_yaml(args))

    args.debug = args.debug or debug

    if args.seed == "random":
        args.seed = np.random.randint(0, 100_000)

    # Backward compatibility for old config files
    if OmegaConf.is_dict(args.global_workspace.selected_domains):
        logging.warning("Selected domains is a dict, converting to list. Use a list in the config file in future runs.")
        args.global_workspace.selected_domains = OmegaConf.create(
            [values for values in args.global_workspace.selected_domains.values()])
    if args.losses.coefs.supervision is not None:
        logging.warning("Using deprecated value `losses.coefs.supervision`. In the future, use `losses.coefs.translation` instead.")
        args.losses.coefs.translation = args.losses.coefs.supervision

    return args
