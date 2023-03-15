import logging
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from bim_gw.utils.constants import PROJECT_DIR
from bim_gw.utils.types import BIMConfig


def has_internet_connection(host='https://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except Exception:
        return False


def split_resolver(key, value, item=None):
    if hasattr(value, "split"):
        return value.split(key)[item]
    return value


def loss_coef_slug_resolver(*, _root_):
    name = ""
    if type(_root_.losses.coefs.translation) in [int,
                                                 float] and \
            _root_.losses.coefs.translation > 0:
        name += "+tr"
    if type(_root_.losses.coefs.contrastive) in [int,
                                                 float] and \
            _root_.losses.coefs.contrastive > 0:
        name += "+cont"
    if type(_root_.losses.coefs.demi_cycles) in [int,
                                                 float] and \
            _root_.losses.coefs.demi_cycles > 0:
        name += "+dcy"
    if type(_root_.losses.coefs.cycles) in [int,
                                            float] and \
            _root_.losses.coefs.cycles > 0:
        name += "+cy"
    return name[1:]


def load_extra_conf_resolver(path):
    return OmegaConf.load(str(PROJECT_DIR / "config" / path))


def load_resolvers_if_needed():
    if not OmegaConf.has_resolver("split"):
        OmegaConf.register_new_resolver("split", split_resolver)
    if not OmegaConf.has_resolver("path"):
        OmegaConf.register_new_resolver("path", load_extra_conf_resolver)
    if not OmegaConf.has_resolver("coef_slug"):
        OmegaConf.register_new_resolver("coef_slug", loss_coef_slug_resolver)


def get_args(
    debug=False, additional_config_files=None, cli=True, verbose=True,
    use_schema=True
):
    load_resolvers_if_needed()

    # Configurations
    default_args = OmegaConf.create(
        {
            "debug": False,
        }
    )

    config_path_env = os.getenv("BIMGW_CONFIG_PATH", None)
    config_path = PROJECT_DIR / "config" if config_path_env is None else Path(
        config_path_env
    )
    main_args = OmegaConf.load(str((config_path / "main.yaml").resolve()))
    if debug and (config_path / "debug.yaml").exists():
        debug_args = OmegaConf.load(
            str((config_path / "debug.yaml").resolve())
        )
    else:
        debug_args = {}

    if cli:
        cli_args = OmegaConf.from_cli()
    else:
        cli_args = OmegaConf.create()
    if (config_path / "local.yaml").exists():
        local_args = OmegaConf.load(
            str((config_path / "local.yaml").resolve())
        )
    else:
        local_args = {}

    args = OmegaConf.merge(default_args, main_args, local_args, debug_args)

    if additional_config_files is not None:
        for file in additional_config_files:
            if file.exists():
                args = OmegaConf.merge(
                    args, OmegaConf.load(str(file.resolve()))
                )

    args = OmegaConf.merge(args, cli_args)

    if verbose:
        print(OmegaConf.to_yaml(cli_args))
        print("Complete args")
        print(OmegaConf.to_yaml(args))

    args.debug = args.debug or debug

    if args.seed == "random":
        args.seed = np.random.randint(0, 100_000)

    # Backward compatibility for old config files
    if OmegaConf.is_dict(args.global_workspace.selected_domains):
        logging.warning(
            "Selected domains is a dict, converting to list. Use a list in "
            "the config file in future runs."
        )
        args.global_workspace.selected_domains = OmegaConf.create(
            [values for values in
             args.global_workspace.selected_domains.values()]
        )
    if args.losses.coefs.supervision is not None:
        logging.warning(
            "Using deprecated value `losses.coefs.supervision`. In the "
            "future, use `losses.coefs.translation` instead."
        )
        args.losses.coefs.translation = args.losses.coefs.supervision
    if "attr" not in args.fetchers or "use_unpaired" not in args.fetchers.attr:
        logging.warning(
            "Missing mandatory value `fetchers.attr.use_unpaired`. "
            "Automatically set to false."
        )
        args.fetchers.attr.use_unpaired = False

    if use_schema:
        schema = OmegaConf.structured(BIMConfig)
        OmegaConf.set_struct(schema, False)
        args = OmegaConf.merge(schema, args)

    return args
