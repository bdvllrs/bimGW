import sys

import torchvision
from matplotlib import pyplot as plt
from neptune.new.types import File
from omegaconf import OmegaConf

from bim_gw.utils import PROJECT_DIR


def has_internet_connection(host='https://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False


def get_args(debug=False, additional_config_files=None, cli=True):
    print("Cli args")
    print(sys.argv)

    # Configurations
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

    args = OmegaConf.merge(main_args, local_args, debug_args, cli_args)

    if additional_config_files is not None:
        for file in additional_config_files:
            if file.exists():
                args = OmegaConf.merge(args, OmegaConf.load(str(file.resolve())))

    args = OmegaConf.merge(args, cli_args)

    print(OmegaConf.to_yaml(cli_args))
    print("Complete args")
    print(OmegaConf.to_yaml(args))

    args.debug = args.debug or debug
    return args


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
    sample_imgs = sample_imgs - sample_imgs.min()
    sample_imgs = sample_imgs / sample_imgs.max()
    img_grid = torchvision.utils.make_grid(sample_imgs, pad_value=1, **kwargs)
    logger.log_image(name, img_grid, step=step)
    #     plt.imshow(img_grid)
    #     plt.title(name)
    #     plt.tight_layout(pad=0)
    #     plt.show()


def val_or_default(d, key, default=None):
    """
    Returns the value of a dict, or default value if key is not in the dict.
    Args:
        d: dict
        key:
        default:

    Returns: d[key] if key in d else default
    """
    if key in d:
        return d[key]
    return default


