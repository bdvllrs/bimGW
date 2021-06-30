import sys

import torchvision
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


def get_args(debug=False):
    print("Cli args")
    print(sys.argv)

    # Configurations
    config_path = PROJECT_DIR / "config"
    main_args = OmegaConf.load(str((config_path / "main.yaml").resolve()))
    if debug and (config_path / "debug.yaml").exists():
        debug_args = OmegaConf.load(str((config_path / "debug.yaml").resolve()))
    else:
        debug_args = {}
    cli_args = OmegaConf.from_cli()
    if (config_path / "local.yaml").exists():
        local_args = OmegaConf.load(str((config_path / "local.yaml").resolve()))
    else:
        local_args = {}

    args = OmegaConf.merge(main_args, local_args, debug_args, cli_args)

    print(OmegaConf.to_yaml(cli_args))
    print("Complete args")
    print(OmegaConf.to_yaml(args))
    return args


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    if logger is not None:
        # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
        sample_imgs = sample_imgs - sample_imgs.min()
        sample_imgs = sample_imgs / sample_imgs.max()
        img_grid = torchvision.utils.make_grid(sample_imgs, **kwargs)
        img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
        logger.experiment[name].log(File.as_image(img_grid), step=step)