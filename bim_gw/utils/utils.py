import logging
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from bim_gw.utils.types import LoadFromData


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
    sample_imgs = sample_imgs - sample_imgs.min()
    sample_imgs = sample_imgs / sample_imgs.max()
    img_grid = torchvision.utils.make_grid(sample_imgs, pad_value=1, **kwargs)
    if logger is not None and hasattr(logger, "log_image"):
        logger.log_image(name, img_grid, step=step)
    else:
        img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
        plt.imshow(img_grid)
        plt.title(name)
        plt.tight_layout(pad=0)
        plt.show()


@contextmanager
def log_if_save_last_images(logger):
    """
    Creates a context manager that saves images if the logger has the attribute do_save_last_images set to True.
    """
    if logger is not None and getattr(logger, "do_save_last_images", False):
        save_images = getattr(logger, "do_save_images", False)
        logger.save_images(True)
        yield
        logger.save_images(save_images)
        return
    yield


def logger_save_images(logger, mode=True):
    if logger is not None and hasattr(logger, "save_images"):
        logger.save_images(mode)


def loggers_save_images(loggers, mode=True):
    for logger in loggers:
        if getattr(logger, "do_save_last_images", False):
            logger_save_images(logger, mode)


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


def replace_legacy_supervision_to_translation(x):
    if x['parameters/losses/coefs/supervision'] is not None:
        return x['parameters/losses/coefs/supervision']
    return x['parameters/losses/coefs/translation']


def replace_legacy_selected_domains(x):
    selected_domains = []
    for column in x.index:
        if column == "parameters/global_workspace/selected_domains":
            selected_domains = x[column]
            break
        if "parameters/global_workspace/selected_domains" in column:
            selected_domains.append(x[column])
    return selected_domains


def update_df_for_legacy_code(df):
    if 'parameters/losses/coefs/translation' not in df.columns:
        df['parameters/losses/coefs/translation'] = df.apply(replace_legacy_supervision_to_translation, axis=1)

    for column in df.columns:
        if "parameters/global_workspace/selected_domains" in column:
            df['parameters/global_workspace/selected_domains'] = df.apply(replace_legacy_selected_domains, axis=1)
            break
    return df


def get_runs_dataframe(args):
    if args.load_from == LoadFromData.csv:
        return update_df_for_legacy_code(pd.read_csv(Path(args.csv_path)))
    elif args.load_from == LoadFromData.wandb:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_entity_project, OmegaConf.to_object(args.wandb_filter))
        columns = {}
        for run in runs:
            vals = run.summary._json_dict
            vals.update(
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')}
            )
            vals["Name"] = run.name

            for k, v in vals.items():
                if isinstance(v, dict) and "min" in v:
                    k += "." + "min"
                    v = v['min']
                if k not in columns:
                    columns[k] = []
                columns[k].append(v)

        return update_df_for_legacy_code(pd.DataFrame(columns))
    raise ValueError(f"Unknown load_from: {args.load_from}")


def get_job_slug_from_coefficients(x):
    name = ""
    if x['parameters/losses/coefs/translation'] > 0:
        name += "+tr"
    if x['parameters/losses/coefs/contrastive'] > 0:
        name += "+cont"
    if x['parameters/losses/coefs/demi_cycles'] > 0:
        name += "+dcy"
    if x['parameters/losses/coefs/cycles'] > 0:
        name += "+cy"
    return name[1:]


def find_best_epoch(ckpt_folder):
    ckpt_folder = Path(ckpt_folder)
    if not ckpt_folder.exists():
        return None
    files = [(str(p), int(str(p).split('/')[-1].split('-')[0][6:])) for p in ckpt_folder.iterdir()]
    if not len(files):
        return None

    last = sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    loaded_path = torch.load(last, map_location=torch.device('cpu'))
    for callback_name, callback in loaded_path['callbacks'].items():
        if 'ModelCheckpoint' in callback_name and 'best_model_path' in callback and os.path.isfile(
                callback['best_model_path']
        ):
            return callback['best_model_path']
    return last


def find_remote_last_epoch(remote_path, ssh):
    stdin, stdout, stderr = ssh.exec_command(f'ls -al {remote_path}')
    files = list(map(lambda x: x.split(" ")[-1], stdout.read().decode("utf-8").split("\n")[3:-1]))
    files = list(map(lambda x: (x, x.split("-")[0][6:]), files))
    if len(files):
        return remote_path / sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    return remote_path


def get_checkpoint_path(path):
    if path is not None and type(path) is str and os.path.isdir(path):
        return find_best_epoch(path)

    elif path is not None and type(path) is str:
        return path

    elif path is not None and 'load_from' in path and path.load_from == "local":
        assert 'local_path' in path, "Missing local_path in value when using load_from='local'"
        return get_checkpoint_path(path.local_path)

    elif path is not None and 'load_from' in path and path.load_from == "remote":
        assert 'local_path' in path, "Missing local_path in value when using load_from='remote'"
        assert 'remote_server' in path, "Missing remote_server in value when using load_from='remote'"
        assert 'remote_checkpoint_path' in path, "Missing remote_checkpoint_path in value when using load_from='remote'"

        logging.info(
            f"Downloading checkpoint from {path.remote_server} in {path.remote_checkpoint_path} "
            f"to local {path.local_path}.")

        local_path = Path(path.local_path)
        best_epoch = find_best_epoch(path.local_path)
        if best_epoch is not None:
            return best_epoch

        local_path.mkdir(parents=True, exist_ok=True)

        from paramiko import SSHClient
        from scp import SCPClient

        remote_user = path.remote_user if 'remote_user' in path else None
        remote_password = path.remote_password if 'remote_password' in path else None
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(path.remote_server, username=remote_user, password=remote_password)
            with SCPClient(ssh.get_transport()) as scp:
                last_epoch = find_remote_last_epoch(Path(path.remote_checkpoint_path), ssh)
                scp.get(str(last_epoch), str(local_path))
        return find_best_epoch(path.local_path)

    return path
