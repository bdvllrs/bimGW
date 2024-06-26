import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Union,
    cast,
)

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from bim_gw.utils.errors import ConfigError
from bim_gw.utils.types import (
    DataSelectorAxesConfig,
    LoadFromData,
    WandbFilterT,
)


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    sample_imgs = sample_imgs - sample_imgs.min()
    sample_imgs = sample_imgs / sample_imgs.max()
    img_grid = torchvision.utils.make_grid(sample_imgs, pad_value=1, **kwargs)
    if logger is not None and hasattr(logger, "log_image"):
        logger.log_image(name, img_grid, step=step)
    else:
        img_grid = torchvision.transforms.ToPILImage(mode="RGB")(
            img_grid.cpu()
        )
        plt.imshow(img_grid)
        plt.title(name)
        plt.tight_layout(pad=0)
        plt.show()


@contextmanager
def log_if_save_last_images(logger):
    """
    Creates a context manager that saves images if the logger has the
    attribute do_save_last_images set to True.
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


@contextmanager
def log_if_save_last_tables(logger):
    """
    Creates a context manager that saves tables if the logger has the
    attribute do_save_last_tables set to True.
    """
    if logger is not None and getattr(logger, "do_save_last_tables", False):
        save_tables = getattr(logger, "do_save_tables", False)
        logger.save_tables(True)
        yield
        logger.save_tables(save_tables)
        return
    yield


def logger_save_tables(logger, mode=True):
    if logger is not None and hasattr(logger, "save_tables"):
        logger.save_tables(mode)


def loggers_save_tables(loggers, mode=True):
    for logger in loggers:
        if getattr(logger, "do_save_last_images", False):
            logger_save_tables(logger, mode)


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
    if x["parameters/losses/coefs/supervision"] is not None:
        return x["parameters/losses/coefs/supervision"]
    return x["parameters/losses/coefs/translation"]


def replace_legacy_selected_domains(x):
    selected_domains = []
    for column in x.index:
        if column == "parameters/global_workspace/selected_domains":
            selected_domains = x[column]
            break
        if "parameters/global_workspace/selected_domains" in column:
            selected_domains.append(x[column])
    return selected_domains


def update_df_for_legacy_code(df: pd.DataFrame) -> pd.DataFrame:
    if "parameters/losses/coefs/translation" not in df.columns:
        df["parameters/losses/coefs/translation"] = df.apply(
            replace_legacy_supervision_to_translation, axis=1
        )
    if "parameters/global_workspace/prop_available_images" not in df.columns:
        df["parameters/global_workspace/prop_available_images"] = 1.0

    for column in df.columns:
        if "parameters/global_workspace/selected_domains" in column:
            df["parameters/global_workspace/selected_domains"] = df.apply(
                replace_legacy_selected_domains, axis=1
            )
            break
    return df


def get_runs_dataframe(args: DataSelectorAxesConfig) -> pd.DataFrame:
    if args.load_from == LoadFromData.csv:
        if args.csv_path is None:
            raise ValueError("csv_path must be set when load_from is csv")
        return update_df_for_legacy_code(
            cast(pd.DataFrame, pd.read_csv(Path(args.csv_path)))
        )
    elif args.load_from == LoadFromData.wandb:
        import wandb

        assert args.wandb_entity_project is not None
        assert args.wandb_filter is not None

        api = wandb.Api()
        runs = api.runs(
            args.wandb_entity_project,
            cast(
                WandbFilterT,
                OmegaConf.to_container(args.wandb_filter, resolve=True),
            ),
        )
        columns: Dict[str, List[Any]] = defaultdict(list)
        tracked_columns: Set[str] = set()
        for n_run, run in enumerate(runs):
            vals = run.summary._json_dict  # noqa
            vals.update(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            vals["Name"] = run.name
            vals["ID"] = run.id

            seen_columns = set(tracked_columns)
            for k, v in vals.items():
                if isinstance(v, dict) and "min" in v:
                    k += "." + "min"
                    v = v["min"]
                if k not in tracked_columns:
                    tracked_columns.add(k)
                    # Previous runs didn't have this, so add a None entry
                    columns[k] = [None] * n_run

                seen_columns.discard(k)

                columns[k].append(v)

            for missing_k in seen_columns:
                columns[missing_k].append(None)
        for k in list(columns.keys()):
            if len(columns[k]) != len(columns["Name"]):
                assert False, "Should never happen"
                # logging.info(
                #     f"Deleted key '{k}' because it has a different "
                #     f"length {len(columns[k])} than the number of runs"
                #     f" {len(columns['Name'])}"
                # )
                # del columns[k]
        return update_df_for_legacy_code(pd.DataFrame(columns))
    raise ValueError(f"Unknown load_from: {args.load_from}")


def get_additional_slug(
    x: pd.DataFrame,
    additional_conds: Mapping[str, Callable[[pd.DataFrame], bool]],
) -> str:
    name = ""
    for val, cond in additional_conds.items():
        if cond(x):
            name += val
    return name


def get_job_slug_from_coefficients(
    x: pd.Series,
    additional_slug: Optional[Callable[[pd.Series], str]] = None,
) -> str:
    name = ""
    if x["parameters/losses/coefs/translation"] > 0:  # type: ignore
        name += "+tr"
    if x["parameters/losses/coefs/contrastive"] > 0:  # type: ignore
        name += "+cont"
    if x["parameters/losses/coefs/demi_cycles"] > 0:  # type: ignore
        name += "+dcy"
    if x["parameters/losses/coefs/cycles"] > 0:  # type: ignore
        name += "+cy"
    if additional_slug is not None:
        r = additional_slug(x)
        name += r
    return name[1:]


def get_job_detailed_slug_from_coefficients(x):
    name = ""
    if x["parameters/losses/coefs/translation"] > 0:
        name += f"+tr_{x['parameters/losses/coefs/translation']:.2f}"
    if x["parameters/losses/coefs/contrastive"] > 0:
        name += f"+cont_{x['parameters/losses/coefs/contrastive']:.2f}"
    if x["parameters/losses/coefs/demi_cycles"] > 0:
        name += f"+dcy_{x['parameters/losses/coefs/demi_cycles']:.2f}"
    if x["parameters/losses/coefs/cycles"] > 0:
        name += f"+cy_{x['parameters/losses/coefs/cycles']:.2f}"
    return name[1:]


def update_checkpoint_for_compat(ckpt_path: Union[str, Path]) -> Path:
    new_path = Path(ckpt_path)
    new_path = new_path.parent / (new_path.stem + "_new" + new_path.suffix)
    if not new_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        checkpoint["state_dict"] = {
            k.replace("domain_mods", "domains._domain_modules")
            .replace("v.encoder_head.0", "v.encoder_head.z_img")
            .replace("t.encoder_head.0", "t.encoder_head.z"): v
            for k, v in checkpoint["state_dict"].items()
        }
        torch.save(checkpoint, new_path)
    return new_path


def find_best_epoch(ckpt_folder):
    ckpt_folder = Path(ckpt_folder)
    if not ckpt_folder.exists():
        return None
    files = [
        (str(p), int(str(p).split("/")[-1].split("-")[0][6:]))
        for p in ckpt_folder.iterdir()
    ]
    if not len(files):
        return None

    last = sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    loaded_path = torch.load(last, map_location=torch.device("cpu"))
    for callback_name, callback in loaded_path["callbacks"].items():
        if (
            "ModelCheckpoint" in callback_name
            and "best_model_path" in callback
            and os.path.isfile(callback["best_model_path"])
        ):
            return callback["best_model_path"]
    return last


def find_remote_last_epoch(remote_path, ssh):
    stdin, stdout, stderr = ssh.exec_command(f"ls -al {remote_path}")
    files = list(
        map(
            lambda x: x.split(" ")[-1],
            stdout.read().decode("utf-8").split("\n")[3:-1],
        )
    )
    files = list(map(lambda x: (x, x.split("-")[0][6:]), files))
    if len(files):
        return (
            remote_path / sorted(files, key=lambda x: x[0], reverse=True)[0][0]
        )
    return remote_path


def get_checkpoint_path(path):
    if path is not None and type(path) is str and os.path.isdir(path):
        return find_best_epoch(path)

    elif path is not None and type(path) is str:
        return path

    elif (
        path is not None and "load_from" in path and path.load_from == "local"
    ):
        if "local_path" not in path:
            raise ConfigError(
                "local_path",
                "Missing local_path in value when using load_from='local'",
            )
        return get_checkpoint_path(path.local_path)

    elif (
        path is not None and "load_from" in path and path.load_from == "remote"
    ):
        if "local_path" not in path:
            raise ConfigError(
                "local_path",
                "Missing local_path in value when using load_from='remote'",
            )
        if "remote_server" not in path:
            raise ConfigError(
                "remote_server",
                "Missing remote_server in value when using "
                "load_from='remote'",
            )
        if "remote_checkpoint_path" not in path:
            raise ConfigError(
                "remote_checkpoint_path",
                "Missing remote_checkpoint_path in value when using "
                "load_from='remote'",
            )

        logging.info(
            f"Downloading checkpoint from {path.remote_server} in "
            f"{path.remote_checkpoint_path} "
            f"to local {path.local_path}."
        )

        local_path = Path(path.local_path)
        best_epoch = find_best_epoch(path.local_path)
        if best_epoch is not None:
            return best_epoch

        local_path.mkdir(parents=True, exist_ok=True)

        from paramiko import SSHClient
        from scp import SCPClient

        remote_user = path.remote_user if "remote_user" in path else None
        remote_password = None
        if "remote_password" in path:
            remote_password = path.remote_password
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect(
                path.remote_server,
                username=remote_user,
                password=remote_password,
            )
            with SCPClient(ssh.get_transport()) as scp:
                last_epoch = find_remote_last_epoch(
                    Path(path.remote_checkpoint_path), ssh
                )
                scp.get(str(last_epoch), str(local_path))
        return find_best_epoch(path.local_path)

    return path


def has_internet_connection(host="https://google.com"):
    import urllib.request

    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except Exception:
        return False
