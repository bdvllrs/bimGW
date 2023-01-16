from pathlib import Path

import pandas as pd
import torchvision
from matplotlib import pyplot as plt
from omegaconf import OmegaConf


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
    sample_imgs = sample_imgs - sample_imgs.min()
    sample_imgs = sample_imgs / sample_imgs.max()
    img_grid = torchvision.utils.make_grid(sample_imgs, pad_value=1, **kwargs)
    if logger is not None:
        logger.log_image(name, img_grid, step=step)
    else:
        img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
        plt.imshow(img_grid)
        plt.title(name)
        plt.tight_layout(pad=0)
        plt.show()


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
    if args.load_from == "csv":
        return update_df_for_legacy_code(pd.read_csv(Path(args.csv_path)))
    elif args.load_from == "wandb":
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_entity_project, OmegaConf.to_object(args.wandb_filter))
        columns = {}
        for run in runs:
            vals = run.summary._json_dict
            vals.update({k: v for k, v in run.config.items()
                         if not k.startswith('_')})
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
    files = [(str(p), int(str(p).split('/')[-1].split('-')[0][6:])) for p in ckpt_folder.iterdir()]
    return sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    # epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    # return max(epochs)
