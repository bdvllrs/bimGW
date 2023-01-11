from pathlib import Path

import numpy as np
import pandas as pd


def get_name(x):
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


def get_fmt(name):
    if name == "tr":
        return {"linestyle": "-", "color": "#17b3f2"}
    if name == "tr+cy":
        return {"linestyle": "-", "marker": "d", "color": "#62C370"}
    if name == "tr+cont":
        return {"linestyle": "--", "color": "#CC3363"}
    if name == "tr+cont+dcy+cy":
        return {"linestyle": "--", "marker": "d", "color": "#FFAA5A"}


def get_fmt_all(name):
    if name == "tr":
        return {"linestyle": "-", "color": "#17b3f2"}
    if name == "tr+dcy":
        return {"linestyle": "-", "color": "#CC3363"}
    if name == "tr+cy":
        return {"linestyle": "-", "color": "#62C370"}
    if name == "tr+dcy+cy":
        return {"linestyle": "-", "color": "#FFAA5A"}
    if name == "dcy":
        return {"linestyle": "-.", "color": "#CC3363"}
    if name == "cy":
        return {"linestyle": "-.", "color": "#62C370"}
    if name == "dcy+cy":
        return {"linestyle": "-.", "color": "#FFAA5A"}
    if name == "tr+cont":
        return {"linestyle": "--", "color": "#978cd9"}
    if name == "cont":
        return {"linestyle": "--", "color": "#17b3f2", "marker": "d"}
    if name == "cont+dcy":
        return {"linestyle": "--", "color": "#CC3363"}
    if name == "cont+cy":
        return {"linestyle": "--", "color": "#62C370"}
    if name == "cont+dcy+cy":
        return {"linestyle": "--", "color": "#FFAA5A"}
    if name == "tr+cont+dcy":
        return {"linestyle": "--", "marker": "d", "color": "#CC3363"}
    if name == "tr+cont+cy":
        return {"linestyle": "--", "marker": "d", "color": "#62C370"}
    if name == "tr+cont+dcy+cy":
        return {"linestyle": "--", "marker": "d", "color": "#FFAA5A"}
    return {"linestyle": "-"}


def sem_fn(x):
    return np.std(x) / np.sqrt(len(x))


def get_agg_args_from_dict(d):
    r = {}
    for attr in d.keys():
        r[attr + "_mean"] = pd.NamedAgg(column=attr, aggfunc="mean")
        r[attr + "_std"] = pd.NamedAgg(column=attr, aggfunc=sem_fn)
    return r


def set_new_cols(df, d):
    for l_name, attr in d.items():
        # average over selected losses
        df[l_name] = df[list(attr)].mean(axis=1)
    return df


def add_translation(x):
    if x['parameters/losses/coefs/supervision'] != None:
        return x['parameters/losses/coefs/supervision']
    return x['parameters/losses/coefs/translation']

def update_df_for_legacy_code(df):
    df['parameters/losses/coefs/translation'] = df.apply(add_translation, axis=1)
    if 'parameters/global_workspace/selected_domains' in df.columns:
        if isinstance(df['parameters/global_workspace/selected_domains'][0], dict):
            df['parameters/global_workspace/selected_domains'] = df['parameters/global_workspace/selected_domains'].apply(lambda x: [x[k] for k in x.keys()])
    return df


def load_df(args, language_domain):
    args = args.visualization.gw_results.axes[language_domain]
    if args.load_from == "csv":
        return update_df_for_legacy_code(pd.read_csv(Path(args.csv_path)))
    elif args.load_from == "wandb":
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_entity_project, {"config.parameters/slurm/slurm/-J": args.wandb_group_name})
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
