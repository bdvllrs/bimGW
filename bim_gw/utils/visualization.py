import numpy as np
import pandas as pd


def get_fmt(name):
    if name == "tr":
        return {"linestyle": "-", "color": "#17b3f2"}
    if name == "tr+cy":
        return {"linestyle": "-", "marker": "d", "color": "#62C370"}
    if name == "tr+cont":
        return {"linestyle": "--", "color": "#CC3363"}
    if name == "tr+cont+dcy+cy":
        return {"linestyle": "--", "marker": "d", "color": "#FFAA5A"}
    if name == "tr+dcy+cy":
        return {"linestyle": "-", "color": "#d124ff"}
    if name == "tr+dcy":
        return {"linestyle": "-", "color": "#e1a4e6"}
    if name == "cont":
        return {"linestyle": "--", "color": "#043505"}


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
