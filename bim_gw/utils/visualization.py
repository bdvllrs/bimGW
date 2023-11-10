import logging
from typing import Any, Callable, Dict, Mapping, Tuple, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import log

from bim_gw.utils.utils import (
    get_additional_slug,
    get_job_slug_from_coefficients,
)


def get_fmt(name: str) -> Dict[str, Any]:
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
    if "baseline:identity" in name:
        return {"linestyle": "-", "color": "#000"}
    if "baseline:random" in name:
        return {"linestyle": "--", "color": "#000"}
    if "baseline:none" in name:
        return {"linestyle": "-.", "color": "#000"}
    return {"linestyle": "-"}


def get_fmt_all(name: str) -> Mapping[str, Any]:
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


slug_to_label = {
    "tr": "translation",
    "dcy": "demi-cycles",
    "cy": "domain cycles",
    "dcy+cy": "all cycles",
    "tr+dcy+cy": "trans. + all cycles",
    "tr+dcy": "trans. + demi-cycles",
    "tr+cy": "trans. + full cycles",
    "tr+cont": "trans. + cont.",
    "cont": "contrastive",
    "tr+cont+dcy+cy": "all sup. + all cycles",
    "baseline:identity": "no encoder + TO classifier",
    "baseline:none": "Task Optimizer (TO) encoder + classifier",
    "baseline:random": "random encoder + TO classifier",
    "tr+cont+dcy+cy+baseline:identity": "no encoder + TO classifier",
    "tr+cont+dcy+cy+baseline:none": "Task Optimizer (TO) encoder + classifier",
    "tr+cont+dcy+cy+baseline:random": "random encoder + TO classifier",
}


def get_ax(
    axes: np.ndarray,
    row: int,
    col: int,
    num_rows: int,
    num_cols: int,
) -> plt.Axes:
    if num_cols == 1 and num_rows == 1:
        return cast(plt.Axes, axes)
    if num_cols == 1:
        return cast(plt.Axes, axes[row])
    if num_rows == 1:
        return cast(plt.Axes, axes[col])
    return cast(plt.Axes, axes[row, col])


def get_x_range_at(
    data_start: pd.DataFrame,
    data_end: pd.DataFrame,
    y_val: float,
    diff_col: str,
) -> Tuple[float, float]:
    order_start = data_start["num_examples"].sort_values().index  # type:ignore
    order_end = data_end["num_examples"].sort_values().index  # type:ignore
    x_axis_start = data_start["num_examples"].loc[order_start]  # type:ignore
    x_axis_end = data_end["num_examples"].loc[order_end]  # type:ignore
    data_start = data_start[diff_col].loc[order_start]  # type:ignore
    data_end = data_end[diff_col].loc[order_end]  # type:ignore
    data_start_log_diff = log(data_start).diff()
    data_end_log_diff = log(data_end).diff()

    # Find the two closest points to y_val
    start_close_index = data_start.iloc[
        (data_start - y_val).abs().argsort()[:1]
    ].index.item()
    end_close_index = data_end.iloc[
        (data_end - y_val).abs().argsort()[:1]
    ].index.item()
    start_next_index = data_start.index.tolist().index(start_close_index)

    if start_next_index < len(data_start) - 1:
        start_next_index = data_start.index[start_next_index + 1]
    end_next_index = data_end.index.tolist().index(end_close_index)
    if end_next_index < len(data_end) - 1:
        end_next_index = data_end.index[end_next_index + 1]

    if (
        data_start[start_close_index] <= y_val <= data_start[start_next_index]
        or data_start[start_next_index]
        <= y_val
        <= data_start[start_close_index]
    ):
        start_close_index = start_next_index
    if (
        data_end[end_close_index] <= y_val <= data_end[end_next_index]
        or data_end[end_next_index] <= y_val <= data_end[end_close_index]
    ):
        end_close_index = end_next_index

    ref_start = (
        log(x_axis_start[start_close_index])
        + (log(y_val) - log(data_start[start_close_index]))
        * log(x_axis_start).diff()[start_close_index]
        / data_start_log_diff[start_close_index]
    )
    ref_end = (
        log(x_axis_end[end_close_index])
        + (np.log(y_val) - log(data_end[end_close_index]))
        * log(x_axis_end).diff()[end_close_index]
        / data_end_log_diff[end_close_index]
    )

    return np.exp(ref_start), np.exp(ref_end)


def prepare_df(df, vis_args):
    df["additional_slug"] = df.apply(
        get_additional_slug,
        axis=1,
        additional_conds={
            slug_cond.slug_value: get_additional_slug_cond(
                slug_cond.key, slug_cond.eq
            )
            for slug_cond in vis_args.additional_slug_conds
        },
    )

    def additional_slug(x):
        return x["additional_slug"]

    df["slug"] = df.apply(
        get_job_slug_from_coefficients,
        axis=1,
        additional_slug=additional_slug,
    )

    df = set_new_cols(df, vis_args.loss_definitions)

    group_by_params = [
        "parameters/global_workspace/prop_labelled_images",
        "parameters/global_workspace/prop_available_images",
        "parameters/losses/coefs/contrastive",
        "parameters/losses/coefs/cycles",
        "parameters/losses/coefs/demi_cycles",
        "parameters/losses/coefs/translation",
        "additional_slug",
    ]

    df = df.groupby(
        group_by_params,
        as_index=False,
    )

    df = df.agg(
        translation_coef=pd.NamedAgg(
            column="parameters/losses/coefs/translation", aggfunc="first"
        ),
        cycles_coef=pd.NamedAgg(
            column="parameters/losses/coefs/cycles", aggfunc="first"
        ),
        demi_cycles_coef=pd.NamedAgg(
            column="parameters/losses/coefs/demi_cycles", aggfunc="first"
        ),
        contrastive_coef=pd.NamedAgg(
            column="parameters/losses/coefs/contrastive", aggfunc="first"
        ),
        slug=pd.NamedAgg(column="slug", aggfunc="first"),
        Name=pd.NamedAgg(column="Name", aggfunc="first"),
        prop_labelled=pd.NamedAgg(
            column="parameters/global_workspace/prop_labelled_images",
            aggfunc="first",
        ),
        prop_available=pd.NamedAgg(
            column="parameters/global_workspace/prop_available_images",
            aggfunc="first",
        ),
        **get_agg_args_from_dict(vis_args.loss_definitions),
    )
    df["num_examples"] = df[vis_args.x_axis] * vis_args.total_num_examples
    min_idx_translation = df.groupby(
        ["prop_labelled", "prop_available", "slug"]
    )
    min_idx_translation = min_idx_translation[vis_args.argmin_over].idxmin()
    df = df.loc[min_idx_translation]
    return df


def plot_ax(
    args,
    loss,
    ax,
    curve_name,
    grp,
    labeled_curves,
    slug_label,
    yerr=None,
):
    if len(grp) > 1:
        ax = grp.plot(
            "num_examples",
            loss,
            ax=ax,
            yerr=yerr,
            label=(
                slug_label
                if slug_label not in labeled_curves
                else "_nolegend_"
            ),
            legend=False,
            **get_fmt(curve_name),
            linewidth=args.visualization.line_width,
        )
    elif len(grp) == 1:
        ax.axhline(
            y=grp[loss].iloc[0],
            label=(
                slug_label
                if slug_label not in labeled_curves
                else "_nolegend_"
            ),
            **get_fmt(curve_name),
            linewidth=args.visualization.line_width,
        )
    else:
        logging.warning(f"No data for {curve_name}")
    return ax


def plot_ax_bars(
    args,
    k,
    loss,
    yerr,
    ax,
    curve_name,
    grp,
    labeled_curves,
    slug_label,
):
    fmt = get_fmt(curve_name)
    if "marker" in fmt:
        del fmt["marker"]
    if "linestyle" in fmt:
        del fmt["linestyle"]
    ax.bar(
        k * 3,
        grp[loss].iloc[0],
        yerr=grp[yerr].iloc[0],
        width=1,
        label=(
            slug_label if slug_label not in labeled_curves else "_nolegend_"
        ),
        edgecolor="black",
        linewidth=2,
        **fmt,
    )
    fmt["hatch"] = "//"
    ax.bar(
        k * 3 + 1,
        grp["ood_" + loss].iloc[0],
        width=1,
        label=(
            slug_label if slug_label not in labeled_curves else "_nolegend_"
        ),
        edgecolor="black",
        linewidth=2,
        **fmt,
    )
    # ax = grp.plot.bar(
    #     "num_examples",
    #     loss,
    #     rot=0,
    #     ax=ax,
    #     label=,
    #     legend=False,
    #     stacked=False,
    #     **fmt,
    # )
    return ax


def get_additional_slug_cond(
    key: str, target: str
) -> Callable[[pd.Series], bool]:
    def fn(x: pd.Series) -> bool:
        return x[key] == target

    return fn
