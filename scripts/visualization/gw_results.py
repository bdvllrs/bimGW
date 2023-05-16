import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import log

from bim_gw.utils import get_args
from bim_gw.utils.utils import (
    get_additional_slug,
    get_job_slug_from_coefficients,
    get_runs_dataframe,
)
from bim_gw.utils.visualization import (
    get_agg_args_from_dict,
    get_fmt,
    set_new_cols,
)

y_axis_labels = {
    "translation": "Translation losses",
    "contrastive": "Contrastive losses",
    "cycles": "Cycle losses",
    "demi_cycles": "Demi-cycle losses",
    "mix_loss": "Averaged losses",
    "ooo_acc_v": "Visual Accuracy",
    "ooo_acc_t": "Language Accuracy",
    "ooo_acc_attr": "Proto-language Accuracy",
}

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
    "baseline:identity": "Baseline: no encoder",
    "baseline:none": "Baseline",
    "baseline:random": "Random performance",
}


def _get_ax(
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
    # if start_next_index > 1:
    #     start_next_index = data_start.index[start_next_index - 1]
    # end_next_index = data_end.index.tolist().index(end_close_index)
    # if end_next_index > 1:
    #     end_next_index = data_end.index[end_next_index - 1]

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
    # x = x_r + (y - y_r) * (x_r+1 - x_r) / (y_r+1 - y_r)
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
    # df['mix_loss'] = (
    #     df[loss_def_translation[0]]
    # )

    df = set_new_cols(df, vis_args.loss_definitions)

    group_by_params = [
        "parameters/global_workspace/prop_labelled_images",
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
        prop_label=pd.NamedAgg(
            column="parameters/global_workspace/prop_labelled_images",
            aggfunc="first",
        ),
        **get_agg_args_from_dict(vis_args.loss_definitions),
    )
    df["num_examples"] = df["prop_label"] * vis_args.total_num_examples
    df.fillna(0.0, inplace=True)
    min_idx_translation = df.groupby(["prop_label", "slug"])
    min_idx_translation = min_idx_translation[vis_args.argmin_over].idxmin()
    df = df.loc[min_idx_translation]
    return df


def plot_ax(args, loss, ax, curve_name, grp, labeled_curves, slug_label):
    if len(grp) > 1:
        ax = grp.plot(
            "num_examples",
            loss + "_mean",
            ax=ax,
            yerr=loss + "_std",
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
            y=grp[loss + "_mean"].iloc[0],
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


def get_additional_slug_cond(
    key: str, target: str
) -> Callable[[pd.Series], bool]:
    def fn(x: pd.Series) -> bool:
        return x[key] == target

    return fn


if __name__ == "__main__":
    args = get_args(debug=bool(os.getenv("DEBUG", 0)))

    vis_args = args.visualization
    for figure in vis_args.figures:
        dataframes = []
        for row in figure.cols:
            row_label = row.label
            df = get_runs_dataframe(row)
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

            df = prepare_df(df, vis_args)

            dataframes.append(
                {
                    "row": row,
                    "data": df,
                }
            )

        loss_evaluations = figure.selected_losses

        n_cols = len(figure.cols)
        n_rows = len(loss_evaluations)
        now = datetime.now().strftime("%d-%m-%YT%H_%M_%S")

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows)
        )
        labeled_curves = []
        handles = []
        labels = []
        for m, (evaluated_loss, loss_args) in enumerate(
            loss_evaluations.items()
        ):
            for n, row in enumerate(dataframes):
                df = row["data"]
                k = m * n_rows + n
                ax = _get_ax(axes, m, n, n_rows, n_cols)
                selected_curves = loss_args.curves
                for curve_name in selected_curves:
                    grp = df[df["slug"] == curve_name]
                    slug_label = curve_name
                    if curve_name in slug_to_label:
                        slug_label = slug_to_label[curve_name]

                    ax = plot_ax(
                        args,
                        evaluated_loss,
                        ax,
                        curve_name,
                        grp,
                        labeled_curves,
                        slug_label,
                    )

                    labeled_curves.append(slug_label)

                    ax.set_xlabel(
                        "Number of bimodal examples ($N$)",
                        fontsize=args.visualization.font_size,
                    )
                    if n == 0:
                        ax.set_ylabel(
                            y_axis_labels[evaluated_loss],
                            fontsize=args.visualization.font_size,
                        )
                    if m == 0:
                        ax.set_title(
                            row["row"].label,
                            fontsize=args.visualization.font_size_title,
                            color=args.visualization.fg_color,
                        )

                ax.tick_params(
                    which="both",
                    labelsize=args.visualization.font_size,
                    color=args.visualization.fg_color,
                    labelcolor=args.visualization.fg_color,
                )
                ax.xaxis.label.set_color(args.visualization.fg_color)
                ax.yaxis.label.set_color(args.visualization.fg_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(args.visualization.fg_color)
                ax.set_facecolor(args.visualization.bg_color)

                for annotation in row["row"].annotations:
                    if evaluated_loss == annotation.loss:
                        x_start, x_end = get_x_range_at(
                            df[df["slug"] == annotation.curve_start],
                            df[df["slug"] == annotation.curve_end],
                            annotation.y,
                            f"{evaluated_loss}_mean",
                        )

                        ratio = max(x_end / x_start, x_start / x_end)
                        ax.text(
                            x_start,
                            annotation.y + annotation.text_yshift,
                            f"x{ratio:.1f}",
                        )
                        ax.annotate(
                            "",
                            xy=(x_start, annotation.y),
                            xytext=(x_end, annotation.y),
                            arrowprops=dict(arrowstyle="<->"),
                        )
                # ax.set_yscale("log")
                ax.set_xscale("log")
                # ax.ticklabel_format(useOffset=False, style='plain')
                ax_handles, ax_labels = ax.get_legend_handles_labels()
                handles.extend(ax_handles)
                labels.extend(ax_labels)
        order = [
            labels.index(slug_to_label.get(x, x)) for x in figure.legend_order
        ]
        fig.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
            ncol=vis_args.legend.num_columns,
            fontsize=args.visualization.font_size,
        )
        # fig.suptitle(figure.title)
        # fig.tight_layout()
        plt.subplots_adjust(
            bottom=figure.bottom_adjust,
            hspace=figure.hspace_adjust,
            top=figure.top_adjust,
            left=figure.left_adjust,
            right=figure.right_adjust,
        )
        fig.patch.set_facecolor(args.visualization.bg_color)
        plt.savefig(
            Path(vis_args.saved_figure_path) / f"{now}_results.pdf",
            bbox_inches="tight",
        )
        plt.show()

        # PLOT COEFFICIENTS
        coefs = ["cycles", "demi_cycles", "contrastive"]
        fig, axes = plt.subplots(
            len(coefs), n_cols, figsize=(3.7 * n_cols, 4 * len(coefs))
        )
        selected_curves = []
        for selected_loss, loss_args in figure.selected_losses.items():
            for curve in loss_args.curves:
                if curve not in selected_curves:
                    selected_curves.append(curve)
        for m, coef in enumerate(coefs):
            for n, row in enumerate(dataframes):
                df = row["data"]
                k = m * len(coefs) + n
                ax = _get_ax(axes, m, n, n_rows, n_cols)
                for slug in selected_curves:
                    grp = df[df["slug"] == slug]
                    slug_label = slug
                    if slug in slug_to_label:
                        slug_label = slug_to_label[slug]
                    if len(grp) > 1:
                        ax = grp.plot(
                            "num_examples",
                            coef + "_coef",
                            ax=ax,
                            label=(slug_label if k == 0 else "_nolegend_"),
                            legend=False,
                            **get_fmt(slug),
                            linewidth=args.visualization.line_width,
                        )

                    ax.set_xlabel(
                        "Number of bimodal examples ($N$)",
                        fontsize=args.visualization.font_size,
                    )
                    if n == 0:
                        ax.set_ylabel(
                            coef, fontsize=args.visualization.font_size
                        )
                    ax.set_xscale("log")
                    if m == 0:
                        ax.set_title(
                            row["row"].label,
                            fontsize=args.visualization.font_size_title,
                            color=args.visualization.fg_color,
                        )

                ax.tick_params(
                    which="both",
                    labelsize=args.visualization.font_size,
                    color=args.visualization.fg_color,
                    labelcolor=args.visualization.fg_color,
                )
                ax.xaxis.label.set_color(args.visualization.fg_color)
                ax.yaxis.label.set_color(args.visualization.fg_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(args.visualization.fg_color)
                ax.set_facecolor(args.visualization.bg_color)

        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
            ncol=vis_args.legend.num_columns,
            fontsize=args.visualization.font_size,
        )
        # fig.tight_layout()
        plt.subplots_adjust(bottom=0.11, hspace=0.3, top=0.97)
        fig.patch.set_facecolor(args.visualization.bg_color)
        plt.savefig(
            Path(vis_args.saved_figure_path)
            / f"{now}_selected_coefficients.pdf",
            bbox_inches="tight",
        )
        plt.show()
