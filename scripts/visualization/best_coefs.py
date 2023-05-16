import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import log

from bim_gw.utils import get_args
from bim_gw.utils.types import BIMConfig
from bim_gw.utils.utils import (
    get_job_detailed_slug_from_coefficients,
    get_job_slug_from_coefficients,
    get_runs_dataframe,
)
from bim_gw.utils.visualization import (
    get_agg_args_from_dict,
    sem_fn,
    set_new_cols,
)

y_axis_labels = {
    "translation": "Translation losses",
    "contrastive": "Contrastive losses",
    "cycles": "Cycle losses",
    "demi_cycles": "Demi-cycle losses",
    "mix_loss": "Averaged losses",
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
}


def _get_ax(
    axes: np.ndarray, row: int, col: int, num_rows: int, num_cols: int
) -> np.ndarray:
    if num_cols == 1 and num_rows == 1:
        return axes
    if num_cols == 1:
        return axes[row]
    if num_rows == 1:
        return axes[col]
    return axes[row, col]


def get_x_range_at(
    data_start: pd.DataFrame,
    data_end: pd.DataFrame,
    y_val: float,
    diff_col: str,
) -> Tuple[float, float]:
    order_start = data_start["num_examples"].sort_values().index
    order_end = data_end["num_examples"].sort_values().index
    x_axis_start = data_start["num_examples"].loc[order_start]
    x_axis_end = data_end["num_examples"].loc[order_end]
    data_start = data_start[diff_col].loc[order_start]
    data_end = data_end[diff_col].loc[order_end]
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


if __name__ == "__main__":
    args: BIMConfig = get_args(debug=int(os.getenv("DEBUG", 0)))

    vis_args = args.visualization
    for figure in vis_args.figures:
        dataframes = []
        for row in figure.cols:
            row_label = row.label
            df = get_runs_dataframe(row)
            df["slug"] = df.apply(get_job_slug_from_coefficients, axis=1)
            df["detailed_slug"] = df.apply(
                get_job_detailed_slug_from_coefficients, axis=1
            )

            tr_coef = vis_args.mix_loss_coefficients["translation"]
            cont_coef = vis_args.mix_loss_coefficients["contrastive"]

            loss_def_translation = vis_args.loss_definitions["translation"]
            loss_def_contrastive = vis_args.loss_definitions["contrastive"]
            df["mix_loss"] = df[loss_def_translation[0]]

            df = set_new_cols(df, vis_args.loss_definitions)

            df = df.groupby(
                [
                    "parameters/global_workspace/prop_labelled_images",
                    "parameters/losses/coefs/contrastive",
                    "parameters/losses/coefs/cycles",
                    "parameters/losses/coefs/demi_cycles",
                    "parameters/losses/coefs/translation",
                ],
                as_index=False,
            )

            df = df.agg(
                mix_loss_mean=pd.NamedAgg(column="mix_loss", aggfunc="mean"),
                mix_loss_std=pd.NamedAgg(column="mix_loss", aggfunc=sem_fn),
                translation_coef=pd.NamedAgg(
                    column="parameters/losses/coefs/translation",
                    aggfunc="first",
                ),
                cycles_coef=pd.NamedAgg(
                    column="parameters/losses/coefs/cycles", aggfunc="first"
                ),
                demi_cycles_coef=pd.NamedAgg(
                    column="parameters/losses/coefs/demi_cycles",
                    aggfunc="first",
                ),
                contrastive_coef=pd.NamedAgg(
                    column="parameters/losses/coefs/contrastive",
                    aggfunc="first",
                ),
                slug=pd.NamedAgg(column="slug", aggfunc="first"),
                detailed_slug=pd.NamedAgg(
                    column="detailed_slug", aggfunc="first"
                ),
                Name=pd.NamedAgg(column="Name", aggfunc="first"),
                prop_label=pd.NamedAgg(
                    column="parameters/global_workspace/prop_labelled_images",
                    aggfunc="first",
                ),
                **get_agg_args_from_dict(vis_args.loss_definitions),
            )
            df["num_examples"] = df["prop_label"] * vis_args.total_num_examples
            df.fillna(0.0, inplace=True)
            min_idx_translation = df.groupby(["prop_label", "detailed_slug"])
            min_idx_translation = min_idx_translation["mix_loss_mean"].idxmin()
            df = df.loc[min_idx_translation]
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
            n_rows, n_cols, figsize=(3.7 * n_cols, 4 * n_rows)
        )
        labeled_curves = []
        for m, (evaluated_loss, loss_args) in enumerate(
            loss_evaluations.items()
        ):
            for n, row in enumerate(dataframes):
                df = row["data"]
                k = m * n_rows + n
                ax = _get_ax(axes, m, n, n_rows, n_cols)
                selected_curve = loss_args.coef_curve
                grp = df[df["slug"] == selected_curve]
                possible_labels = grp["detailed_slug"].unique()
                for curve_name in possible_labels:
                    grp = df[df["detailed_slug"] == curve_name]
                    slug_label = curve_name
                    if curve_name in slug_to_label:
                        slug_label = slug_to_label[curve_name]
                    if len(grp) > 1:
                        ax = grp.plot(
                            "num_examples",
                            evaluated_loss + "_mean",
                            ax=ax,
                            yerr=evaluated_loss + "_std",
                            label=(
                                slug_label
                                if slug_label not in labeled_curves
                                else "_nolegend_"
                            ),
                            legend=True,
                            linewidth=args.visualization.line_width,
                        )
                    elif len(grp) == 1:
                        ax.axhline(
                            y=grp[evaluated_loss + "_mean"].iloc[0],
                            label=(
                                slug_label
                                if slug_label not in labeled_curves
                                else "_nolegend_"
                            ),
                        )
                    else:
                        logging.warning(f"No data for {curve_name}")
                    # labeled_curves.append(slug_label)

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
                ax.set_yscale("log")
                ax.set_xscale("log")
        # fig.legend(
        #     loc='lower center', bbox_to_anchor=(0.5, 0),
        #     bbox_transform=fig.transFigure,
        #     ncol=vis_args.legend.num_columns,
        #     fontsize=args.visualization.font_size
        # )
        # fig.suptitle(figure.title)
        # fig.tight_layout()
        plt.subplots_adjust(
            bottom=figure.bottom_adjust,
            hspace=figure.hspace_adjust,
            top=figure.top_adjust,
        )
        fig.patch.set_facecolor(args.visualization.bg_color)
        plt.savefig(
            Path(vis_args.saved_figure_path) / f"{now}_coefs.pdf",
            bbox_inches="tight",
        )
        plt.show()
