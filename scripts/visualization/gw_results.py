import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bim_gw.utils import get_args
from bim_gw.utils.types import BIMConfig
from bim_gw.utils.utils import (
    get_job_slug_from_coefficients,
    get_runs_dataframe
)
from bim_gw.utils.visualization import (
    get_agg_args_from_dict, get_fmt,
    sem_fn,
    set_new_cols
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


if __name__ == '__main__':
    args: BIMConfig = get_args(debug=int(os.getenv("DEBUG", 0)))

    vis_args = args.visualization
    for figure in vis_args.figures:
        dataframes = []
        for row in figure.cols:
            row_label = row.label
            df = get_runs_dataframe(row)
            df['slug'] = df.apply(get_job_slug_from_coefficients, axis=1)

            tr_coef = vis_args.mix_loss_coefficients['translation']
            cont_coef = vis_args.mix_loss_coefficients['contrastive']

            loss_def_translation = vis_args.loss_definitions['contrastive']
            loss_def_contrastive = vis_args.loss_definitions['contrastive']
            df['mix_loss'] = (
                    tr_coef * df[loss_def_translation[0]]
                    + cont_coef * df[loss_def_contrastive[0]]
            )

            df = set_new_cols(df, vis_args.loss_definitions)

            df = df.groupby(
                [
                    "parameters/global_workspace/prop_labelled_images",
                    "parameters/losses/coefs/contrastive",
                    "parameters/losses/coefs/cycles",
                    "parameters/losses/coefs/demi_cycles",
                    "parameters/losses/coefs/translation",
                ], as_index=False
            )

            df = df.agg(
                mix_loss_mean=pd.NamedAgg(column='mix_loss', aggfunc="mean"),
                mix_loss_std=pd.NamedAgg(column='mix_loss', aggfunc=sem_fn),
                translation_coef=pd.NamedAgg(
                    column='parameters/losses/coefs/translation',
                    aggfunc='first'
                ),
                cycles_coef=pd.NamedAgg(
                    column='parameters/losses/coefs/cycles', aggfunc='first'
                ),
                demi_cycles_coef=pd.NamedAgg(
                    column='parameters/losses/coefs/demi_cycles',
                    aggfunc='first'
                ),
                contrastive_coef=pd.NamedAgg(
                    column='parameters/losses/coefs/contrastive',
                    aggfunc='first'
                ),
                slug=pd.NamedAgg(column='slug', aggfunc='first'),
                Name=pd.NamedAgg(column='Name', aggfunc='first'),
                prop_label=pd.NamedAgg(
                    column='parameters/global_workspace/prop_labelled_images',
                    aggfunc='first'
                ),
                **get_agg_args_from_dict(
                    vis_args.loss_definitions
                ),
            )
            df['num_examples'] = (
                    df['prop_label'] * vis_args.total_num_examples
            )
            df.fillna(0., inplace=True)
            min_idx_translation = df.groupby(["prop_label", "slug"])
            min_idx_translation = min_idx_translation['mix_loss_mean'].idxmin()
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
        for m, evaluated_loss in enumerate(loss_evaluations):
            for n, row in enumerate(dataframes):
                df = row['data']
                k = m * n_rows + n
                ax = _get_ax(axes, m, n, n_rows, n_cols)
                selected_curves = figure.selected_curves
                for curve_name in selected_curves:
                    grp = df[df['slug'] == curve_name]
                    slug_label = curve_name
                    if curve_name in slug_to_label:
                        slug_label = slug_to_label[curve_name]
                    if len(grp) > 1:
                        ax = grp.plot(
                            'num_examples', evaluated_loss + '_mean', ax=ax,
                            yerr=evaluated_loss + "_std",
                            label=(slug_label if k == 0 else '_nolegend_'),
                            legend=False, **get_fmt(curve_name),
                            linewidth=args.visualization.line_width
                        )
                    else:
                        ax.axhline(
                            y=grp[evaluated_loss + "_mean"].iloc[0],
                            label=(slug_label if k == 0 else '_nolegend_'),
                            **get_fmt(curve_name)
                        )

                    ax.set_xlabel(
                        "Number of bimodal examples ($N$)",
                        fontsize=args.visualization.font_size
                    )
                    if n == 0:
                        ax.set_ylabel(
                            y_axis_labels[evaluated_loss],
                            fontsize=args.visualization.font_size
                        )
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                    if m == 0:
                        ax.set_title(
                            row['row'].label,
                            fontsize=args.visualization.font_size_title,
                            color=args.visualization.fg_color
                        )

                ax.tick_params(
                    which='both', labelsize=args.visualization.font_size,
                    color=args.visualization.fg_color,
                    labelcolor=args.visualization.fg_color
                )
                ax.xaxis.label.set_color(args.visualization.fg_color)
                ax.yaxis.label.set_color(args.visualization.fg_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(args.visualization.fg_color)
                ax.set_facecolor(args.visualization.bg_color)

        fig.legend(
            loc='lower center', bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
            ncol=vis_args.legend.num_columns,
            fontsize=args.visualization.font_size
        )
        # fig.suptitle(figure.title)
        # fig.tight_layout()
        plt.subplots_adjust(
            bottom=args.visualization.bottom_adjust,
            hspace=args.visualization.hspace_adjust,
            top=args.visualization.top_adjust
        )
        fig.patch.set_facecolor(args.visualization.bg_color)
        plt.savefig(
            Path(
                vis_args.saved_figure_path
            ) / f"{now}_results.pdf", bbox_inches="tight"
        )
        plt.show()

        # PLOT COEFFICIENTS
        # coefs = ["cycles", "demi_cycles", "contrastive"]
        # fig, axes = plt.subplots(
        #     len(coefs), n_cols, figsize=(3.7 * n_cols, 4 * len(coefs))
        # )
        # for m, coef in enumerate(coefs):
        #     for n, (df_name, df) in enumerate(dataframes.items()):
        #         k = m * len(coefs) + n
        #         ax = axes[m, n] if n_cols > 1 else axes
        #         selected_curves = figure.selected_curves
        #         for slug in selected_curves:
        #             grp = df[df['slug'] == slug]
        #             slug_label = slug
        #             if slug in slug_to_label:
        #                 slug_label = slug_to_label[slug]
        #             if len(grp) > 1:
        #                 ax = grp.plot(
        #                     'num_examples', coef + '_coef', ax=ax,
        #                     label=(slug_label if k == 0 else '_nolegend_'),
        #                     legend=False, **get_fmt(slug),
        #                     linewidth=args.visualization.line_width
        #                 )
        #
        #             ax.set_xlabel(
        #                 "Number of bimodal examples ($N$)",
        #                 fontsize=args.visualization.font_size
        #             )
        #             if n == 0:
        #                 ax.set_ylabel(coef,
        #                 fontsize=args.visualization.font_size)
        #             ax.set_xscale('log')
        #             if m == 0:
        #                 ax.set_title(
        #                     title_labels[df_name],
        #                     fontsize=args.visualization.font_size_title,
        #                     color=args.visualization.fg_color
        #                 )
        #
        #         ax.tick_params(
        #             which='both', labelsize=args.visualization.font_size,
        #             color=args.visualization.fg_color,
        #             labelcolor=args.visualization.fg_color
        #         )
        #         ax.xaxis.label.set_color(args.visualization.fg_color)
        #         ax.yaxis.label.set_color(args.visualization.fg_color)
        #         for spine in ax.spines.values():
        #             spine.set_edgecolor(args.visualization.fg_color)
        #         ax.set_facecolor(args.visualization.bg_color)
        #
        # fig.legend(
        #     loc='lower center', bbox_to_anchor=(0.5, 0),
        #     bbox_transform=fig.transFigure,
        #     ncol=vis_args.legend.num_columns,
        #     fontsize=args.visualization.font_size
        # )
        # # fig.tight_layout()
        # plt.subplots_adjust(bottom=0.11, hspace=0.3, top=0.97)
        # fig.patch.set_facecolor(args.visualization.bg_color)
        # plt.savefig(
        #     Path(
        #         vis_args.saved_figure_path
        #     ) / f"{now}_selected_coefficients.pdf",
        #     bbox_inches="tight"
        # )
        # plt.show()
