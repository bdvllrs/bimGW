import os
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

from bim_gw.utils.config import get_args
from bim_gw.utils.utils import get_runs_dataframe
from bim_gw.utils.visualization import (
    get_ax,
    plot_ax,
    prepare_df,
    slug_to_label,
)

if __name__ == "__main__":
    args = get_args(debug=bool(bool(int(os.getenv("DEBUG", 0)))))

    vis_args = args.visualization
    for figure in vis_args.figures:
        dataframes = []
        for row in figure.cols:
            row_label = row.label
            df = get_runs_dataframe(row)

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
        if figure.transpose_fig:
            n_cols, n_rows = n_rows, n_cols

        now = datetime.now().strftime("%d-%m-%YT%H_%M_%S")

        x_label_short = (
            "$N$" if vis_args.x_axis == "prop_labelled" else "$N+M$"
        )
        has_bimodal = " bimodal" if vis_args.x_axis == "prop_labelled" else ""

        x_label_long = f"Number of{has_bimodal} examples ({x_label_short})"

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
        labeled_curves = []
        for m, coef in enumerate(coefs):
            for n, row in enumerate(dataframes):
                df = row["data"]
                k = m * len(coefs) + n
                ax = get_ax(axes, m, n, n_rows, n_cols)
                for slug in selected_curves:
                    grp = df[df["slug"] == slug]
                    slug_label = slug
                    if slug in slug_to_label:
                        slug_label = slug_to_label[slug]
                    ax = plot_ax(
                        args,
                        coef + "_coef",
                        ax,
                        slug,
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
