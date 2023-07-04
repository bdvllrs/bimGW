import os
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

from bim_gw.utils.config import get_args
from bim_gw.utils.utils import get_runs_dataframe
from bim_gw.utils.visualization import (
    get_ax,
    get_x_range_at,
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

        x_label_long = (
            f"Number of{has_bimodal} examples {x_label_short} (N=5000)"
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            layout="constrained",
        )
        labeled_curves = []
        handles = []
        labels = []
        for m, (evaluated_loss, loss_args) in enumerate(
            loss_evaluations.items()
        ):
            for n, row in enumerate(dataframes):
                df = row["data"]
                n_col, n_row = m, n
                if figure.transpose_fig:
                    n_col, n_row = n, m
                k = n_col * n_rows + n_row
                ax = get_ax(axes, n_col, n_row, n_rows, n_cols)
                selected_curves = loss_args.curves
                for curve_name in selected_curves:
                    grp = df[df["slug"] == curve_name]
                    slug_label = curve_name
                    if curve_name in slug_to_label:
                        slug_label = slug_to_label[curve_name]

                    ax = plot_ax(
                        args,
                        evaluated_loss + "_mean",
                        ax,
                        curve_name,
                        grp,
                        labeled_curves,
                        slug_label,
                        yerr=evaluated_loss + "_std",
                    )

                    labeled_curves.append(slug_label)

                    ax.set_xlabel(
                        x_label_long if k == 0 else x_label_short,
                        fontsize=args.visualization.font_size,
                    )

                    if n_row == 0:
                        label = loss_args.label
                        if figure.transpose_fig:
                            label = row["row"].label
                        ax.set_ylabel(
                            label,
                            fontsize=args.visualization.font_size,
                        )
                    if n_col == 0:
                        label = row["row"].label
                        if figure.transpose_fig:
                            label = loss_args.label
                        ax.set_title(
                            label,
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
                ax_handles, ax_labels = ax.get_legend_handles_labels()
                handles.extend(ax_handles)
                labels.extend(ax_labels)
        order = []
        for x in figure.legend_order:
            label = slug_to_label.get(x, x)
            if label in labels:
                order.append(labels.index(label))
        fig.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
            ncol=vis_args.legend.num_columns,
            fontsize=args.visualization.font_size,
        )
        fig.get_layout_engine().set(
            hspace=vis_args.hspace, wspace=vis_args.wspace
        )
        fig.patch.set_facecolor(args.visualization.bg_color)
        plt.savefig(
            Path(vis_args.saved_figure_path) / f"{now}_results.pdf",
            bbox_inches="tight",
        )
        plt.show()
