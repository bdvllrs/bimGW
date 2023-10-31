import os

from bim_gw.scripts.extend_shapes_dataset import extend_shapes_dataset
from bim_gw.utils import get_args

if __name__ == "__main__":
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))
    extend_shapes_dataset(
        args,
        args.datasets.shapes.n_train_examples,
        args.datasets.shapes.n_val_examples,
        args.datasets.shapes.n_test_examples,
        args.datasets.shapes.possible_categories,
        args.datasets.shapes.min_x,
        args.datasets.shapes.max_x,
        args.datasets.shapes.min_y,
        args.datasets.shapes.max_y,
        args.datasets.shapes.min_scale,
        args.datasets.shapes.max_scale,
        args.datasets.shapes.min_rotation,
        args.datasets.shapes.max_rotation,
        args.datasets.shapes.min_lightness,
        args.datasets.shapes.max_lightness,
        args.datasets.shapes.min_hue,
        args.datasets.shapes.max_hue,
    )
