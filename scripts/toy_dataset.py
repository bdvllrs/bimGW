import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, patches
from tqdm import tqdm

from bim_gw.utils.shapes import generate_image, generate_dataset, generate_transformations


def save_dataset(path_root, dataset, imsize):
    dpi = 1
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, scale, rotation, color) in tqdm(enumerate(zip(classes, locations, radii, rotations, colors)),
                                                           total=len(classes)):
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        generate_image(ax, cls, location, scale, rotation, color, imsize)
        ax.set_facecolor("black")
        # patch = patches.Circle((location[0], location[1]), 1, facecolor="white")
        # ax.add_patch(patch)
        plt.tight_layout(pad=0)
        # plt.show()
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)


def save_labels(path_root, dataset, dataset_transfo):
    classes, locations, sizes = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    colors_hls = dataset["colors_hls"]
    classes_transfo, locations_transfo, sizes_transfo = dataset_transfo["classes"], dataset_transfo["locations"], \
                                                        dataset_transfo["sizes"]
    rotations_transfo, colors_transfo = dataset_transfo["rotations"], dataset_transfo["colors"]
    colors_hls_transfo = dataset_transfo["colors_hls"]
    header = ["class", "x", "y", "scale", "rotation", "r", "g", "b", "h", "l", "s", "t_class", "d_x", "d_y", "d_scale",
              "d_rotation", "d_r", "d_g", "d_b", "d_h", "d_s", "d_l"]
    with open(path_root, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(classes)):
            # for cls, location, scale, rotation, color, color_hls in zip(classes, locations, sizes, rotations, colors, colors_hls):
            writer.writerow([
                classes[k], locations[k][0], locations[k][1], sizes[k], rotations[k], colors[k][0], colors[k][1],
                colors[k][2], colors_hls[k][0], colors_hls[k][1], colors_hls[k][2],
                classes_transfo[k], locations_transfo[k][0], locations_transfo[k][1], sizes_transfo[k],
                rotations_transfo[k], colors_transfo[k][0], colors_transfo[k][1], colors_transfo[k][2],
                colors_hls_transfo[k][0], colors_hls_transfo[k][1], colors_hls_transfo[k][2]
            ])


def main():
    seed = 0
    image_size = 32

    dataset_location = Path("/mnt/SSD/datasets/shapes_v13")
    dataset_location.mkdir(exist_ok=True)

    size_train_set = 1_000_000
    size_val_set = 50_000
    size_test_set = 50_000

    # in pixels
    min_scale = 10
    max_scale = 25
    min_lightness = 46  # of the HSL format. Higher value generates lighter images. Min 0, Max 256
    max_lightness = 256

    np.random.seed(seed)

    train_labels = generate_dataset(size_train_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    train_transfo_labels, train_transfo = generate_transformations(train_labels,
                                                                   generate_dataset(size_train_set, min_scale,
                                                                                    max_scale,
                                                                                    min_lightness, max_lightness,
                                                                                    image_size))
    val_labels = generate_dataset(size_val_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    val_transfo_labels, val_transfo = generate_transformations(val_labels,
                                                               generate_dataset(size_val_set, min_scale, max_scale,
                                                                                min_lightness,
                                                                                max_lightness, image_size))
    test_labels = generate_dataset(size_test_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    test_transfo_labels, test_transfo = generate_transformations(test_labels,
                                                                 generate_dataset(size_test_set, min_scale, max_scale,
                                                                                  min_lightness,
                                                                                  max_lightness, image_size))

    print("Save labels...")
    save_labels(dataset_location / "train_labels.csv", train_labels, train_transfo)
    save_labels(dataset_location / "val_labels.csv", val_labels, val_transfo)
    save_labels(dataset_location / "test_labels.csv", test_labels, test_transfo)

    print("Saving training set...")
    (dataset_location / "transformed").mkdir(exist_ok=True)
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    (dataset_location / "transformed" / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "train", train_transfo_labels, image_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    (dataset_location / "transformed" / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "val", val_transfo_labels, image_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)
    (dataset_location / "transformed" / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "test", test_transfo_labels, image_size)

    print('done!')


if __name__ == '__main__':
    main()
