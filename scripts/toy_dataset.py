import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from bim_gw.utils.shapes import generate_image, generate_dataset


def save_dataset(path_root, dataset, imsize):
    dpi = 1
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, scale, rotation, color) in tqdm(enumerate(zip(classes, locations, radii, rotations, colors)), total=len(classes)):
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        generate_image(ax, cls, location, scale, rotation, color, imsize)
        ax.set_facecolor("black")
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        # plt.show()
        plt.close(fig)


def save_labels(path_root, dataset):
    classes, locations, sizes = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    colors_hls = dataset["colors_hls"]
    header = ["class", "x", "y", "scale", "rotation", "r", "g", "b", "h", "l", "s"]
    with open(path_root, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cls, location, scale, rotation, color, color_hls in zip(classes, locations, sizes, rotations, colors, colors_hls):
            writer.writerow([cls, location[0], location[1], scale, rotation, color[0], color[1], color[2],
                             color_hls[0], color_hls[1], color_hls[2]])


def main():
    seed = 0
    image_size = 32

    dataset_location = Path("/mnt/SSD/datasets/shapes_v10")
    dataset_location.mkdir(exist_ok=True)

    size_train_set = 500_000
    size_val_set = 50_000
    size_test_set = 50_000

    # in pixels
    min_scale = 10
    max_scale = 25
    min_lightness = 46  # of the HSL format. Higher value generates lighter images. Min 0, Max 256
    max_lightness = 256

    np.random.seed(seed)

    train_labels = generate_dataset(size_train_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    val_labels = generate_dataset(size_val_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    test_labels = generate_dataset(size_test_set, min_scale, max_scale, min_lightness, max_lightness, image_size)

    print("Saving training set...")
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)

    print("Save labels...")
    save_labels(dataset_location / "train_labels.csv", train_labels)
    save_labels(dataset_location / "val_labels.csv", val_labels)
    save_labels(dataset_location / "test_labels.csv", test_labels)

    print('done!')


if __name__ == '__main__':
    main()
