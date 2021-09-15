import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from bim_gw.utils.shapes import generate_image, generate_dataset


def save_dataset(path_root, dataset, imsize):
    dpi = 1
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, radius, rotation, color) in enumerate(zip(classes, locations, radii, rotations, colors)):
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        generate_image(ax, cls, location, radius, rotation, color, imsize)
        ax.set_facecolor("black")
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        # plt.show()
        plt.close(fig)


def save_labels(path_root, dataset):
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    header = ["class", "x", "y", "radius", "rotation", "r", "g", "b"]
    with open(path_root, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cls, location, radius, rotation, color in zip(classes, locations, radii, rotations, colors):
            writer.writerow([cls, location[0], location[1], radius, rotation, color[0], color[1], color[2]])


def main():
    seed = 0
    image_size = 32
    dataset_location = Path("/mnt/SSD/datasets/shapes_v4")
    size_train_set = 500_000
    size_val_set = 50_000
    size_test_set = 50_000

    # in pixels
    min_radius = 5
    max_radius = 11
    min_lightness = 46  # of the HSL format. Higher value generates lighter images. Max is 256
    max_lightness = 256  # of the HSL format. Higher value generates lighter images. Max is 256
    class_names = np.array(["square", "circle", "triangle"])

    np.random.seed(seed)

    train_labels = generate_dataset(size_train_set, class_names, min_radius, max_radius, min_lightness, max_lightness, image_size)
    val_labels = generate_dataset(size_val_set, class_names, min_radius, max_radius, min_lightness, max_lightness, image_size)
    test_labels = generate_dataset(size_test_set, class_names, min_radius, max_radius, min_lightness, max_lightness, image_size)

    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)

    save_labels(dataset_location / "train_labels.csv", train_labels)
    save_labels(dataset_location / "val_labels.csv", val_labels)
    save_labels(dataset_location / "test_labels.csv", test_labels)

    print('done!')


if __name__ == '__main__':
    main()
