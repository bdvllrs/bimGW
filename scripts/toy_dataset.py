import csv
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from bim_gw.utils.shapes import generate_image


def generate_radius(n_samples, min, max):
    assert max > min
    return np.random.randint(min, max, n_samples)


def generate_color(n_samples, max_lightness=256):
    assert 0 <= max_lightness <= 256
    hls = np.random.randint([0, 0, 0], [181, max_lightness, 256], size=(1, n_samples, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0].astype(np.float) / 255
    return rgb


def generate_rotation(n_samples):
    return np.random.rand(n_samples) * 360


def generate_location(n_samples, radius, imsize):
    assert (radius <= imsize / (2 * np.sqrt(2))).all()
    radii = np.sqrt(2) * np.stack((radius, radius), axis=1)
    locations = np.random.randint(radii, imsize - radii, (n_samples, 2))
    return locations


def generate_class(n_samples, classes):
    return np.random.randint(len(classes), size=n_samples)


def generate_dataset(n_samples, class_names, min_radius, max_radius, max_lightness, imsize):
    classes = generate_class(n_samples, class_names)
    sizes = generate_radius(n_samples, min_radius, max_radius)
    locations = generate_location(n_samples, sizes, imsize)
    rotation = generate_rotation(n_samples)
    colors = generate_color(n_samples, max_lightness)
    return dict(classes=classes, locations=locations, sizes=sizes, rotations=rotation, colors=colors)


def save_dataset(path_root, dataset, imsize):
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, radius, rotation, color) in enumerate(zip(classes, locations, radii, rotations, colors)):
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize, imsize), dpi=1)
        generate_image(ax, cls, location, radius, rotation, color, imsize)
        plt.tight_layout(pad=0)
        plt.savefig(path_file, format="png")
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
    dataset_location = Path("/mnt/SSD/datasets/shapes_v2")
    size_train_set = 50_000
    size_val_set = 5_000
    size_test_set = 5_000

    # in pixels
    min_radius = 5
    max_radius = 11
    max_lightness = 210  # of the HSL format. Higher value generates lighter images. Max is 256
    class_names = np.array(["square", "circle", "triangle"])

    np.random.seed(seed)

    train_labels = generate_dataset(size_train_set, class_names, min_radius, max_radius, max_lightness, image_size)
    val_labels = generate_dataset(size_val_set, class_names, min_radius, max_radius, max_lightness, image_size)
    test_labels = generate_dataset(size_test_set, class_names, min_radius, max_radius, max_lightness, image_size)

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
