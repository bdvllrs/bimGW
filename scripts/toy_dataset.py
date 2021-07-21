import csv
from pathlib import Path

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt


def get_square_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    patch = patches.Rectangle((x - radius, y - radius), 2 * radius, 2 * radius, rotation, facecolor=color)
    return patch


def get_triangle_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    coordinates = np.array([[radius, -radius],
                            [0, radius],
                            [radius, radius]])
    origin = np.array([x, y])
    rotation_m = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    patch = patches.Polygon(origin + coordinates @ rotation_m, facecolor=color)
    return patch


def get_circle_patch(location, radius, rotation, color):
    x, y = location[0], location[1]
    patch = patches.Circle((x, y), radius, facecolor=color)
    return patch


def generate_image(path, cls, location, radius, rotation, color, imsize=32):
    if cls == 0:
        patch = get_square_patch(location, radius, rotation, color)
    elif cls == 1:
        patch = get_circle_patch(location, radius, rotation, color)
    elif cls == 2:
        patch = get_triangle_patch(location, radius, rotation, color)
    else:
        raise ValueError("Class does not exist.")

    fig, ax = plt.subplots(figsize=(imsize, imsize), dpi=1)

    ax.add_patch(patch)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)
    plt.tight_layout(pad=0)
    plt.savefig(path, format="png")
    # plt.show()


def generate_radius(n_samples, min, max):
    assert max > min
    return np.random.randint(min, max, n_samples)


def generate_color(n_samples):
    return np.random.rand(n_samples, 3)


def generate_rotation(n_samples):
    return np.random.rand(n_samples) * 360


def generate_location(n_samples, radius, imsize):
    radii = (1 / np.sqrt(2)) * np.stack((radius, radius), axis=1)
    locations = np.random.randint(radii, imsize - radii, (n_samples, 2))
    return locations


def generate_class(n_samples, classes):
    return np.random.randint(len(classes), size=n_samples)


def generate_dataset(n_samples, class_names, min_radius, max_radius, imsize):
    classes = generate_class(n_samples, class_names)
    sizes = generate_radius(n_samples, min_radius, max_radius)
    locations = generate_location(n_samples, sizes, imsize)
    rotation = generate_rotation(n_samples)
    colors = generate_color(n_samples)
    return dict(classes=classes, locations=locations, sizes=sizes, rotations=rotation, colors=colors)


def save_dataset(path_root, dataset, imsize):
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, radius, rotation, color) in enumerate(zip(classes, locations, radii, rotations, colors)):
        path_file = path_root / f"{k}.png"
        generate_image(path_file, cls, location, radius, rotation, color, imsize)


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
    image_size = 32
    dataset_location = Path("/mnt/SSD/datasets/shapes")
    size_train_set = 50_000
    size_val_set = 5_000
    size_test_set = 5_000

    # in pixels
    min_radius = 3
    max_radius = 14
    class_names = np.array(["square", "circle", "triangle"])

    train_labels = generate_dataset(size_train_set, class_names, min_radius, max_radius, image_size)
    val_labels = generate_dataset(size_val_set, class_names, min_radius, max_radius, image_size)
    test_labels = generate_dataset(size_test_set, class_names, min_radius, max_radius, image_size)

    (dataset_location / "train").mkdir()
    save_dataset(dataset_location / "train", train_labels, image_size)
    (dataset_location / "val").mkdir()
    save_dataset(dataset_location / "val", val_labels, image_size)
    (dataset_location / "test").mkdir()
    save_dataset(dataset_location / "test", test_labels, image_size)

    save_labels(dataset_location / "train_labels.txt", train_labels)
    save_labels(dataset_location / "val_labels.txt", val_labels)
    save_labels(dataset_location / "test_labels.txt", test_labels)

    print('done!')


if __name__ == '__main__':
    main()
