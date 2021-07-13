from collections import OrderedDict
from pathlib import Path
import csv

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt


def get_random_population(categories, size):
    cls = []
    for attr_name, attrs in categories.items():
        cls.append(np.random.randint(len(attrs), size=size))
    return np.vstack(cls).transpose()


def create_image_from_label(label, imsize, location):
    fig, ax = plt.subplots(figsize=(5, 5))

    x, y = 0, 0

    side = label["location"]
    if side[1] == "l":
        x = imsize / 6
    elif side[1] == "c":
        x = imsize / 2
    elif side[1] == "r":
        x = 5 * imsize / 6
    if side[0] == "b":
        y = imsize / 6
    elif side[0] == "m":
        y = imsize / 2
    elif side[0] == "t":
        y = 5 * imsize / 6
    else:
        raise ValueError

    if label["size"] == "large":
        radius = 0.10 * imsize + np.random.rand() * 0.11 * imsize
        x, y = x + np.random.randn() * 0.05 * imsize, y + np.random.rand() * 0.05 * imsize
    elif label["size"] == "small":
        radius = 0.03 * imsize + np.random.rand() * 0.04 * imsize
        x, y = x + np.random.randn() * 0.05 * imsize, y + np.random.rand() * 0.05 * imsize
    else:
        raise ValueError

    rotation = np.random.rand() * 360

    if label["shape"] == "square":
        patch = patches.Rectangle((x - radius, y - radius), 2 * radius, 2 * radius, rotation, facecolor=label["color"])
    elif label["shape"] == "round":
        patch = patches.Circle((x, y), radius, facecolor=label["color"])
    elif label["shape"] == "triangle":
        coordinates = np.array([[radius, -radius],
                                [0, radius],
                                [radius, radius]])
        origin = np.array([x, y])
        rotation_m = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        patch = patches.Polygon(origin + coordinates @ rotation_m, facecolor = label["color"])
    else:
        raise ValueError

    ax.add_patch(patch)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)
    plt.tight_layout(pad=0)
    plt.savefig(location, format="png")
    # plt.show()


def save_dataset(categories, labels, imsize, path_root):
    for k, l in enumerate(labels):
        label = {cat: categories[cat][k] for cat, k in zip(categories.keys(), l)}
        path_file = path_root / f"{k}.png"
        create_image_from_label(label, imsize, str(path_file))


def save_labels(location, categories, labels, header=True):
    with open(location, "w") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(list(categories.keys()))
        writer.writerows(labels)

def main():
    image_size = 32
    dataset_location = Path("/mnt/SSD/datasets/shapes")
    size_train_set = 50_000
    size_val_set = 5_000
    size_test_set = 5_000

    # domain "text"
    categories = OrderedDict([
        ("shape", ["round", "square", "triangle"]),
        ("color", ["black", "blue", "red"]),
        ("size", ["small", "large"]),
        ("location", ["tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br"])
    ])

    train_labels = get_random_population(categories, size_train_set)
    val_labels = get_random_population(categories, size_val_set)
    test_labels = get_random_population(categories, size_test_set)
    labels = []
    for cat, vals in categories.items():
        labels.append([cat] + vals)
    save_labels(dataset_location / "labels.txt", categories, labels, header=False)

    save_labels(dataset_location / "train_labels.txt", categories, train_labels)
    save_labels(dataset_location / "val_labels.txt", categories, val_labels)
    save_labels(dataset_location / "test_labels.txt", categories, test_labels)

    (dataset_location / "train").mkdir()
    save_dataset(categories, train_labels, image_size, dataset_location / "train")
    (dataset_location / "val").mkdir()
    save_dataset(categories, val_labels, image_size, dataset_location / "val")
    (dataset_location / "test").mkdir()
    save_dataset(categories, test_labels, image_size, dataset_location / "test")

    print('done!')


if __name__ == '__main__':
    main()
