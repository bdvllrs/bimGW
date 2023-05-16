import random
from typing import Mapping, TypeVar

import numpy as np

from bim_gw.utils.types import DistLiteral, SubsetableDataset


def in_interval(
    x: float, xmin: float, xmax: float, val_min: float, val_max: float
) -> bool:
    if val_min <= xmin <= xmax <= val_max:
        return xmin <= x <= xmax
    if xmax < xmin:
        return xmin <= x <= val_max or val_min <= x <= xmax
    return False


def split_in_out_dist(dataset, ood_attrs):
    in_dist_items = []
    out_dist_items = []

    for i, label in zip(dataset.ids, dataset.y_axis_labels):
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        hue = label[8]
        keep = True
        for k in range(3):
            n_cond_checked = 0
            if (
                "shape" in ood_attrs["selected_attributes"][k]
                and ood_attrs["shape"][k] == cls
            ):
                n_cond_checked += 1
            if "position" in ood_attrs["selected_attributes"][
                k
            ] and in_interval(
                x, ood_attrs["x"][k], ood_attrs["x"][(k + 1) % 3], 0, 32
            ):
                n_cond_checked += 1
            if "position" in ood_attrs["selected_attributes"][
                k
            ] and in_interval(
                y, ood_attrs["y"][k], ood_attrs["y"][(k + 1) % 3], 0, 32
            ):
                n_cond_checked += 1
            if "color" in ood_attrs["selected_attributes"][k] and in_interval(
                hue,
                ood_attrs["color"][k],
                ood_attrs["color"][(k + 1) % 3],
                0,
                256,
            ):
                n_cond_checked += 1
            if "size" in ood_attrs["selected_attributes"][k] and in_interval(
                size,
                ood_attrs["size"][k],
                ood_attrs["size"][(k + 1) % 3],
                0,
                25,
            ):
                n_cond_checked += 1
            if "rotation" in ood_attrs["selected_attributes"][
                k
            ] and in_interval(
                rotation,
                ood_attrs["rotation"][k],
                ood_attrs["rotation"][(k + 1) % 3],
                0,
                2 * np.pi,
            ):
                n_cond_checked += 1
            if n_cond_checked >= len(ood_attrs["selected_attributes"][k]):
                keep = False
                break
        if keep:
            in_dist_items.append(i)
        else:
            out_dist_items.append(i)
    return in_dist_items, out_dist_items


def create_ood_split(datasets):
    """
    Splits the space of each attribute in 3 and hold one ood part
    Args:
        datasets:
    Returns:
    """
    shape_boundary = random.randint(0, 2)
    shape_boundaries = (
        shape_boundary,
        (shape_boundary + 1) % 3,
        (shape_boundary + 2) % 3,
    )

    color_boundary = random.randint(0, 255)
    color_boundaries = (
        color_boundary,
        (color_boundary + 85) % 256,
        (color_boundary + 170) % 256,
    )

    size_boundary = random.randint(10, 25)
    size_boundaries = (
        size_boundary,
        10 + (size_boundary - 5) % 16,
        10 + size_boundary % 16,
    )

    rotation_boundary = random.random() * 2 * np.pi
    rotation_boundaries = (
        rotation_boundary,
        (rotation_boundary + 2 * np.pi / 3) % (2 * np.pi),
        (rotation_boundary + 4 * np.pi / 3) % (2 * np.pi),
    )

    x_boundary = random.random() * 32
    x_boundaries = (
        x_boundary,
        (x_boundary + 11.6) % 32,
        (x_boundary + 23.2) % 32,
    )

    y_boundary = random.random() * 32
    y_boundaries = (
        y_boundary,
        (y_boundary + 11.6) % 32,
        (y_boundary + 23.2) % 32,
    )

    print("boundaries")
    print("shape", shape_boundaries)
    print("color", color_boundaries)
    print("size", size_boundaries)
    print("rotation", rotation_boundaries)
    print("x", x_boundaries)
    print("y", y_boundaries)

    selected_attributes = []
    for i in range(3):
        holes_k = []
        choices = ["position", "rotation", "size", "color", "shape"]
        for k in range(2):
            choice = random.randint(0, len(choices) - 1)
            holes_k.append(choices[choice])
            choices.pop(choice)
        selected_attributes.append(holes_k)
    print("OOD: ", selected_attributes)

    ood_boundaries = {
        "x": x_boundaries,
        "y": y_boundaries,
        "size": size_boundaries,
        "rotation": rotation_boundaries,
        "color": color_boundaries,
        "selected_attributes": selected_attributes,
        "shape": shape_boundaries,
    }

    out_datasets = []
    for dataset in datasets:
        in_dist, out_dist = split_in_out_dist(dataset, ood_boundaries)
        out_datasets.append((in_dist, out_dist))
    return out_datasets, ood_boundaries


X = TypeVar("X", bound=SubsetableDataset)


def split_ood_sets(dataset: X, id_ood_split=None) -> Mapping[DistLiteral, X]:
    return {
        "in_dist": dataset.subset(id_ood_split[1][0])
        if id_ood_split is not None
        else dataset,
        "ood": dataset.subset(id_ood_split[1][1])
        if id_ood_split is not None
        else None,
    }
