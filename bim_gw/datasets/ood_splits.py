from dataclasses import dataclass
from enum import Enum
from math import pi
from random import randint, sample
from random import seed as set_seed
from typing import Any, List, Sequence, Tuple


class BoundaryKind(str, Enum):
    shape = "shape"
    color = "color"
    size = "size"
    rotation = "rotation"
    x = "x"
    y = "y"


class BoundaryBase:
    def choice(self) -> None:
        raise NotImplementedError

    def filter(self, x: Any) -> bool:
        raise NotImplementedError

    def description(self) -> List[str]:
        raise NotImplementedError


class ChoiceBoundary(BoundaryBase):
    def __init__(self, choices: Sequence):
        self.choices = choices

        self.boundary = self.choice()

    def choice(self):
        return sample(self.choices, 1)[0]

    def filter(self, x) -> bool:
        return x == self.boundary

    def description(self) -> List[str]:
        return [str(self.boundary), str(self.boundary)]


class BinsBoundary(BoundaryBase):
    def __init__(self, bins: Sequence):
        self.bins = bins

        self.boundary = self.choice()

    def choice(self):
        k = randint(1, len(self.bins) - 1)
        return k

    def filter(self, x) -> bool:
        return self.bins[self.boundary - 1] <= x < self.bins[self.boundary]

    def description(self) -> List[str]:
        return [
            str(self.bins[self.boundary - 1]),
            str(self.bins[self.boundary]),
        ]


class RotationBoundary(BinsBoundary):
    def description(self) -> List[str]:
        rot_min = str(int(self.bins[self.boundary - 1] * 180 / pi))
        rot_max = str(int(self.bins[self.boundary] * 180 / pi))
        return [rot_min, rot_max]


class MultiBinsBoundary(BoundaryBase):
    def __init__(self, bins: Sequence):
        self.bins = [BinsBoundary(b) for b in bins]

        self.choice()

    def choice(self) -> List[Any]:
        return [b.choice() for b in self.bins]

    def filter(self, x) -> bool:
        return all(b.filter(k) for k, b in zip(x, self.bins))

    def description(self) -> List[str]:
        return [
            "/".join(b.description()[0] for b in self.bins),
            "/".join(b.description()[1] for b in self.bins),
        ]


@dataclass
class BoundaryInfo:
    kind: BoundaryKind
    boundary: BoundaryBase


def attr_boundaries(
    imsize: int, min_size: int, max_size: int
) -> List[BoundaryInfo]:
    size_range = (max_size - min_size) / 3
    margin = max_size // 2
    x_range = (imsize - 2 * margin) / 3

    boundaries: List[BoundaryInfo] = [
        BoundaryInfo(
            BoundaryKind.shape,
            ChoiceBoundary([0, 1, 2]),
        ),
        BoundaryInfo(
            BoundaryKind.color,
            BinsBoundary([0, 60, 120, 180]),
        ),
        BoundaryInfo(
            BoundaryKind.size,
            BinsBoundary(
                [
                    min_size,
                    min_size + size_range,
                    min_size + 2 * size_range,
                    max_size,
                ]
            ),
        ),
        BoundaryInfo(
            BoundaryKind.rotation,
            RotationBoundary(
                [
                    0,
                    2 * pi / 3,
                    4 * pi / 3,
                    2 * pi,
                ]
            ),
        ),
        BoundaryInfo(
            BoundaryKind.x,
            BinsBoundary(
                [
                    margin,
                    margin + x_range,
                    margin + 2 * x_range,
                    imsize - margin,
                ],
            ),
        ),
        BoundaryInfo(
            BoundaryKind.y,
            BinsBoundary(
                [
                    margin,
                    margin + x_range,
                    margin + 2 * x_range,
                    imsize - margin,
                ],
            ),
        ),
    ]
    return boundaries


def ood_split(
    imsize: int, min_size: int, max_size: int, n_boundaries: int, seed: int
) -> List[BoundaryInfo]:
    set_seed(seed)

    boundaries = attr_boundaries(imsize, min_size, max_size)
    return sample(boundaries, n_boundaries)


def filter_dataset(
    dataset, boundary_infos: List[BoundaryInfo]
) -> Tuple[List[int], List[int]]:
    in_dist_idx: List[int] = []
    ood_idx: List[int] = []
    labels = dataset.labels
    for k in range(len(labels)):
        cls = int(labels[k][0])
        x, y = labels[k][1], labels[k][2]
        size = labels[k][3]
        rotation = labels[k][4]
        hue = labels[k][8]
        for boundary_info in boundary_infos:
            if boundary_info.kind == BoundaryKind.shape:
                value = cls
            elif boundary_info.kind == BoundaryKind.color:
                value = hue
            elif boundary_info.kind == BoundaryKind.size:
                value = size
            elif boundary_info.kind == BoundaryKind.rotation:
                value = rotation
            elif boundary_info.kind == BoundaryKind.x:
                value = x
            elif boundary_info.kind == BoundaryKind.y:
                value = y
            else:
                raise ValueError(f"Unknown boundary kind {boundary_info.kind}")

            if not boundary_info.boundary.filter(value):
                in_dist_idx.append(k)
                break
        else:
            ood_idx.append(k)
    return in_dist_idx, ood_idx


def create_ood_split(datasets, boundary_infos):
    out_datasets = []
    for dataset in datasets:
        in_dist, out_dist = filter_dataset(dataset, boundary_infos)
        out_datasets.append((in_dist, out_dist))
    return out_datasets


def get_generation_boundary(boundary_infos):
    generation_vals = {
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
        "min_rotation": 0,
        "max_rotation": 360,
        "min_hue": 0,
        "max_hue": 180,
        "possible_categories": None,
        "min_scale": 7,
        "max_scale": 14,
    }
    for boundary_info in boundary_infos:
        if boundary_info.kind == BoundaryKind.x:
            generation_vals["min_x"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary - 1
            ]
            generation_vals["max_x"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary
            ]
        elif boundary_info.kind == BoundaryKind.y:
            generation_vals["min_y"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary - 1
            ]
            generation_vals["max_y"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary
            ]
        elif boundary_info.kind == BoundaryKind.shape:
            generation_vals["possible_categories"] = [
                boundary_info.boundary.boundary
            ]
        elif boundary_info.kind == BoundaryKind.size:
            generation_vals["min_scale"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary - 1
            ]
            generation_vals["max_scale"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary
            ]
        elif boundary_info.kind == BoundaryKind.rotation:
            generation_vals["min_rotation"] = (
                boundary_info.boundary.bins[
                    boundary_info.boundary.boundary - 1
                ]
                * 180
                / pi
            )
            generation_vals["max_rotation"] = (
                boundary_info.boundary.bins[boundary_info.boundary.boundary]
                * 180
                / pi
            )
        elif boundary_info.kind == BoundaryKind.color:
            generation_vals["min_hue"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary - 1
            ]
            generation_vals["max_hue"] = boundary_info.boundary.bins[
                boundary_info.boundary.boundary
            ]
    return generation_vals
