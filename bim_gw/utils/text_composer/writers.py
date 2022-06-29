import numpy as np

from attributes_to_language.utils import COLORS_LARGE_SET, COLORS_SPARSE
from attributes_to_language.writers import QuantizedWriter, Writer, OptionsWriter, BinsWriter, Bins2dWriter, \
    ContinuousAngleWriter

shapes_writer = OptionsWriter(
    choices={
        2: ["isosceles triangle", "triangle"],
        1: ["egg", "water droplet", "isosceles triangle that has round corners", "bullet",
            "oval shaped structure", "triangle-like shape with rounded vertices", "guitar pick"],
        0: ["diamond", "trapezoidal shape", "four-sided shape", "kite", "quadrilateral", "arrow-shaped polygon",
            "deformed square shape"],
    }
)

cardinal_rotation_writer = QuantizedWriter(
    quantized_values=np.array(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]),
    caption="{rotated} {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to the", "pointing towards the", "pointing"]
    },
    labels=["north", "northwest", "west", "southwest", "south", "southeast", "east", "northeast"]
)

cardinal_rotation_preicions_writer = QuantizedWriter(
    quantized_values=np.array(
        [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 5 * np.pi / 8, 3 * np.pi / 4, 7 * np.pi / 8,
         np.pi, 9 * np.pi / 8, 5 * np.pi / 4, 11 * np.pi / 8, 3 * np.pi / 2, 13 * np.pi / 8, 7 * np.pi / 4,
         15 * np.pi / 8]),
    caption="{rotated} {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to the", "pointing towards the", "pointing"]
    },
    labels=["north", "north-northwest", "northwest", "west-northwest", "west", "west-southwest",
            "southwest", "south-southwest", "south", "south-southeast", "southeast", "east-southeast", "east",
            "east-northeast", "northeast", "north-northeast"]
)

corner_rotation_writer = QuantizedWriter(
    quantized_values=np.array(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]),
    caption="{rotated} the {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to", "pointing towards"],
        "corner": ["", " corner"],
        "side": ["", " side"]
    },
    labels=["top", "top-left{corner}", "left{side}", "bottom-left{corner}", "bottom", "bottom-right{corner}",
            "right{side}", "top-right{corner}"],
)

corner_rotation_precision_writer = QuantizedWriter(
    quantized_values=np.array(
        [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 5 * np.pi / 8, 3 * np.pi / 4, 7 * np.pi / 8,
         np.pi, 9 * np.pi / 8, 5 * np.pi / 4, 11 * np.pi / 8, 3 * np.pi / 2, 13 * np.pi / 8, 7 * np.pi / 4,
         15 * np.pi / 8]),
    caption="{rotated} the {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to", "pointing towards"],
        "corner": ["", " corner"],
        "side": ["", " side"]
    },
    labels=["top", "top top-left{corner}", "top-left{corner}", "left top-left{corner}",
            "left{side}", "left bottom-left{corner}", "bottom-left{corner}", "bottom bottom-left{corner}",
            "bottom", "bottom bottom-right{corner}", "bottom-right{corner}", "right bottom-right{corner}",
            "right{side}", "right top-right{corner}", "top-right{corner}", "top top-right{corner}"],
)

continuous_rotation_writer = ContinuousAngleWriter(
    caption="rotated {val} degrees {anti_clock}",
    variants={
        "anti_clock": ["", "anti clockwise"],
    }
)

size_writer = BinsWriter(
    bins=np.array([9, 11, 13]),
    labels=[
        "tiny",
        ["small", "little"],  # one of
        ["average sized", "medium sized", "medium"],
        ["big", "large"],
    ]
)

location_writer = QuantizedWriter(
    quantized_values=np.array([[10, 16, 22, 10, 16, 22, 10, 16, 22],
                               [10, 10, 10, 16, 16, 16, 22, 22, 22]]),
    labels=[
        ["bottom left", "lower left"],
        ["bottom center", "bottom"],
        ["bottom right", "lower right"],
        ["middle left", "center left"],
        ["center", "middle"],
        ["middle right", "center right"],
        ["top left", "upper left"],
        ["top center", "top"],
        ["top right", "upper right"],
    ],
    variants={
        "located": ["", "located"],
        "prefix": ["in the", "at the"],
        "postfix": [" corner", ""],
        "of_image": ["", " of the image"]
    }
)

location_writer_bins = Bins2dWriter(
    bins=np.array([[13, 19],
                   [13, 19]]),
    labels=[
        [["top left", "upper left"], "top", "top right"],
        ["left", "middle", "right"],
        ["bottom left", "bottom", "bottom right"]
    ],
)

location_precision_writer_bins = Bins2dWriter(
    bins=np.array([[9, 12, 14, 18, 20, 23],
                   [9, 12, 14, 18, 20, 23]]),
    variants={
        "bottom": ["bottom", "lower side"],
        "middle": ["middle", "center"],
        "top": ["top", "upper side"],
        "on": ["on", "at"],
        "side?": [" side", ""]
    },
    labels=np.array([
        [
            "very {bottom}, {on} the very left{side?}",
            "very {bottom}, {on} the left{side?}",
            "very {bottom}, slightly left",
            "very {bottom}, at the {middle}",
            "very {bottom}, slightly right",
            "very {bottom}, {on} the right{side?}",
            "very {bottom}, {on} the very right{side?}",
        ],
        [
            "{bottom}, {on} the very left{side?}",
            "{bottom}, {on} the left{side?}",
            "{bottom}, slightly left",
            "{bottom} {middle}",
            "{bottom}, slightly right",
            "{bottom}, {on} the right{side?}",
            "{bottom} right, {on} the very right{side?}",
        ],
        [
            "slightly {bottom}, {on} the very left{side?}",
            "slightly {bottom}, {on} the left{side?}",
            "slightly {bottom}, slightly left",
            "slightly {bottom}, at the {middle}",
            "slightly {bottom}, slightly right",
            "slightly {bottom}, {on} the right{side?}",
            "slightly {bottom}, {on} the very right{side?}",
        ],
        [
            "{middle}, {on} the very left {side?}",
            "{middle}, {on} the left{side?}",
            "{middle}, slightly left",
            "{middle}",
            "{middle}, slightly right",
            "{middle}, {on} the right{side?}",
            "{middle} right, {on} the very right{side?}",
        ],
        [
            "slightly {top} left, {on} the very left{side?}",
            "slightly {top}, {on} the left{side?}",
            "slightly {top}, slightly left",
            "slightly {top}, at the {middle}",
            "slightly {top}, slightly right",
            "slightly {top}, {on} the right{side?}",
            "slightly {top}, {on} the very right{side?}",
        ],
        [
            "{top} left, {on} the very left{side?}",
            "{top}, {on} the left{side?}",
            "{top}, slightly left",
            "{top} {middle}",
            "{top}, slightly right",
            "{top}, {on} the right{side?}",
            "{top} right, {on} the very right{side?}",
        ],
        [
            "very {top} left, {on} the very left{side?}",
            "very {top}, {on} the left{side?}",
            "very {top}, slightly left",
            "very {top}, at the {middle}",
            "very {top}, slightly right",
            "very {top}, {on} the right{side?}",
            "very {top}, {on} the very right{side?}",
        ],
    ]).transpose()
)

color_large_set_writer = QuantizedWriter(
    quantized_values=COLORS_LARGE_SET["rgb"],
    labels=COLORS_LARGE_SET["labels"]
)

color_sparse_writer = QuantizedWriter(
    quantized_values=COLORS_SPARSE["rgb"],
    labels=COLORS_SPARSE["labels"]
)

writers = {
    "shape": [shapes_writer],
    "rotation": [corner_rotation_precision_writer, cardinal_rotation_preicions_writer],
    "size": [size_writer],
    "color": [color_large_set_writer],
    "location": [location_precision_writer_bins]
}


def test_rotation_writer(writer):
    print(writer(np.pi / 6))
    print(writer(np.pi / 3))
    print(writer(2 * np.pi / 3))


def test_shapes_writer(writer):
    for k in range(5):
        print(writer(2))
        print(writer(1))
        print(writer(0))


def test_size_writer(writer):
    for k in range(5):
        for i in range(5, 32, 3):
            print(writer(i))


def test_location_writer(writer):
    for k in range(5):
        for i in range(0, 32, 6):
            for j in range(0, 32, 6):
                print(i, j, writer(i, j))


def test_colors(writer):
    print(writer(255, 165, 87))


if __name__ == '__main__':
    test_location_writer(location_precision_writer_bins)
    # test_rotation_writer(cardinal_rotation_writer)
    # test_rotation_writer(corner_rotation_writer)
    # test_rotation_writer(continuous_rotation_writer)
    # test_shapes_writer(shapes_writer)
    # test_size_writer(size_writer)
    # test_location_writer(location_writer)
    # test_location_writer(location_writer_bins)
    # test_colors(color_large_set_writer)
    # test_colors(color_sparse_writer)
