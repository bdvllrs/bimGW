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
    labels=["north", "north-west", "west", "south-west", "south", "south-east", "east", "north-east"]
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
    "rotation": [corner_rotation_writer, cardinal_rotation_writer, continuous_rotation_writer],
    "size": [size_writer],
    "color": [color_large_set_writer],
    "location": [location_writer]
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
                print(writer(i, j))


def test_colors(writer):
    print(writer(255, 165, 87))


if __name__ == '__main__':
    test_rotation_writer(cardinal_rotation_writer)
    test_rotation_writer(corner_rotation_writer)
    test_rotation_writer(continuous_rotation_writer)
    test_shapes_writer(shapes_writer)
    test_size_writer(size_writer)
    test_location_writer(location_writer)
    test_location_writer(location_writer_bins)
    test_colors(color_large_set_writer)
    test_colors(color_sparse_writer)
