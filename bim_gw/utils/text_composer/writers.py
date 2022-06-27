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
    labels=[
        [
            ["very top right, near the sides", "very upper right, near the sides"],
            ["very top right", "very upper right"],
            ["very top center, slightly left", "very top, slightly left"],
            ["very top center", "very top"],
            ["very top center, slightly left", "very top, slightly left"],
            ["very top left", "very upper left"],
            ["very top left, near the sides", "very upper left, near the sides"],
        ],
        [
            ["top right, near the right side", "upper right, near the right side"],
            ["top right", "upper right"],
            ["top center, slightly left", "top, slightly left"],
            ["top center", "top"],
            ["top center, slightly left", "top, slightly left"],
            ["top left", "upper left"],
            ["top left, near the left side", "upper left, near the left side"],
        ],
        [
            ["slightly top right, near the right side", "slightly upper right, near the right side"],
            ["slightly top right", "slightly upper right"],
            ["slightly top center, slightly left", "slightly top, slightly left"],
            ["slightly top center", "slightly top"],
            ["slightly top center, slightly left", "slightly top, slightly left"],
            ["slightly top left", "slightly upper left"],
            ["slightly top left, near the left side", "slightly upper left, near the left side"],
        ],
        [
            ["middle right, near the right side", "center right, near the right side"],
            ["middle right", "center right"],
            ["center, slightly right", "middle, slightly right"],
            ["center", "middle"],
            ["center, slightly left", "middle, slightly left"],
            ["middle left", "center left"],
            ["middle left, near the left side", "center left, near the left side"],
        ],
        [
            ["slightly bottom right, near the right side", "slightly lower right, near the right side"],
            ["slightly bottom right", "slightly lower right"],
            ["slightly bottom center, slightly right", "slightly bottom, slightly right"],
            ["slightly bottom center", "slightly bottom"],
            ["slightly bottom center, slightly left", "slightly bottom, slightly left"],
            ["slightly bottom left", "slightly lower left"],
            ["slightly bottom left, near the left side", "slightly lower left, near the left side"],
        ],
        [
            ["bottom right, near the right side", "lower right, near the right side"],
            ["bottom right", "lower right"],
            ["bottom center, slightly right", "bottom, slightly right"],
            ["bottom center", "bottom"],
            ["bottom center, slightly left", "bottom, slightly left"],
            ["bottom left", "lower left"],
            ["bottom left, near the left side", "lower left, near the left side"],
        ],
        [
            ["very bottom right, near the sides", "very lower right, near the sides"],
            ["very bottom right", "very lower right"],
            ["very bottom, slightly right"],
            ["very bottom, at the center", "very bottom"],
            ["very bottom, slightly left"],
            ["very bottom left", "very lower left"],
            ["very bottom left, near the sides", "very lower left, near the sides"],
        ],
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
    "rotation": [corner_rotation_writer, cardinal_rotation_writer],
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
                print(writer(i, j))


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
