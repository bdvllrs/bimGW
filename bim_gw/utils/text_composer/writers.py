import numpy as np
from attributes_to_language.utils import (
    COLORS_LARGE_SET,
    COLORS_SPARSE,
    COLORS_XKCD,
)
from attributes_to_language.writers import (
    Bins2dWriter,
    BinsWriter,
    ContinuousAngleWriter,
    OptionsWriter,
    QuantizedWriter,
)

shapes_writer = OptionsWriter(
    caption="{val}",
    choices={
        2: ["isosceles triangle", "triangle"],
        1: [
            "egg",
            "water droplet",
            "isosceles triangle that has round corners",
            "bullet",
            "oval shaped structure",
            "triangle-like shape with rounded vertices",
            "guitar pick",
        ],
        0: [
            "diamond",
            "trapezoidal shape",
            "four-sided shape",
            "kite",
            "quadrilateral",
            "arrow-shaped polygon",
            "deformed square shape",
        ],
    },
)

cardinal_rotation_writer = QuantizedWriter(
    quantized_values=np.array(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            3 * np.pi / 2,
            7 * np.pi / 4,
        ]
    ),
    caption="{rotated} {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to the", "pointing towards the", "pointing"],
    },
    labels=[
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
        "east",
        "northeast",
    ],
)

cardinal_rotation_preicions_writer = QuantizedWriter(
    quantized_values=np.array(
        [
            0,
            np.pi / 8,
            np.pi / 4,
            3 * np.pi / 8,
            np.pi / 2,
            5 * np.pi / 8,
            3 * np.pi / 4,
            7 * np.pi / 8,
            np.pi,
            9 * np.pi / 8,
            5 * np.pi / 4,
            11 * np.pi / 8,
            3 * np.pi / 2,
            13 * np.pi / 8,
            7 * np.pi / 4,
            15 * np.pi / 8,
        ]
    ),
    caption="{rotated} {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to the", "pointing towards the", "pointing"],
    },
    labels=[
        "north",
        "north-northwest",
        "northwest",
        "west-northwest",
        "west",
        "west-southwest",
        "southwest",
        "south-southwest",
        "south",
        "south-southeast",
        "southeast",
        "east-southeast",
        "east",
        "east-northeast",
        "northeast",
        "north-northeast",
    ],
)

corner_rotation_writer = QuantizedWriter(
    quantized_values=np.array(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi,
            5 * np.pi / 4,
            3 * np.pi / 2,
            7 * np.pi / 4,
        ]
    ),
    caption="{rotated} the {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to", "pointing towards"],
        "corner": ["", " corner"],
        "side": ["", " side"],
    },
    labels=[
        "top",
        "top-left{corner}",
        "left{side}",
        "bottom-left{corner}",
        "bottom",
        "bottom-right{corner}",
        "right{side}",
        "top-right{corner}",
    ],
)

corner_rotation_precision_writer = QuantizedWriter(
    quantized_values=np.array(
        [
            0,
            np.pi / 8,
            np.pi / 4,
            3 * np.pi / 8,
            np.pi / 2,
            5 * np.pi / 8,
            3 * np.pi / 4,
            7 * np.pi / 8,
            np.pi,
            9 * np.pi / 8,
            5 * np.pi / 4,
            11 * np.pi / 8,
            3 * np.pi / 2,
            13 * np.pi / 8,
            7 * np.pi / 4,
            15 * np.pi / 8,
        ]
    ),
    caption="{rotated} the {val}",
    variants={
        "of_image": ["", "of the image"],
        "rotated": ["pointing to", "pointing towards"],
        "corner": ["", " corner"],
        "side": ["", " side"],
    },
    labels=[
        "top",
        "top top-left{corner}",
        "top-left{corner}",
        "left top-left{corner}",
        "left{side}",
        "left bottom-left{corner}",
        "bottom-left{corner}",
        "bottom bottom-left{corner}",
        "bottom",
        "bottom bottom-right{corner}",
        "bottom-right{corner}",
        "right bottom-right{corner}",
        "right{side}",
        "right top-right{corner}",
        "top-right{corner}",
        "top top-right{corner}",
    ],
)

continuous_rotation_writer = ContinuousAngleWriter(
    caption="rotated {val} degrees {anti_clock}",
    variants={
        "anti_clock": ["", "anti clockwise"],
    },
)

size_writer = BinsWriter(
    bins=np.array([9, 11, 13]),
    labels=[
        "tiny",
        ["small", "little"],  # one of
        ["average sized", "medium sized", "medium"],
        ["big", "large"],
    ],
)

location_writer = QuantizedWriter(
    quantized_values=np.array(
        [
            [10, 16, 22, 10, 16, 22, 10, 16, 22],
            [10, 10, 10, 16, 16, 16, 22, 22, 22],
        ]
    ),
    caption="{located?}{val}",
    labels=[
        [
            "{at_the} {bottom} left{side?}",
            "{at_the} {bottom}, {on_the} left{side?}",
        ],
        ["{at_the} {bottom} {middle}", "{at_the} {bottom}"],
        ["{at_the} {bottom} right{side?}"],
        ["{in_the} {middle} left{side?}"],
        ["{in_the} {middle}"],
        ["{in_the} {middle} right{side?}"],
        ["{at_the} {top} left{side?}"],
        ["{at_the} {top} {middle}", "{top}"],
        ["{at_the} {top} right{side?}"],
    ],
    variants={
        "bottom": ["bottom", "lower side"],
        "middle": ["middle", "center"],
        "top": ["top", "upper side"],
        "at_the": ["at the"],
        "in_the": ["in the", "at the"],
        "on_the": ["on the", "at the"],
        "side?": [" side", ""],
        "located?": ["located ", " "],
    },
)

location_writer_bins = Bins2dWriter(
    bins=np.array([[13, 19], [13, 19]]),
    labels=[
        [["top left", "upper left"], "top", "top right"],
        ["left", "middle", "right"],
        ["bottom left", "bottom", "bottom right"],
    ],
)

location_precision_writer_bins = Bins2dWriter(
    caption="{located?}{val}",
    bins=np.array([[9, 12, 14, 18, 20, 23], [9, 12, 14, 18, 20, 23]]),
    variants={
        "bottom": ["bottom", "lower side"],
        "middle": ["middle", "center"],
        "top": ["top", "upper side"],
        "at_the": ["at the"],
        "in_the": ["in the", "at the"],
        "on_the": ["on the", "at the"],
        "side?": [" side", ""],
        "located?": ["located ", " "],
    },
    labels=np.array(
        [
            [
                "{at_the} very {bottom}, {on_the} very left{side?}",
                "{at_the} very {bottom}, {on_the} left{side?}",
                "{at_the} very {bottom}, slightly left",
                "{at_the} very {bottom}, {in_the} {middle}",
                "{at_the} very {bottom}, slightly right",
                "{at_the} very {bottom}, {on_the} right{side?}",
                "{at_the} very {bottom}, {on_the} very right{side?}",
            ],
            [
                "{at_the} {bottom}, {on_the} very left{side?}",
                "{at_the} {bottom}, {on_the} left{side?}",
                "{at_the} {bottom}, slightly left",
                "{at_the} {bottom} {middle}",
                "{at_the} {bottom}, slightly right",
                "{at_the} {bottom}, {on_the} right{side?}",
                "{at_the} {bottom}, {on_the} very right{side?}",
            ],
            [
                "slightly {bottom}, {on_the} very left{side?}",
                "slightly {bottom}, {on_the} left{side?}",
                "slightly {bottom}, slightly left",
                "slightly {bottom}, {in_the} {middle}",
                "slightly {bottom}, slightly right",
                "slightly {bottom}, {on_the} right{side?}",
                "slightly {bottom}, {on_the} very right{side?}",
            ],
            [
                "{in_the} {middle}, {on_the} very left {side?}",
                "{in_the} {middle}, {on_the} left{side?}",
                "{in_the} {middle}, slightly left",
                "{in_the} {middle}",
                "{in_the} {middle}, slightly right",
                "{in_the} {middle}, {on_the} right{side?}",
                "{in_the} {middle}, {on_the} very right{side?}",
            ],
            [
                "slightly {top}, {on_the} very left{side?}",
                "slightly {top}, {on_the} left{side?}",
                "slightly {top}, slightly left",
                "slightly {top}, {in_the} {middle}",
                "slightly {top}, slightly right",
                "slightly {top}, {on_the} right{side?}",
                "slightly {top}, {on_the} very right{side?}",
            ],
            [
                "{at_the} {top}, {on_the} very left{side?}",
                "{at_the} {top}, {on_the} left{side?}",
                "{at_the} {top}, slightly left",
                "{at_the} {top} {middle}",
                "{at_the} {top}, slightly right",
                "{at_the} {top}, {on_the} right{side?}",
                "{at_the} {top}, {on_the} very right{side?}",
            ],
            [
                "{at_the} very {top}, {on_the} very left{side?}",
                "{at_the} very {top}, {on_the} left{side?}",
                "{at_the} very {top}, slightly left",
                "{at_the} very {top}, {in_the} {middle}",
                "{at_the} very {top}, slightly right",
                "{at_the} very {top}, {on_the} right{side?}",
                "{at_the} very {top}, {on_the} very right{side?}",
            ],
        ]
    ).transpose(),
)

color_large_set_writer = QuantizedWriter(
    caption="{val}",
    quantized_values=COLORS_LARGE_SET["rgb"],
    labels=COLORS_LARGE_SET["labels"],
)

color_xkcd_writer = QuantizedWriter(
    caption="{val}{color?}",
    variants={"color?": ["", "-color", " colored"]},
    quantized_values=COLORS_XKCD["rgb"],
    labels=COLORS_XKCD["labels"],
)

color_sparse_writer = QuantizedWriter(
    caption="{val}{color?}",
    variants={"color?": ["", " color", " colored"]},
    quantized_values=COLORS_SPARSE["rgb"],
    labels=COLORS_SPARSE["labels"],
)

# writers = {
#     "shape": [shapes_writer],
#     "rotation": [corner_rotation_writer, cardinal_rotation_writer],
#     "size": [size_writer],
#     "color": [color_large_set_writer],
#     "location": [location_writer_bins]
# }

writers = {
    "shape": [shapes_writer],
    "rotation": [
        corner_rotation_writer,
        cardinal_rotation_writer,
        corner_rotation_precision_writer,
        cardinal_rotation_preicions_writer,
    ],
    "size": [size_writer],
    "color": [color_large_set_writer],
    "location": [location_precision_writer_bins, location_writer],
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


if __name__ == "__main__":
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
