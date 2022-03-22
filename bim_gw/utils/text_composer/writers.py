import random

import numpy as np


def get_closest_key(values, comp, norm="2"):
    fn = np.square
    if norm == "1":
        fn = np.abs
    if len(comp) > 1:
        d = np.sum(fn(values - np.array(comp)[:, None]), axis=0)
    else:
        comp = comp[0]
        d = fn(values - comp)
    return np.argmin(d)


# List of colors obtained from matplotlib colors
colors = {"alice blue": [240, 248, 255],
          "antique white": [250, 235, 215],
          "aqua": [0, 255, 255],
          "aqua marine": [127, 255, 212],
          "azure": [240, 255, 255],
          "beige": [245, 245, 220],
          "bisque": [255, 228, 196],
          "black": [0, 0, 0],
          "blanched almond": [255, 235, 205],
          "blue": [0, 0, 255],
          "blue violet": [138, 43, 226],
          "brown": [165, 42, 42],
          "burly wood": [222, 184, 135],
          "cadet blue": [95, 158, 160],
          "chartreuse": [127, 255, 0],
          "chocolate": [210, 105, 30],
          "coral": [255, 127, 80],
          "cornflower blue": [100, 149, 237],
          "cornsilk": [255, 248, 220],
          "crimson": [220, 20, 60],
          "cyan": [0, 255, 255],
          "dark blue": [0, 0, 139],
          "dark cyan": [0, 139, 139],
          "dark goldenrod": [184, 134, 11],
          "dark grey": [169, 169, 169],
          "dark green": [0, 100, 0],
          "dark khaki": [189, 183, 107],
          "dark magenta": [139, 0, 139],
          "dark olive green": [85, 107, 47],
          "dark orange": [255, 140, 0],
          "dark orchid": [153, 50, 204],
          "dark red": [139, 0, 0],
          "dark salmon": [233, 150, 122],
          "dark sea green": [143, 188, 143],
          "dark slate blue": [72, 61, 139],
          "dark slate grey": [47, 79, 79],
          "dark turquoise": [0, 206, 209],
          "dark violet": [148, 0, 211],
          "deep pink": [255, 20, 147],
          "deep sky blue": [0, 191, 255],
          "dim grey": [105, 105, 105],
          "dodger blue": [30, 144, 255],
          "fire brick": [178, 34, 34],
          "floral white": [255, 250, 240],
          "forest green": [34, 139, 34],
          "fuchsia": [255, 0, 255],
          "gainsboro": [220, 220, 220],
          "ghost white": [248, 248, 255],
          "gold": [255, 215, 0],
          "goldenrod": [218, 165, 32],
          "grey": [128, 128, 128],
          "green": [0, 128, 0],
          "green yellow": [173, 255, 47],
          "honeydew": [240, 255, 240],
          "hot pink": [255, 105, 180],
          "indian red": [205, 92, 92],
          "indigo": [75, 0, 130],
          "ivory": [255, 255, 240],
          "khaki": [240, 230, 140],
          "lavender": [230, 230, 250],
          "lavender blush": [255, 240, 245],
          "lawn green": [124, 252, 0],
          "lemon chiffon": [255, 250, 205],
          "light blue": [173, 216, 230],
          "light coral": [240, 128, 128],
          "light cyan": [224, 255, 255],
          "light goldenrod yellow": [250, 250, 210],
          "light grey": [211, 211, 211],
          "light green": [144, 238, 144],
          "light pink": [255, 182, 193],
          "light salmon": [255, 160, 122],
          "light sea green": [32, 178, 170],
          "light sky blue": [135, 206, 250],
          "light slate grey": [119, 136, 153],
          "light steel blue": [176, 196, 222],
          "light yellow": [255, 255, 224],
          "lime": [0, 255, 0],
          "lime green": [50, 205, 50],
          "linen": [250, 240, 230],
          "magenta": [255, 0, 255],
          "maroon": [128, 0, 0],
          "medium aquamarine": [102, 205, 170],
          "medium blue": [0, 0, 205],
          "medium orchid": [186, 85, 211],
          "medium purple": [147, 112, 219],
          "medium seagreen": [60, 179, 113],
          "medium slate blue": [123, 104, 238],
          "medium spring green": [0, 250, 154],
          "medium turquoise": [72, 209, 204],
          "medium violet red": [199, 21, 133],
          "midnight blue": [25, 25, 112],
          "mint cream": [245, 255, 250],
          "misty rose": [255, 228, 225],
          "moccasin": [255, 228, 181],
          "navajo white": [255, 222, 173],
          "navy": [0, 0, 128],
          "old lace": [253, 245, 230],
          "olive": [128, 128, 0],
          "olive drab": [107, 142, 35],
          "orange": [255, 165, 0],
          "orange red": [255, 69, 0],
          "orchid": [218, 112, 214],
          "pale goldenrod": [238, 232, 170],
          "pale green": [152, 251, 152],
          "pale turquoise": [175, 238, 238],
          "pale violet red": [219, 112, 147],
          "papaya whip": [255, 239, 213],
          "peach puff": [255, 218, 185],
          "peru": [205, 133, 63],
          "pink": [255, 192, 203],
          "plum": [221, 160, 221],
          "powder blue": [176, 224, 230],
          "purple": [128, 0, 128],
          "rebecca purple": [102, 51, 153],
          "red": [255, 0, 0],
          "rosy brown": [188, 143, 143],
          "royal blue": [65, 105, 225],
          "saddle brown": [139, 69, 19],
          "salmon": [250, 128, 114],
          "sandy brown": [244, 164, 96],
          "sea green": [46, 139, 87],
          "sea shell": [255, 245, 238],
          "sienna": [160, 82, 45],
          "silver": [192, 192, 192],
          "skyblue": [135, 206, 235],
          "slate blue": [106, 90, 205],
          "slate grey": [112, 128, 144],
          "snow": [255, 250, 250],
          "spring green": [0, 255, 127],
          "steel blue": [70, 130, 180],
          "tan": [210, 180, 140],
          "teal": [0, 128, 128],
          "thistle": [216, 191, 216],
          "tomato": [255, 99, 71],
          "turquoise": [64, 224, 208],
          "violet": [238, 130, 238],
          "wheat": [245, 222, 179],
          "white": [255, 255, 255],
          "white smoke": [245, 245, 245],
          "yellow": [255, 255, 0],
          "yellow green": [154, 205, 50]}

# A smaller set of colors
colors_sparse = ["azure", "beige", "black", "blue", "brown", "crimson", "cyan", "dark blue", "dark cyan", "dark grey",
                 "dark green", "dark orange", "dark red", "dark violet", "grey", "green", "indigo",
                 "light blue", "light grey", "light green", "light pink", "light yellow", "lime", "magenta",
                 "medium blue",
                 "orange", "pale green", "pink", "purple", "red", "turquoise", "violet", "white", "yellow"]


class Writer:
    """
    Writer instances generate the text associated to a specific value of an attribute.
    They can have different ways of describing the same element. This is defined by the "label_type" property.
    """
    captions = dict()
    variants = dict()

    def __init__(self, label_type):
        assert label_type in self.captions, "Provided label_type does not exist."
        self.label_type = label_type

    def __call__(self, val):
        variants = dict()
        if self.label_type in self.variants:
            for k, choices in self.variants[self.label_type].items():
                variants[k] = random.choice(choices)

        val = str(val).format(**variants)
        text = self.captions[self.label_type].format(val=val, **variants)
        return text


class OptionsWriter:
    choices = dict()

    def __call__(self, val):
        assert val in self.choices, f"{self.choices} does not contain options for {val}"
        selected_option = random.choice(self.choices[val])
        return selected_option


class QuantizedWriter(Writer):
    quantized_values = np.empty(1)
    labels = dict()
    norm = "2"

    def __call__(self, *val):
        quantized_val = get_closest_key(self.quantized_values, val, self.norm)
        text = self.labels[self.label_type][quantized_val]
        if type(text) is list:
            text = random.choice(text)
        return super(QuantizedWriter, self).__call__(text)


class BinsWriter(Writer):
    bins = np.empty(1)
    labels = dict()

    def __call__(self, val):
        index = np.digitize([val], self.bins)[0]
        text = self.labels[self.label_type][index]
        if type(text) is list:
            text = random.choice(text)
        return super(BinsWriter, self).__call__(text)


class Bins2dWriter(Writer):
    bins = np.empty(1)
    labels = dict()

    def __call__(self, *val):
        label = self.labels[self.label_type]
        for k in range(self.bins.shape[0]):
            bins = self.bins[k]
            index = np.digitize([val[k]], bins)[0]
            label = label[index]
        text = label
        if type(text) is list:
            text = random.choice(text)
        return super(Bins2dWriter, self).__call__(text)


class CardinalRotationWriter(QuantizedWriter):
    quantized_values = np.array(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4])

    labels = {
        "corners":
            ["top", "top-left{corner}", "left{side}", "bottom-left{corner}", "bottom", "bottom-right{corner}",
             "right{side}", "top-right{corner}"],
        "cardinals":
            ["north", "north-west", "west", "south-west", "south", "south-east", "east", "north-east"]
    }

    captions = {
        "corners": "{rotated} the {val}",
        "cardinals": "{rotated} {val}"
    }

    variants = {
        "corners": {
            "of_image": ["", "of the image"],
            "rotated": ["pointing to", "pointing towards"],
            "corner": ["", " corner"],
            "side": ["", " side"]
        },
        "cardinals": {
            "of_image": ["", "of the image"],
            "rotated": ["pointing to the", "pointing towards the", "pointing"]
        }
    }


class ContinuousRotationWriter(Writer):
    captions = {
        "degrees": "rotated {val} degrees {anti_clock}",
    }

    variants = {
        "degrees": {
            "anti_clock": ["", "anti clockwise"],
        }
    }

    def __call__(self, rotation):
        if rotation < 0:
            rotation = 2 * np.pi + rotation
        # In the code the 0 rotation is at the top. In text, it is more natural to put 0 degrees when facing right.
        rotation += np.pi / 2
        # round to every 5 degrees
        deg = int(5 * round(rotation * 360 / (2 * np.pi) / 5)) % 360
        return super(ContinuousRotationWriter, self).__call__(deg)


class ShapesWriter(OptionsWriter):
    choices = {
        2: ["isosceles triangle", "triangle"],
        1: ["egg", "water droplet", "isosceles triangle that has round corners", "bullet",
            "oval shaped structure", "triangle-like shape with rounded vertices", "guitar pick"],
        0: ["diamond", "trapezoidal shape", "four-sided shape", "kite", "quadrilateral", "arrow-shaped polygon",
            "deformed square shape"],
    }


class SizeWriter(QuantizedWriter):
    quantized_values = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25])

    labels = {
        0: [
            "tiny",
            ["very small", "quite small", "very little"],
            ["small", "little"],
            ["average sized", "medium sized", "medium"],
            ["big", "large"],
            ["very big", "very large"],
            "huge"
        ],
    }

    captions = {
        0: "{val}",
    }


class LocationWriter(QuantizedWriter):
    quantized_values = np.array([[10, 16, 22, 10, 16, 22, 10, 16, 22],
                                 [10, 10, 10, 16, 16, 16, 22, 22, 22]])

    labels = {
        0: [
            ["top left", "upper left"],
            ["top center", "top"],
            ["top right", "upper right"],
            ["middle left", "center left"],
            ["center", "middle"],
            ["middle right", "center right"],
            ["bottom left", "lower left"],
            ["bottom center", "bottom"],
            ["bottom right", "lower right"],
        ],
    }

    captions = {
        0: "{val}",
    }

    variants = {
        0: {
            "located": ["", "located"],
            "prefix": ["in the", "at the"],
            "postfix": [" corner", ""],
            "of_image": ["", " of the image"]
        }
    }


class LocationQuantizer(Bins2dWriter):
    bins = np.array([[13, 19],
                     [13, 19]])

    labels = {
        0: [
            ["topleft", "top", "topright"],
            ["left", "mid", "right"],
            ["botleft", "bot", "botright"]
        ],
    }

    captions = {
        0: "{val}",
    }

    variants = {
        0: {}
    }


class RotationQuantizer(BinsWriter):
    bins = np.array(
        [np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, 7 * np.pi / 8, 9 * np.pi / 8, 11 * np.pi / 8, 13 * np.pi / 8, 15 * np.pi / 8])

    labels = {
        0: [
            "right", "topright", "top", "topleft", "left", "botleft", "bot", "botright", "right"
        ],
    }

    captions = {
        0: "{val}",
    }

    def __call__(self, rotation_x, rotation_y):
        rotation_x = rotation_x * 2 - 1
        rotation_y = rotation_y * 2 - 1
        rotation = np.arctan2(rotation_y, rotation_x)
        if rotation < 0:
            rotation = 2 * np.pi + rotation
        # In the code the 0 rotation is at the top. In text, it is more natural to put 0 degrees when facing right.
        rotation += np.pi / 2
        return super(RotationQuantizer, self).__call__(rotation % (2 * np.pi))


class ColorWriter(QuantizedWriter):
    def __init__(self, label_type):
        labels, self.quantized_values = zip(*colors.items())
        self.quantized_values = np.stack(list(self.quantized_values), axis=1)
        self.labels = {
            0: list(labels)
        }
        self.captions = {
            0: "{val}"
        }

        super().__init__(label_type)


class SimpleColorWriter(QuantizedWriter):
    def __init__(self, label_type):
        labels = colors_sparse
        self.quantized_values = [colors[label] for label in labels]
        self.quantized_values = np.stack(list(self.quantized_values), axis=1)
        self.labels = {
            0: list(labels)
        }
        self.captions = {
            0: "{val}"
        }

        super().__init__(label_type)


writers = {
    "shape": [ShapesWriter()],
    "rotation": [CardinalRotationWriter("corners"), CardinalRotationWriter("cardinals"),
                 ContinuousRotationWriter("degrees")],
    "size": [SizeWriter(0)],
    "color": [ColorWriter(0)],
    "location": [LocationWriter(0)]
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
    test_rotation_writer(CardinalRotationWriter("corners"))
    test_rotation_writer(CardinalRotationWriter("cardinals"))
    test_rotation_writer(ContinuousRotationWriter("degrees"))
    test_shapes_writer(ShapesWriter())
    test_size_writer(SizeWriter(0))
    test_location_writer(LocationWriter(0))
    test_colors(ColorWriter(0))
