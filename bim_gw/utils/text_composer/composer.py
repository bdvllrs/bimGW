import random

import numpy as np
import re

from bim_gw.utils.text_composer.writers import writers


class Composer:
    script_structures = [
        "{start} {size} {color} {shape}, {located} {in_the} {location} {corner}. It is {pointing} the {rotation} {corner}.",
        "{start} {size} {shape} in {color} color, {located} {in_the} {location} {corner}. It is {pointing} the {rotation} {corner}.",
        "{start} {size} {shape} in {color} color. It is {located} {in_the} {location} {corner} and {pointing} the {rotation} {corner}.",
        "{start} {size} {color} {shape}. It is {located} {in_the} {location} {corner} and {pointing} the {rotation} {corner}.",
    ]

    variants = {
        "start": ["A", "It is a", "A kind of", "This is a", "There is a"],
        "located": ["", "located"],
        "in_the": ["in the", "at the"],
        "corner": ["", "corner"],
        "pointing": ["pointing towards", "rotated towards"]
    }

    def __init__(self, writers):
        self.writers = writers

    def __call__(self, attributes):
        selected_structure = random.choice(self.script_structures)
        variants = dict()
        for k, choices in self.variants.items():
            variants[k] = random.choice(choices)
        written_attrs = dict()
        for attr_name, attr in attributes.items():
            writer = random.choice(self.writers[attr_name])
            if isinstance(attr, (list, tuple)):
                written_attrs[attr_name] = writer(*attr)
            else:
                written_attrs[attr_name] = writer(attr)
        final_caption = selected_structure.format(**written_attrs, **variants).strip()
        return re.sub(' +', ' ', final_caption).replace(" .", ".")


if __name__ == '__main__':
    composer = Composer(writers)

    print(composer({
        "shape": 2,
        "rotation": np.pi/6,
        "color": (129, 76, 200),
        "size": 20,
        "location": (29, 8)
    }))
