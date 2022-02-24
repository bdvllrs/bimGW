import random

import numpy as np
import re

from bim_gw.utils.text_composer.writers import writers


class Composer:
    # Templates for the overall text. The items in {} can come from the associated key in the "variants" dict attributes
    # or from a kwargs given to the __call__ method.
    script_structures = [
        "{start} {size} {color} {shape}, {located} {in_the} {location}{link} {rotation}.",
        "{start} {color} {size} {shape}, {located} {in_the} {location}{link} {rotation}.",
        "{start} {size} {shape} in {color} color, {located} {in_the} {location}{link} {is?}{rotation}.",
        "{start} {size} {shape} in {color} color{link} {located} {in_the} {location} and {is?}{rotation}.",
        "{start} {size} {color} {shape}{link} {located} {in_the} {location} and {is?}{rotation}.",
        "{start} {color} {size} {shape}{link} {located} {in_the} {location} and {is?}{rotation}.",
    ]

    # Elements in the list of each variant is randomly chosen.
    variants = {
        "start": ["A", "It is a", "A kind of", "This is a", "There is a",
                  "The image is a", "The image represents a", "The image contains a"],
        "located": ["", "located"],
        "in_the": ["in the", "at the"],
        "link": [". It is", ", and is"],
        "is?": ["", "is "]
    }

    def __init__(self, available_writers):
        """
        Args:
            available_writers: Dict of list of writers for the attributes. The key of the dict corresponds to the attribute name
            and the value is a list of type "Writer".
        """
        self.writers = available_writers

    def __call__(self, attributes):
        """
        Compose one sentence from a dict of attributes
        Args:
            attributes: Dictionary where a key is an attribute name and the value is the value of the attribute that
                will be provided to the associated writer.

        Returns: The composed sentence.
        """
        # Select one of the templates
        selected_structure = random.choice(self.script_structures)
        # Decide which variants will be used
        variants = dict()
        for k, choices in self.variants.items():
            variants[k] = random.choice(choices)
        # Select a writer for each attribute and generate the str.
        written_attrs = dict()
        for attr_name, attr in attributes.items():
            writer = random.choice(self.writers[attr_name])
            if isinstance(attr, (list, tuple)):
                written_attrs[attr_name] = writer(*attr)
            else:
                written_attrs[attr_name] = writer(attr)
        # Create final caption
        final_caption = selected_structure.format(**written_attrs, **variants).strip()
        # remove multiple spaces and spaces in front of "."
        return re.sub(' +', ' ', final_caption).replace(" .", ".")


if __name__ == '__main__':
    composer = Composer(writers)

    for k in range(20):
        print(composer({
            "shape": 2,
            "rotation": np.pi/6,
            "color": (129, 76, 200),
            "size": 20,
            "location": (29, 8)
        }))
