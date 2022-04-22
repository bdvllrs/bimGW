import numpy as np

from attributes_to_language.composer import Composer
from bim_gw.utils.text_composer.writers import writers

# or from a kwargs given to the composer method.
script_structures = [
    "{start} {size} {color} {shape}, {located} {in_the} {location}{link} {rotation}.",
    "{start} {color} {size} {shape}, {located} {in_the} {location}{link} {rotation}.",
    "{start} {size} {shape} in {color} color, {located} {in_the} {location}{link} {rotation}.",
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

composer = Composer(script_structures, variants, writers)

if __name__ == '__main__':
    for k in range(5):
        print(composer({
            "shape": 2,
            "rotation": np.pi/6,
            "color": (129, 76, 200),
            "size": 20,
            "location": (29, 8)
        }))