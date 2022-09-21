import numpy as np

from attributes_to_language.composer import Composer

from bim_gw.utils.text_composer.modifiers import MixModifier, DeleteModifier
from bim_gw.utils.text_composer.writers import writers

# or from a kwargs given to the composer method.
script_structures = [
    "{start} {size}{link}{color}{link}{shape}{link}{location}{link}{rotation}.",
    "{start} {color}{link}{size}{link}{shape}{link}{location}{link}{rotation}.",
    "{start} {size}{link}{shape}{link}{color}{link}{location}{link}{rotation}.",
    "{start} {size}{link}{shape}{link}{location}{link}{color}{link}{rotation}.",
    "{start} {shape}{link}{location}{link}{size}{link}{color}{link}{rotation}.",
]

# Elements in the list of each variant is randomly chosen.
variants = {
    "start": ["", "It is", "A kind of", "This is", "There is",
              "The image is", "The image represents", "The image contains"],
    "link": [". It is ", ", and is ", " ", ", it's ", ". It's "],
}

modifiers = None
# modifiers = [
#     DeleteModifier(0.2, 3),
#     MixModifier(0.2)
# ]

composer = Composer(script_structures, variants, writers, modifiers)

script_structures = [
    "{start} {size} {color} {shape}, {location}{link2} {rotation}.",
    "{start} {color} {size} {shape}, {location}{link2} {rotation}.",
    "{start} {size} {shape} in {color} color, {location}{link2} {rotation}.",
    "{start} {size} {shape} in {color} color{link} {location}{link2} {rotation}.",
    "{start} {size} {color} {shape}{link} {location}{link2} {rotation}.",
    "{start} {color} {size} {shape}{link} {location}{link2} {rotation}.",
]

# Elements in the list of each variant is randomly chosen.
variants = {
    "start": ["A", "It is a", "A kind of", "This is a", "There is a",
              "The image is a", "The image represents a", "The image contains a"],
    "link": [". It is", ", it is", ", and is"],
    "link2": [", and is", ", and", ". It is", ", it is"]
}

val_composer = Composer(script_structures, variants, writers, modifiers)

if __name__ == '__main__':
    for k in range(5):
        print(composer({
            "shape": 2,
            "rotation": np.pi/6,
            "color": (129, 76, 200),
            "size": 20,
            "location": (29, 8)
        }))