import os
from pathlib import Path

import numpy as np
from attributes_to_language.composer import Composer
from tqdm import tqdm

from bim_gw.utils import get_args
from bim_gw.utils.text_composer.utils import inspect_all_choices
# from bim_gw.utils.text_composer.modifiers import MixModifier, DeleteModifier
from bim_gw.utils.text_composer.writers import writers


def a_has_n(sentence):
    def aux(attributes):
        vowels = ["a", "e", "i", "o", "u"]
        if attributes[attributes['_next']][0] in vowels:
            return sentence.format(**{"n?": "n"})
        return sentence.format(**{"n?": ""})

    return aux


# or from a kwargs given to the composer method.
script_structures = [
    "{start} <{size}>{link}<{color}>{link}<{shape}>{link}<{location}>{"
    "link}<{rotation}>.",
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

random_composer = Composer(script_structures, writers, variants, modifiers)

script_structures = [
    "{start} {size} {colorBefore} {shape}{link} <{location}>{link} <{"
    "rotation}>.",
    "{start} {size} {colorBefore} {object} <{location}>{link} <{rotation}>{"
    "link2} {a} {shape}.",
    "{start} {size} {object} <{location}>{link} <{rotation}>{link} {"
    "colorBoth}{link2} {a} {shape}.",
    "{start} {size} {object} in {colorAfter} <{location}>{link} <{"
    "rotation}>{link2} {a} {shape}.",
    "{start} {shape}{link} {size}{link3} {colorBoth}{link} <{location}>{"
    "link} <{rotation}>.",
    "{start} {color}{colored?} {size} {shape}{link} <{location}>{link} <{"
    "rotation}>.",
    "{start} {color}{colored?} {shape}{link} <{size}>{link} <{location}>{"
    "link} <{rotation}>.",
]

start_variant = ["A", "It is a", "This is a", "There is a",
                 "The image is a", "The image represents a",
                 "The image contains a"]
start_variant = [a_has_n(x + "{n?}") for x in start_variant] + ['A kind of']

# Elements in the list of each variant is randomly chosen.
variants = {
    "start": start_variant,
    "a": [a_has_n("a{n?}")],
    "object": ["shape", "object"],
    "colored?": ["", " colored"],
    "colorBefore": ["{color}", "{color} colored"],
    "colorAfter": ["{color}", "{color} color"],
    "colorBoth": ["{color}", "in {color}", "{color} colored",
                  "in {color} color"],
    ",?": [", ", " "],
    "link": [". It is", ", it is", "{,?}and is", "{,?}and it is", ",", ],
    "link2": [". It looks like", ", it looks like", "{,?}and looks like",
              "{,?}and it looks like"],
    "link3": [". It is", ", it is", "{,?}and is", "{,?}and it is", ",",
              " and"],
}

composer = Composer(script_structures, writers, variants, modifiers)

if __name__ == '__main__':

    args = get_args(debug=bool(os.getenv("DEBUG", False)))
    all_choices = inspect_all_choices(composer)
    dataset_location = Path(args.simple_shapes_path)
    for split in ["train", "val", "test"]:
        labels = np.load(str(dataset_location / f"{split}_labels.npy"))
        captions = []
        choices = []
        for k in tqdm(range(len(labels)), total=len(labels)):
            caption, choice = composer(
                {
                    "shape": int(labels[k][0]),
                    "rotation": labels[k][4],
                    "color": (labels[k][5], labels[k][6], labels[k][7]),
                    "size": labels[k][3],
                    "location": (labels[k][1], labels[k][2])
                }
            )
            captions.append(caption)
            choices.append(choice)
        np.save(str(dataset_location / f"{split}_captions.npy"), captions)
        np.save(
            str(dataset_location / f"{split}_caption_choices.npy"), choices
        )
