import math
import re
from itertools import permutations
from typing import Any, Dict

from attributes_to_language.composer import Composer


def inspect_writers(composer):
    choices = dict()
    for writer_name, writers in composer.writers.items():
        if len(writers) > 1:
            choices[f"writer_{writer_name}"] = len(writers)
        for k, writer in enumerate(writers):
            for variant_name, variant in writer.variants.items():
                if len(variant) > 1:
                    choices[f"writer_{writer_name}_{k}_{variant_name}"] = len(
                        variant
                    )
    return choices


def inspect_all_choices(composer: Composer):
    num_structures = 0
    choices = dict()
    for structure in composer.script_structures:
        num_structures += math.factorial(
            len(re.findall(r"<[^>]+>", structure))
        )
    choices["structure"] = num_structures
    for variant_name, variant in composer.variants.items():
        if len(variant) > 1:
            choices[f"variant_{variant_name}"] = len(variant)
    choices.update(inspect_writers(composer))
    return choices


def get_categories(
    composer: Composer, choices: Dict[str, Any]
) -> Dict[str, int]:
    categories = dict()
    # structure
    class_val = 0
    for k, structure in enumerate(composer.script_structures):
        groups = re.findall(r"<[^>]+>", structure)
        if choices["structure"] != k:
            class_val += math.factorial(len(groups))
        else:
            for permutation in permutations(range(len(groups))):
                if choices["groups"] == list(permutation):
                    categories["structure"] = class_val
                    break
                class_val += 1
    # variants
    for name in composer.variants.keys():
        if name in choices["variants"]:
            categories[f"variant_{name}"] = choices["variants"][name]
        else:
            categories[f"variant_{name}"] = 0
    # writers
    for name in inspect_writers(composer).keys():
        split_name = name.split("_")
        writer_name = split_name[1]
        categories[name] = 0
        if len(split_name) == 2:
            categories[name] = choices["writers"][writer_name]["_writer"]
        elif writer_name in choices["writers"]:
            variant_name = split_name[3]
            variant_choice = int(split_name[2])
            if (
                variant_name in choices["writers"][writer_name]
                and choices["writers"][writer_name]["_writer"]
                == variant_choice
            ):
                categories[name] = choices["writers"][writer_name][
                    variant_name
                ]
    return categories


def get_choices_from_structure_category(composer, grammar_predictions):
    all_choices = []
    for i in range(len(grammar_predictions["structure"])):
        choices = {
            "variants": {
                name.replace("variant_", ""): variant[i]
                for name, variant in grammar_predictions.items()
                if "variant_" in name
            },
            "writers": {},
        }
        # writers
        for name, variant in grammar_predictions.items():
            if "writer_" in name:
                split_name = name.split("_")
                writer_name = split_name[1]
                if writer_name not in choices["writers"]:
                    choices["writers"][writer_name] = {}
                if len(split_name) == 2:
                    choices["writers"][writer_name]["_writer"] = variant[i]
                else:
                    variant_choice = int(split_name[2])
                    if (
                        grammar_predictions[f"writer_{writer_name}"][i]
                        == variant_choice
                    ):
                        variant_name = split_name[3]
                        choices["writers"][writer_name][
                            variant_name
                        ] = variant[i]
        # structure
        category = grammar_predictions["structure"][i]
        for k, structure in enumerate(composer.script_structures):
            groups = re.findall(r"<[^>]+>", structure)
            if category < math.factorial(len(groups)):
                choices["structure"] = k
                choices["groups"] = list(
                    list(permutations(range(len(groups))))[category]
                )
                all_choices.append(choices)
                break
            category -= math.factorial(len(groups))
    return all_choices
