import re
import math

from attributes_to_language.composer import Composer
from attributes_to_language.writers import Bins2dWriter, BinsWriter, QuantizedWriter, OptionsWriter


def inspect_all_choices(composer: Composer):
    num_structures = 0
    choices = dict()
    for structure in composer.script_structures:
        num_structures += math.factorial(len(re.findall(r"<[^>]+>", structure)))
    choices['structures'] = num_structures
    for variant_name, variant in composer.variants.items():
        if len(variant) > 1:
            choices[f'variant_{variant_name}'] = len(variant)
    for writer_name, writers in composer.writers.items():
        if len(writers) > 1:
            choices[f"writer_{writer_name}"] = len(writers)
        for k, writer in enumerate(writers):
            for variant_name, variant in writer.variants.items():
                if len(variant) > 1:
                    choices[f"writer_{writer_name}_{k}_{variant_name}"] = len(variant)
            if isinstance(writer, (Bins2dWriter, BinsWriter, QuantizedWriter)):
                for i, label in enumerate(writer.labels):
                    if type(label) is list:
                        if len(label) > 1:
                            choices[f"writer_{writer_name}_{k}_val_{i}"] = len(label)
            elif isinstance(writer, OptionsWriter):
                for i, label in enumerate(writer.choices):
                    if type(label) is list:
                        if len(label) > 1:
                            choices[f"writer_{writer_name}_{k}_val_{i}"] = len(label)
    return choices







