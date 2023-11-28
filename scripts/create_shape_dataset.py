import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.utils import get_args
from bim_gw.utils.shapes import (
    generate_dataset,
    generate_unpaired_attr,
    load_labels,
    save_dataset,
    save_labels,
)
from bim_gw.utils.text_composer.bert import save_bert_latents
from bim_gw.utils.text_composer.composer import composer


def main(args):
    seed = args.seed
    image_size = args.img_size

    dataset_location = Path(args.simple_shapes_path)
    dataset_location.mkdir(exist_ok=True)

    size_train_set = args.datasets.shapes.n_train_examples
    size_val_set = args.datasets.shapes.n_val_examples
    size_test_set = args.datasets.shapes.n_test_examples

    # in pixels
    min_scale = args.datasets.shapes.min_scale
    max_scale = args.datasets.shapes.max_scale
    # of the HSL format. Higher value generates lighter images.
    # Min 0, Max 256
    min_lightness = args.datasets.shapes.min_lightness
    max_lightness = args.datasets.shapes.max_lightness

    min_hue = args.datasets.shapes.min_hue
    max_hue = args.datasets.shapes.max_hue

    possible_categories = args.datasets.shapes.possible_categories

    min_rotation = args.datasets.shapes.min_rotation
    max_rotation = args.datasets.shapes.max_rotation

    min_x = args.datasets.shapes.min_x
    max_x = args.datasets.shapes.max_x

    min_y = args.datasets.shapes.min_y
    max_y = args.datasets.shapes.max_y

    shapes_color_range = args.datasets.shapes.shapes_color_range

    np.random.seed(seed)

    train_labels = generate_dataset(
        size_train_set,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        shapes_color_range,
        image_size,
    )
    val_labels = generate_dataset(
        size_val_set,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        shapes_color_range,
        image_size,
    )
    test_labels = generate_dataset(
        size_test_set,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        shapes_color_range,
        image_size,
    )

    print("Save labels...")
    save_labels(dataset_location / "train_labels.npy", train_labels)
    save_labels(dataset_location / "val_labels.npy", val_labels)
    save_labels(dataset_location / "test_labels.npy", test_labels)

    print("Saving training set...")
    (dataset_location / "transformed").mkdir(exist_ok=True)
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)

    print("Saving captions...")

    for split in ["train", "val", "test"]:
        labels = np.load(str(dataset_location / f"{split}_labels.npy"))
        captions = []
        choices = []
        for k in tqdm(range(labels.shape[0]), total=labels.shape[0]):
            caption, choice = composer(
                {
                    "shape": int(labels[k][0]),
                    "rotation": labels[k][4],
                    "color": (labels[k][5], labels[k][6], labels[k][7]),
                    "size": labels[k][3],
                    "location": (labels[k][1], labels[k][2]),
                }
            )
            captions.append(caption)
            choices.append(choice)
        np.save(str(dataset_location / f"{split}_captions.npy"), captions)
        np.save(
            str(dataset_location / f"{split}_caption_choices.npy"), choices
        )

    print("Extracting BERT features...")
    bert_latents = args.domain_loader.t.bert_latents
    args.domain_loader.t.bert_latents = None
    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.0
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = OmegaConf.create(["t"])
    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")
    save_bert_latents(
        data,
        args.global_workspace.bert_path,
        bert_latents,
        args.simple_shapes_path,
        device,
    )

    print("done!")


def other():
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))
    seed = args.seed
    np.random.seed(seed)

    dataset_location = Path(args.simple_shapes_path)
    for path_name in ["train_labels_2", "val_labels", "test_labels"]:
        # for path_name in ["val_labels"]:
        dataset, dataset_transfo = load_labels(
            dataset_location / (path_name + ".npy")
        )
        dataset["unpaired"] = generate_unpaired_attr(
            dataset["classes"].shape[0]
        )
        dataset_transfo["unpaired"] = np.zeros_like(dataset["unpaired"])
        save_labels(
            dataset_location / f"{path_name}_2.npy", dataset, dataset_transfo
        )


if __name__ == "__main__":
    main(get_args(debug=bool(int(os.getenv("DEBUG", 0)))))
