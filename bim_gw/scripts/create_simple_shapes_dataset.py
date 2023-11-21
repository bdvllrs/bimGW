from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.utils.shapes import (
    generate_dataset,
    generate_transformations,
    save_dataset,
    save_labels,
)
from bim_gw.utils.text_composer.bert import save_bert_latents
from bim_gw.utils.text_composer.composer import composer


def create_dataset(
    simple_shapes_path,
    seed,
    image_size,
    n_train,
    n_val,
    n_test,
    possible_categories,
    min_x,
    max_x,
    min_y,
    max_y,
    min_scale,
    max_scale,
    min_rotation,
    max_rotation,
    min_lightness,
    max_lightness,
    min_hue,
    max_hue,
    args=None,
):
    dataset_location = Path(simple_shapes_path)
    dataset_location.mkdir(exist_ok=True)

    np.random.seed(seed)

    train_labels = generate_dataset(
        n_train,
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
        image_size,
    )
    train_transfo_labels, train_transfo = generate_transformations(
        train_labels,
        generate_dataset(
            n_train,
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
            image_size,
        ),
    )
    val_labels = generate_dataset(
        n_val,
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
        image_size,
    )
    val_transfo_labels, val_transfo = generate_transformations(
        val_labels,
        generate_dataset(
            n_val,
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
            image_size,
        ),
    )
    test_labels = generate_dataset(
        n_test,
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
        image_size,
    )
    test_transfo_labels, test_transfo = generate_transformations(
        test_labels,
        generate_dataset(
            n_test,
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
            image_size,
        ),
    )

    print("Save labels...")
    save_labels(
        dataset_location / "train_labels.npy", train_labels, train_transfo
    )
    save_labels(dataset_location / "val_labels.npy", val_labels, val_transfo)
    save_labels(
        dataset_location / "test_labels.npy", test_labels, test_transfo
    )

    print("Saving training set...")
    (dataset_location / "transformed").mkdir(exist_ok=True)
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    (dataset_location / "transformed" / "train").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "train",
        train_transfo_labels,
        image_size,
    )
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    (dataset_location / "transformed" / "val").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "val",
        val_transfo_labels,
        image_size,
    )
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)
    (dataset_location / "transformed" / "test").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "test",
        test_transfo_labels,
        image_size,
    )

    if args is not None:
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
