import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.utils import get_args
from bim_gw.utils.shapes import generate_image, generate_dataset, generate_transformations
from bim_gw.utils.text_composer.bert import get_bert_latents
from bim_gw.utils.text_composer.composer import composer


def save_dataset(path_root, dataset, imsize):
    dpi = 1
    classes, locations, radii = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    for k, (cls, location, scale, rotation, color) in tqdm(enumerate(zip(classes, locations, radii, rotations, colors)),
                                                           total=len(classes)):
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        generate_image(ax, cls, location, scale, rotation, color, imsize)
        ax.set_facecolor("black")
        # patch = patches.Circle((location[0], location[1]), 1, facecolor="white")
        # ax.add_patch(patch)
        plt.tight_layout(pad=0)
        # plt.show()
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)


def save_labels(path_root, dataset, dataset_transfo):
    classes, locations, sizes = dataset["classes"], dataset["locations"], dataset["sizes"]
    rotations, colors = dataset["rotations"], dataset["colors"]
    colors_hls = dataset["colors_hls"]
    classes_transfo, locations_transfo, sizes_transfo = dataset_transfo["classes"], dataset_transfo["locations"], \
                                                        dataset_transfo["sizes"]
    rotations_transfo, colors_transfo = dataset_transfo["rotations"], dataset_transfo["colors"]
    colors_hls_transfo = dataset_transfo["colors_hls"]
    # header = ["class", "x", "y", "scale", "rotation", "r", "g", "b", "h", "l", "s", "t_class", "d_x", "d_y", "d_scale",
    #           "d_rotation", "d_r", "d_g", "d_b", "d_h", "d_s", "d_l"]
    labels = np.concatenate([
        classes.reshape((-1, 1)), locations, sizes.reshape((-1, 1)), rotations.reshape((-1, 1)), colors, colors_hls,
        classes_transfo.reshape((-1, 1)), locations_transfo, sizes_transfo.reshape((-1, 1)),
        rotations_transfo.reshape((-1, 1)), colors_transfo, colors_hls_transfo
    ], axis=1).astype(np.float32)
    np.save(path_root, labels)


def main():
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    seed = args.seed
    image_size = args.img_size

    dataset_location = Path(args.simple_shapes_path)
    dataset_location.mkdir(exist_ok=True)

    size_train_set = 1_000_000
    size_val_set = 50_000
    size_test_set = 50_000

    # in pixels
    min_scale = 7
    max_scale = 14
    min_lightness = 46  # of the HSL format. Higher value generates lighter images. Min 0, Max 256
    max_lightness = 256

    np.random.seed(seed)

    train_labels = generate_dataset(size_train_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    train_transfo_labels, train_transfo = generate_transformations(train_labels,
                                                                   generate_dataset(size_train_set, min_scale,
                                                                                    max_scale,
                                                                                    min_lightness, max_lightness,
                                                                                    image_size))
    val_labels = generate_dataset(size_val_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    val_transfo_labels, val_transfo = generate_transformations(val_labels,
                                                               generate_dataset(size_val_set, min_scale, max_scale,
                                                                                min_lightness,
                                                                                max_lightness, image_size))
    test_labels = generate_dataset(size_test_set, min_scale, max_scale, min_lightness, max_lightness, image_size)
    test_transfo_labels, test_transfo = generate_transformations(test_labels,
                                                                 generate_dataset(size_test_set, min_scale, max_scale,
                                                                                  min_lightness,
                                                                                  max_lightness, image_size))

    print("Save labels...")
    save_labels(dataset_location / "train_labels.npy", train_labels, train_transfo)
    save_labels(dataset_location / "val_labels.npy", val_labels, val_transfo)
    save_labels(dataset_location / "test_labels.npy", test_labels, test_transfo)

    print("Saving training set...")
    (dataset_location / "transformed").mkdir(exist_ok=True)
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "train", train_labels, image_size)
    (dataset_location / "transformed" / "train").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "train", train_transfo_labels, image_size)
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", val_labels, image_size)
    (dataset_location / "transformed" / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "val", val_transfo_labels, image_size)
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "test", test_labels, image_size)
    (dataset_location / "transformed" / "test").mkdir(exist_ok=True)
    save_dataset(dataset_location / "transformed" / "test", test_transfo_labels, image_size)

    print("Saving captions...")

    for split in ["train", "val", "test"]:
        labels = np.load(str(dataset_location / f"{split}_labels.npy"))
        captions = []
        for k in tqdm(range(labels.shape[0]), total=labels.shape[0]):
            captions.append(composer({
                "shape": int(labels[k][0]),
                "rotation": labels[k][4],
                "color": (labels[k][5], labels[k][6], labels[k][7]),
                "size": labels[k][3] // 4,
                "location": (labels[k][1] // 4, labels[k][2] // 4)
            }))
        np.save(str(dataset_location / f"{split}_captions.npy"), captions)

    print("Extracting BERT features...")
    bert_latents = args.fetchers.t.bert_latents
    args.fetchers.t.bert_latents = None
    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = {"t": "t"}
    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")
    get_bert_latents(data, args.global_workspace.bert_path, bert_latents, args.simple_shapes_path, device)

    print('done!')


if __name__ == '__main__':
    main()
