import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from tqdm import tqdm

from bim_gw.utils import get_args


def closest_item(ref, labels, keys):
    dists = np.linalg.norm(labels[:, keys] - ref[keys], axis=1)
    return np.argsort(dists)[1]


def far_item(ref1, ref2, labels, keys):
    dists = np.minimum(np.min(np.abs(labels - ref1), axis=1), np.min(np.abs(labels - ref2), axis=1))
    # dists += np.linalg.norm(labels[:, keys] - ref1[keys], axis=1) + np.linalg.norm(labels[:, keys] - ref2[keys], axis=1)
    sorted_dists = np.argsort(-dists)
    # return sorted_dists[0]
    return np.random.choice(sorted_dists[:sorted_dists.shape[0] // 1000], 1)[0]

def normalize_labels(labels):
    labels -= labels.min(axis=0)
    labels /= labels.max(axis=0)
    return labels

def get_img(path, split, idx):
    with open(path / split / f"{idx}.png", 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

def frame_image(img, frame_width):
    img = ImageOps.crop(img, border=frame_width)
    img_with_border = ImageOps.expand(img, border=frame_width, fill='red')
    return img_with_border

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    root_path = Path(args.simple_shapes_path)
    split = "train"
    possible_keys = [[0], [1, 2], [3], [4], [5, 6, 7]]
    all_labels = np.load(str(root_path / f"{split}_labels.npy"))[:, :8]
    all_labels = normalize_labels(all_labels)

    for split in ["train"]:
        dataset = []
        if split == "train":
            n_examples = 8
            labels = all_labels[:500_000]
        elif split == "val":
            n_examples = 1000
            labels = all_labels[500_000:750_000]
        else:
            n_examples = 1000
            labels = all_labels[750_000:]

        fig, axes = plt.subplots(n_examples, 3, figsize=(3, n_examples))
        for i in tqdm(range(n_examples), total=n_examples):
            ref = labels[i]
            key = random.choice(possible_keys)
            closest_key = closest_item(ref, labels, key)
            rd = far_item(ref, labels[closest_key], labels, key)
            order = np.random.permutation(3)
            idx = [i, closest_key, rd]
            dataset.append([idx[order[0]], idx[order[1]], idx[order[2]], np.where(order==2)[0][0]])
            axes[i, order[0]].imshow(get_img(root_path, split, i))
            axes[i, order[1]].imshow(get_img(root_path, split, closest_key))
            axes[i, order[2]].imshow(frame_image(get_img(root_path, split, rd), 2))
            print(key)
            for k in range(3):
                axes[i, k].set_xticks([])
                axes[i, k].set_yticks([])
                axes[i, k].grid(False)
                axes[i, k].set_aspect('equal')
        plt.tight_layout(pad=0)
        plt.savefig('../data/example_odd_dataset.pdf')
        plt.show()
        # np.save(str(root_path / f"{split}_odd_image_labels.npy"), dataset)
