import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.functional import normalize
from tqdm import tqdm

from bim_gw.datasets import load_dataset
from bim_gw.datasets.domain import DomainItems, collate_fn
from bim_gw.datasets.simple_shapes.data_modules import SimpleShapesDataModule
from bim_gw.datasets.simple_shapes.types import ShapesAvailableDomains
from bim_gw.modules import GlobalWorkspace
from bim_gw.utils import get_args
from bim_gw.utils.scripts import get_domains
from bim_gw.utils.shapes import generate_dataset
from bim_gw.utils.text_composer.bert import embed_text, get_language_model
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.utils import get_checkpoint_path


def get_captions(labels, shape, choices=None):
    captions = []
    new_choices = []
    n_captions = labels["rotations"].shape[0]
    for k in tqdm(range(n_captions), total=n_captions):
        caption, new_choice = composer(
            {
                "shape": shape,
                "rotation": labels["rotations"][k],
                "color": (
                    labels["colors"][k, 0],
                    labels["colors"][k, 1],
                    labels["colors"][k, 2],
                ),
                "size": labels["sizes"][k],
                "location": (
                    labels["locations"][k, 0],
                    labels["locations"][k, 1],
                ),
            },
            choices[k] if choices is not None else None,
        )
        captions.append(caption)
        new_choices.append(new_choice)
    return captions, new_choices


if __name__ == "__main__":
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))

    args.global_workspace.selected_domains = OmegaConf.create(
        ["v", "attr", "t"]
    )

    data: SimpleShapesDataModule = load_dataset(
        args, args.global_workspace, add_unimodal=False
    )
    data.prepare_data()
    data.setup(stage="fit")

    args.global_workspace.selected_domains = OmegaConf.create(["v", "t"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, transformer = get_language_model(
        device, args.global_workspace.bert_path
    )

    num_captions = 1000
    random_attributes = generate_dataset(
        num_captions,
        args.datasets.shapes.min_scale,
        args.datasets.shapes.max_scale,
        args.datasets.shapes.min_lightness,
        args.datasets.shapes.max_lightness,
        args.datasets.shapes.possible_categories,
        args.datasets.shapes.min_hue,
        args.datasets.shapes.max_hue,
        args.datasets.shapes.min_rotation,
        args.datasets.shapes.max_rotation,
        args.datasets.shapes.min_x,
        args.datasets.shapes.max_x,
        args.datasets.shapes.min_y,
        args.datasets.shapes.max_y,
        args.datasets.shapes.shapes_color_range,
        args.img_size,
    )
    diamond_captions, choices = get_captions(random_attributes, 0)
    new_choices = []
    for choice in choices:
        choice["writers"] = {
            key: val
            for key, val in choice["writers"].items()
            if key != "shape"
        }
        new_choices.append(choice)
    egg_captions, _ = get_captions(random_attributes, 1, new_choices)
    new_choices = []
    for choice in choices:
        choice["writers"] = {
            key: val
            for key, val in choice["writers"].items()
            if key != "shape"
        }
        new_choices.append(choice)
    triangle_captions, _ = get_captions(random_attributes, 2, new_choices)

    captions = [*diamond_captions, *egg_captions, *triangle_captions]

    bert_captions = embed_text(tokenizer, transformer, captions, device)
    bert_mean = torch.from_numpy(
        np.load(
            Path(args.simple_shapes_path)
            / f"mean_{args.global_workspace.bert_path}.npy"
        )
    ).to(device)
    bert_std = torch.from_numpy(
        np.load(
            Path(args.simple_shapes_path)
            / f"std_{args.global_workspace.bert_path}.npy"
        )
    ).to(device)
    bert_captions = (bert_captions - bert_mean) / bert_std

    # bert_diamond = bert_captions[:num_captions].mean(dim=0)
    # bert_egg = bert_captions[num_captions : 2 * num_captions].mean(dim=0)
    # bert_triangle = bert_captions[2 * num_captions :].mean(dim=0)
    #
    # bert_captions = torch.stack([bert_diamond, bert_egg, bert_triangle])

    t = ShapesAvailableDomains.t

    bert_items = collate_fn(
        [
            {
                t: DomainItems.singular(
                    bert=bert,
                    text=caption,
                    is_available=False,
                    choices=data.test_set["in_dist"]
                    .domain_loaders[t]
                    .null_choice,
                )
            }
            for bert, caption in zip(bert_captions, captions)
        ]
    )
    bert_items[t].to_device(device)

    for model in args.downstream.clip_cls.checkpoints:
        print(model)
        checkpoint_path = get_checkpoint_path(model)
        domain_mods = get_domains(args, data.img_size)
        gw = cast(
            GlobalWorkspace,
            GlobalWorkspace.load_from_checkpoint(
                str(checkpoint_path),
                domain_mods=domain_mods,
                optim_unsupervised_losses_after=args.global_workspace.optim.unsupervised_losses_after_n_epochs,
                domain_examples=data.domain_examples,
                strict=False,
            ),
        )
        gw.eval().freeze()
        gw.to(device)
        dataset = iter(data.val_dataloader()[0])

        with gw.domains.pass_through(False):
            target_latents = gw.domains.encode(bert_items)
            predicted_v_latent = gw.translate(
                target_latents[t].sub_parts, "t", "v"
            )
            # gw.domains["v"].log_domain_from_latent(
            #     None, predicted_v_latent, "test"
            # )
            target_gw = gw.project(
                {"t": target_latents[t].sub_parts},
                ["t"],
            )
            target_gw_diamond = target_gw[:num_captions].mean(dim=0)
            target_gw_egg = target_gw[num_captions : 2 * num_captions].mean(
                dim=0
            )
            target_gw_triangles = target_gw[2 * num_captions :].mean(dim=0)
            target_gw = torch.stack(
                [target_gw_diamond, target_gw_egg, target_gw_triangles]
            )
            target_gw_n = normalize(target_gw, dim=1)

        acc, total = 0, 0
        for domains in dataset:
            domains_v = {"v": domains["v"].to_device(device)}
            latent_v = gw.domains.encode(domains_v)["v"]
            state_v = gw.project({"v": latent_v.sub_parts}, ["v"])
            state_v_n = normalize(state_v, dim=1)
            pred = gw.domains.adapt(gw.predict(state_v))
            # print(gw.domains["t"].decode(pred["t"])["text"][:32])
            # gw.domains["v"].log_domain_from_latent(
            #     None, domains["v"].sub_parts, "test", 32
            # )
            # gw.domains["v"].log_domain_from_latent(None, latent_v, "test", 32)

            cos_v_t = state_v_n @ target_gw_n.t()
            prediction = torch.argmax(cos_v_t, dim=1)
            target = domains["attr"].sub_parts["cls"].to(device)
            acc += (prediction == target).sum()
            total += prediction.size(0)
            break
        print(acc / total)
