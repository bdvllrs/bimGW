import os

import torch
from omegaconf import OmegaConf

from bim_gw.datasets import load_dataset
from bim_gw.utils import get_args
from bim_gw.utils.text_composer.bert import save_bert_latents

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    assert args.global_workspace.load_pre_saved_latents is not None, "Pre-saved latent path should be defined."

    args.seed = 0
    bert_latents = args.fetchers.t.bert_latents
    args.fetchers.t.bert_latents = None
    args.global_workspace.use_pre_saved = False
    args.global_workspace.prop_labelled_images = 1.
    args.global_workspace.split_ood = False
    args.global_workspace.sync_uses_whole_dataset = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.global_workspace.selected_domains = OmegaConf.create(["t"])

    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    save_bert_latents(data, args.global_workspace.bert_path, bert_latents, args.simple_shapes_path, device)
