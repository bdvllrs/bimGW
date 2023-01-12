import os

import torch
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.gw import split_domains_available_domains
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    seed_everything(args.seed)

    args.global_workspace.batch_size = 32
    args.global_workspace.use_pre_saved = False

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    sentences = [
        "There is an object in the image.",
        "There is an object in the bottom-right hand corner of the image.",
        "A diamond is at the top.",
        "A diamond is at the bottom.",
        "A diamond. It is at the bottom.",
        "A diamond is at the very bottom.",
        "A diamond. It is at the very bottom.",
        "A big red egg.",
        "A big red color egg.",
        "Purple.",
        "Purple color.",
        "A medium four-sided shape in bright red color. It is in the upper side, slightly left and is pointing to the left top-left corner.",
        "A medium four-sided shape in bright red color.",
        "The image represents an egg shape. It is in the upper side, slightly left."
    ]

    sentences += ["" for k in range(32 - len(sentences))]

    global_workspace = GlobalWorkspace.load_from_checkpoint(args.checkpoint,
                                                            domain_mods=get_domains(args, data.img_size), strict=False)

    visual_model = global_workspace.domain_mods["v"]
    text_model = global_workspace.domain_mods["t"]

    bert_latents = text_model.get_bert_latent(sentences)
    domains = {"t": [bert_latents, sentences]}

    latents = global_workspace.encode_uni_modal(domains)

    # visual_model.log_domain(None, visual_model.decode(latents["v"]), "original v")
    text_model.log_domain(None, text_model.decode(latents["t"]), "original t")

    # state_v = global_workspace.project(latents, keep_domain="v")
    state_t = global_workspace.project(latents, keep_domain="t")
    predictions_from_t = global_workspace.adapt(global_workspace.predict(state_t))
    # noisy_v = [predictions_from_t["v"][0] + torch.randn_like(predictions_from_t["v"][0]) * 0.1]

    visual_model.log_domain(None, visual_model.decode(predictions_from_t["v"]), "Translation t to v")
    # visual_model.log_domain(None, visual_model.decode(noisy_v), "Noisy recons")
    # text_model.log_domain(None, text_model.decode(predictions_from_t["t"]), "Demi-cycle t")

    cycle_state_t_v = global_workspace.project(predictions_from_t, keep_domain="v")

    cycle_predictions_from_t = global_workspace.predict(cycle_state_t_v)

    visual_model.log_domain(None, visual_model.decode(cycle_predictions_from_t["v"]), "Cycle v through t")
    text_model.log_domain(None, text_model.decode(cycle_predictions_from_t["t"]), "Demi-cycle t from predicted by v")
    print("plot")
