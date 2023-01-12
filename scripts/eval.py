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

    dataset = iter(data.test_dataloader()[0])
    domains = next(dataset)
    domains, targets = domains[0], domains[1]
    available_domains, domains = split_domains_available_domains(domains)
    _, targets = split_domains_available_domains(targets)

    global_workspace = GlobalWorkspace.load_from_checkpoint(args.checkpoint,
                                                            domain_mods=get_domains(args, data.img_size), strict=False)

    visual_model = global_workspace.domain_mods["v"]
    text_model = global_workspace.domain_mods["t"]

    latents = global_workspace.encode_uni_modal(domains)

    visual_model.log_domain(None, visual_model.decode(latents["v"]), "original v")
    # text_model.log_domain(None, text_model.decode(latents["t"]), "original t")

    state_v = global_workspace.project(latents, available_domains, keep_domains=["v"])
    state_t = global_workspace.project(latents, available_domains, keep_domains=["t"])

    predictions_from_v = global_workspace.adapt(global_workspace.predict(state_v))
    predictions_from_t = global_workspace.adapt(global_workspace.predict(state_t))
    noisy_v = [predictions_from_t["v"][0] + torch.randn_like(predictions_from_t["v"][0]) * 0.1]

    visual_model.log_domain(None, visual_model.decode(predictions_from_v["v"]), "Demi-cycle v")
    text_model.log_domain(None, text_model.decode(predictions_from_v["t"]), "Translation v to t")
    visual_model.log_domain(None, visual_model.decode(predictions_from_t["v"]), "Translation t to v")
    # visual_model.log_domain(None, visual_model.decode(noisy_v), "Noisy recons")
    text_model.log_domain(None, text_model.decode(predictions_from_t["t"]), "Demi-cycle t")

    cycle_state_v_t = global_workspace.project(predictions_from_v, keep_domains=["t"])
    cycle_state_t_v = global_workspace.project(predictions_from_t, keep_domains=["v"])

    cycle_predictions_from_v = global_workspace.predict(cycle_state_v_t)
    cycle_predictions_from_t = global_workspace.predict(cycle_state_t_v)

    visual_model.log_domain(None, visual_model.decode(cycle_predictions_from_t["v"]), "Cycle v through t")
    text_model.log_domain(None, text_model.decode(cycle_predictions_from_t["t"]), "Demi-cycle t from predicted by v")
    text_model.log_domain(None, text_model.decode(cycle_predictions_from_v["t"]), "Cycle t through v")
    visual_model.log_domain(None, visual_model.decode(cycle_predictions_from_v["v"]), "Demi-cycle v from predicted by t")
    print("plot")
