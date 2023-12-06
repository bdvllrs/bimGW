import os
from collections import defaultdict
from typing import cast

import torch
from pytorch_lightning import seed_everything

from bim_gw.datasets.utils import load_dataset
from bim_gw.modules.domain_modules.simple_shapes.text import SimpleShapesText
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.utils import get_args
from bim_gw.utils.loggers.loggers import get_loggers
from bim_gw.utils.scripts import get_domains
from bim_gw.utils.text_composer.utils import get_categories
from bim_gw.utils.utils import get_checkpoint_path

if __name__ == "__main__":
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))

    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace)

    checkpoint_path = get_checkpoint_path(args.checkpoint)
    # global_workspace = GlobalWorkspace.load_from_checkpoint(
    #     checkpoint_path,
    #     domain_mods=get_domains(args, data.img_size),
    # )
    global_workspace = GlobalWorkspace(
        get_domains(args, data.img_size),
        args.global_workspace.z_size,
        args.global_workspace.hidden_size,
        args.global_workspace.n_layers.encoder,
        args.global_workspace.n_layers.decoder,
        args.global_workspace.n_layers.decoder_head,
        args.losses.coefs.demi_cycles,
        args.losses.coefs.cycles,
        args.losses.coefs.translation,
        args.losses.coefs.contrastive,
        args.global_workspace.optim.lr,
        args.global_workspace.optim.weight_decay,
        args.global_workspace.optim.unsupervised_losses_after_n_epochs,
        args.global_workspace.scheduler.mode,
        args.global_workspace.scheduler.interval,
        args.global_workspace.scheduler.step,
        args.global_workspace.scheduler.gamma,
        args.losses.schedules,
        args.global_workspace.monitor_grad_norms,
        args.global_workspace.remove_sync_domains,
    )

    text_domain: SimpleShapesText = cast(
        SimpleShapesText, global_workspace.domains["t"]
    )

    data.prepare_data()
    data.setup(stage="fit")
    dataloader = data.val_dataloader()[0]

    slurm_job_id = os.getenv("SLURM_JOBID", None)

    tags = None
    version = args.run_name
    if slurm_job_id is not None:
        tags = ["slurm"]
    source_files = [
        "../**/*.py",
        "../README.md",
        "../requirements.txt",
        "../**/*.yaml",
    ]

    if args.logger_resume_id is not None:
        for logger in args.loggers:
            logger.args.version = args.logger_resume_id
            logger.args.id = args.logger_resume_id
            logger.args.resume = True

    loggers = get_loggers(
        "eval_grammar",
        version,
        args.loggers,
        global_workspace,
        args,
        tags,
        source_files,
    )

    n_equals = defaultdict(int)
    n_total = defaultdict(int)

    for batch in dataloader:
        latents = global_workspace.domains.encode(batch)
        latents_sub_parts = {
            domain_name: latent.sub_parts
            for domain_name, latent in latents.items()
        }
        visual_multimodal_latent = global_workspace.project(
            latents_sub_parts, keep_domains=["v"]
        )
        predicted_latents = global_workspace.domains.adapt(
            global_workspace.predict(visual_multimodal_latent)
        )["t"]

        decoded_outputs = text_domain.decode(predicted_latents)["choices"]

        outputs = {
            name: torch.tensor(
                [
                    get_categories(
                        text_domain.text_composer, decoded_outputs[k]
                    )[name]
                    for k in range(len(decoded_outputs))
                ]
            )
            for name in batch["t"]["choices"].keys()
        }

        for name, prediction in outputs.items():
            n_equals[name] = (prediction == batch["t"]["choices"][name]).sum()
            n_total[name] = prediction.size(0)

    acc = {name: n_equals[name] / n_total[name] for name in n_equals.keys()}
    print(acc)

    for logger in loggers:
        for name, val in acc.items():
            logger.log_metric(f"acc_{name}", val)
