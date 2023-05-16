import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers
from bim_gw.utils.scripts import get_domains
from bim_gw.utils.utils import get_checkpoint_path

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    args.global_workspace.selected_domains = OmegaConf.create(["attr"])

    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    assert args.downstream.unpaired_cls.checkpoint is not None, (
        "You must " "provide a " "checkpoint " "for this " "script."
    )
    checkpoint_path = get_checkpoint_path(
        args.downstream.unpaired_cls.checkpoint
    )
    domain_mods = get_domains(args, data.img_size)
    global_workspace = GlobalWorkspace.load_from_checkpoint(
        checkpoint_path,
        domain_mods=domain_mods,
        domain_examples=data.domain_examples,
        strict=False,
    )
    global_workspace.eval().freeze()

    args.losses.coefs = OmegaConf.create(
        {
            "translation": global_workspace.hparams["loss_coef_translation"],
            "cycles": global_workspace.hparams["loss_coef_cycles"],
            "demi_cycles": global_workspace.hparams["loss_coef_demi_cycles"],
            "contrastive": global_workspace.hparams["loss_coef_contrastive"],
        }
    )

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
    loggers = get_loggers(
        "train_odd_image",
        version,
        args.loggers,
        global_workspace,
        args,
        tags,
        source_files,
    )

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=(args.distributed_backend if args.devices > 1 else None),
        max_epochs=args.max_epochs,
        logger=loggers,
    )

    trainer.validate(global_workspace, data)
    trainer.test(global_workspace, data)
