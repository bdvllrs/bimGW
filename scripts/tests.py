import os

from pytorch_lightning import seed_everything, Trainer

from bim_gw.datasets import load_dataset
from bim_gw.datasets.utils import get_lm
from bim_gw.modules.domain_modules.vae import VAE
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.utils import get_args
from bim_gw.utils.loggers.neptune import NeptuneLogger


def tests(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace, bimodal=True)
    data.prepare_data()
    data.setup(stage="fit")

    vae = VAE.load_from_checkpoint(
        args.global_workspace.vae_checkpoint,
        mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
        kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
    ).eval()
    vae.freeze()

    lm = get_lm(args, data)
    lm.freeze()

    global_workspace = GlobalWorkspace.load_from_checkpoint(
        args.checkpoint, domain_mods={
            "v": vae,
            "t": lm
        }
    )
    global_workspace.eval()

    logger = None
    if args.neptune.project_name is not None:
        slurm_job_id = os.getenv("SLURM_JOBID", None)
        tags = None
        if slurm_job_id is not None:
            tags = ["calmip", slurm_job_id]

        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="test",
            run=args.neptune.resume,
            mode=args.neptune.mode,
            tags=tags,
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

        logger.experiment["parameters"] = dict(args)

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        gpus=args.gpus, logger=logger,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs, log_every_n_steps=1
    )

    trainer.validate(global_workspace, data.val_dataloader())


if __name__ == "__main__":
    tests(get_args(debug=int(os.getenv("DEBUG", 0))))
