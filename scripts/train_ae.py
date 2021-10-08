import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from bim_gw.datasets import load_dataset
from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules.ae import AE
from bim_gw.utils import get_args


def train_ae(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")

    ae = AE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size,
        args.n_validation_examples,
        args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
        data.validation_domain_examples["v"]
    )

    # checkpoint = torch.load(args.resume_from_checkpoint)
    # import matplotlib.pyplot as plt

    logger = None
    if args.neptune.project_name is not None:
        slurm_job_id = os.getenv("SLURM_JOBID", None)
        tags = None
        if slurm_job_id is not None:
            tags = ["calmip", slurm_job_id]

        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="train_ae",
            run=args.neptune.resume,
            mode=args.neptune.mode,
            tags=tags,
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

        logger.experiment["parameters"] = dict(args)

    # Callbacks
    callbacks = [ModelCheckpoint(save_top_k=2, mode="min", monitor="val_total_loss")]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        # fast_dev_run=True,
        gpus=args.gpus, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs,log_every_n_steps=1
    )

    trainer.fit(ae, data)


if __name__ == "__main__":
    train_ae(get_args(debug=int(os.getenv("DEBUG", 0))))