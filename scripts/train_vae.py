import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from bim_gw.datasets import ImageNetData
from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args


def train_vae(args):
    seed_everything(args.seed)

    data = ImageNetData(args.image_net_path, args.batch_size, args.img_size,
                        args.dataloader.num_workers, args.data_augmentation)

    vae = VAE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size, args.vae.beta,
        args.n_validation_examples,
        args.optim.lr, args.optim.weight_decay, args.scheduler.step, args.scheduler.gamma,
        data.validation_reconstructed_images
    )

    logger = None
    if args.neptune.project_name is not None:
        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="train_vae",
            run=args.neptune.resume,
            mode=args.neptune.mode,
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

        logger.experiment["parameters"] = dict(args)

    # Callbacks
    callbacks = [ModelCheckpoint(save_top_k=-1, mode="min", monitor="val_total_loss")]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        # fast_dev_run=True,
        gpus=args.gpus, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs
    )

    trainer.fit(vae, data)


if __name__ == "__main__":
    train_vae(get_args(debug=int(os.getenv("DEBUG", 0))))
