import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules.language_model import LanguageModel
from bim_gw.utils import get_args


def train_lm(args):
    seed_everything(args.seed)

    lm = LanguageModel(
        args.gensim_model_path, 1000, 300, 128,
        args.optim.lr, args.optim.weight_decay, args.scheduler.step, args.scheduler.gamma,
    )

    logger = None
    if args.neptune.project_name is not None:
        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project_name=args.neptune.project_name,
            experiment_name="train_vae",
            mode=args.neptune.mode,
            params=dict(args),
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

    # Callbacks
    model_checkpoints = ModelCheckpoint(save_top_k=-1, mode="min", monitor="val_total_loss")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        # fast_dev_run=True,
        gpus=args.gpus, logger=logger,
        callbacks=[model_checkpoints, lr_monitor],
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs
    )

    trainer.fit(lm)


if __name__ == "__main__":
    train_lm(get_args(debug=int(os.getenv("DEBUG", 0))))
