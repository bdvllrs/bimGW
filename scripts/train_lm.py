import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from bim_gw.datasets.simple_shapes_lm_attributes import SimpleShapesData
from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules.language_model import ShapesLM
from bim_gw.utils import get_args


def train_lm(args):
    seed_everything(args.seed)

    data = SimpleShapesData(args.simple_shapes_path, args.lm.batch_size, args.dataloader.num_workers,
                            args.lm.n_validation_examples)
    data.prepare_data()
    data.setup(stage="fit")

    lm = ShapesLM(args.lm.z_size, len(data.classes), data.img_size, args.global_workspace.bert_path,
                  args.lm.optim.lr, args.lm.optim.weight_decay, args.lm.scheduler.step, args.lm.scheduler.gamma,
                  data.validation_domain_examples)

    logger = None
    if args.neptune.project_name is not None:
        slurm_job_id = os.getenv("SLURM_JOBID", None)
        tags = None
        if slurm_job_id is not None:
            tags = ["calmip", slurm_job_id]

        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="train_lm",
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
        max_epochs=args.max_epochs, log_every_n_steps=1
    )

    trainer.fit(lm, data)


if __name__ == "__main__":
    train_lm(get_args(debug=int(os.getenv("DEBUG", 0))))
