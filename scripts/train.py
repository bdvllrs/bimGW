import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from bim_gw.datasets import load_dataset
from bim_gw.datasets.utils import get_lm
from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args


def train_gw(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace, bimodal=True)
    data.prepare_data()
    data.setup(stage="fit")

    vae = VAE.load_from_checkpoint(args.global_workspace.vae_checkpoint).eval()
    vae.freeze()

    lm = get_lm(args, data)
    lm.freeze()

    global_workspace = GlobalWorkspace({
        "v": vae,
        "t": lm
    }, args.global_workspace.z_size, args.global_workspace.hidden_size, len(data.classes),
        args.losses.coefs.demi_cycles,
        args.losses.coefs.cycles, args.losses.coefs.supervision,
        args.global_workspace.cycle_loss_fn, args.global_workspace.supervision_loss_fn,
        args.global_workspace.optim.lr, args.global_workspace.optim.weight_decay,
        args.global_workspace.scheduler.step, args.global_workspace.scheduler.gamma,
        data.validation_domain_examples,
        args.global_workspace.monitor_grad_norms
    )

    logger = None
    if args.neptune.project_name is not None:
        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="train_gw",
            run=args.neptune.resume,
            mode=args.neptune.mode,
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

        logger.experiment["parameters"] = dict(args)

    # Callbacks
    # callbacks = [ModelCheckpoint(save_top_k=1, mode="min", monitor="val_total_loss")]
    callbacks = []
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        # fast_dev_run=True,
        gpus=args.gpus, logger=logger,
        callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs,
        multiple_trainloader_mode="max_size_cycle",
    )

    trainer.fit(global_workspace, data)


if __name__ == "__main__":
    train_gw(get_args(debug=int(os.getenv("DEBUG", 0))))
