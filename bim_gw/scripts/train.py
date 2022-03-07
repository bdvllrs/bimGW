import os
from copy import deepcopy

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from bim_gw.datasets import load_dataset
from bim_gw.datasets.simple_shapes import SimpleShapesData
from bim_gw.datasets.utils import get_lm
from bim_gw.loggers.neptune import NeptuneLogger
from bim_gw.modules import VAE, AE, GlobalWorkspace, ActionModule, ShapesLM


def train_lm(args):
    seed_everything(args.seed)

    data = SimpleShapesData(args.simple_shapes_path, args.lm.batch_size, args.dataloader.num_workers, False, 1.,
                            args.lm.n_validation_examples, False, {"a": "attr", "t": "t"})
    data.prepare_data()
    data.setup(stage="fit")

    lm = ShapesLM(args.lm.z_size, len(data.classes), data.img_size, args.global_workspace.bert_path,
                  args.lm.optim.lr, args.lm.optim.weight_decay, args.lm.scheduler.step, args.lm.scheduler.gamma,
                  data.validation_domain_examples["in_dist"])

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
        fast_dev_run=args.fast_dev_run,
        gpus=args.gpus, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs, log_every_n_steps=1
    )

    trainer.fit(lm, data)


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
        fast_dev_run=args.fast_dev_run,
        gpus=args.gpus, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs, log_every_n_steps=1
    )

    trainer.fit(ae, data)


def train_vae(args):
    seed_everything(args.seed)

    args.vae.prop_labelled_images = 1.
    args.vae.split_ood = False
    args.vae.selected_domains = {"v": "v"}

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")
    data.compute_inception_statistics(32, torch.device("cuda" if args.gpus >= 1 else "cpu"))

    vae = VAE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size, args.vae.beta, args.vae.type,
        args.n_validation_examples,
        args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
        data.validation_domain_examples["in_dist"]["v"], args.vae.n_FID_samples
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
            name="train_vae",
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
        fast_dev_run=args.fast_dev_run,
        gpus=args.gpus, logger=logger, callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs, log_every_n_steps=1
    )

    trainer.fit(vae, data)

    vae.n_FID_samples = data.val_dataset_size  # all the dataset
    trainer.validate(data.val_dataloader())


def train_gw(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    vae = VAE.load_from_checkpoint(
        args.global_workspace.vae_checkpoint,
        mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
        kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
    ).eval()
    vae.freeze()

    lm = get_lm(args, data).eval()
    lm.freeze()

    def get_domain_model(name):
        if "v" in name:
            return deepcopy(vae)
        elif "t" in name:
            return deepcopy(lm)
        elif name == "a":
            return ActionModule(len(data.classes), data.img_size)

    global_workspace = GlobalWorkspace({
        name: get_domain_model(name) for name in args.global_workspace.selected_domains.keys()
    }, args.global_workspace.z_size, args.global_workspace.hidden_size, len(data.classes),
        args.losses.coefs.demi_cycles,
        args.losses.coefs.cycles, args.losses.coefs.supervision,
        args.global_workspace.optim.lr, args.global_workspace.optim.weight_decay,
        args.global_workspace.scheduler.step, args.global_workspace.scheduler.gamma,
        data.validation_domain_examples,
        args.global_workspace.monitor_grad_norms
    )

    logger = None
    if args.neptune.project_name is not None:
        slurm_job_id = os.getenv("SLURM_JOBID", None)
        tags = None
        if slurm_job_id is not None:
            tags = ["calmip", slurm_job_id]

        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project=args.neptune.project_name,
            name="train_gw",
            run=args.neptune.resume,
            mode=args.neptune.mode,
            tags=tags,
            source_files=['../**/*.py', '../readme.md',
                          '../requirements.txt', '../**/*.yaml']
        )

        logger.experiment["parameters"] = dict(args)

    # Callbacks
    callbacks = []
    # callbacks = [
    #     EarlyStopping("val_total_loss", patience=6),
    # ]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        callbacks.append(ModelCheckpoint(save_top_k=1, mode="min", monitor="val_in_dist_total_loss"))

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        # fast_dev_run=True,
        gpus=args.gpus, logger=logger,
        callbacks=callbacks,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs,
        # val_check_interval=0.25,
        multiple_trainloader_mode="min_size",
    )

    trainer.fit(global_workspace, data)
