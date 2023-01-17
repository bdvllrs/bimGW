import os
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from bim_gw.modules.workspace_module import PassThroughWM
from bim_gw.utils.domains import DomainRegistry
from bim_gw.utils.loggers import get_loggers


def get_domain(domain_name, args, img_size=None):
    try:
        domain = DomainRegistry().get(domain_name)(args, img_size)
    except KeyError:
        raise ValueError(f"Domain {domain_name} not found in registry.")

    if args.global_workspace.use_pre_saved and domain_name in args.global_workspace.load_pre_saved_latents.keys():
        domain = PassThroughWM(domain)

    domain.eval()
    domain.freeze()
    return domain


def get_domains(args, img_size=None):
    return {
        domain: get_domain(domain, args, img_size) for domain in args.global_workspace.selected_domains
    }


def get_trainer(name, args, model, monitor_loss="val_total_loss", early_stopping_patience=None, trainer_args=None):
    slurm_job_id = os.getenv("SLURM_JOBID", None)

    tags = None
    version = None
    if slurm_job_id is not None:
        tags = ["slurm", slurm_job_id]
        version = "-".join(tags)
    source_files = ['../**/*.py', '../readme.md',
                    '../requirements.txt', '../**/*.yaml']
    loggers = get_loggers(name, version, args.loggers, model, args, tags, source_files)

    # Callbacks
    callbacks = []
    if len(loggers):
        callbacks.append(
            LearningRateMonitor(logging_interval="epoch")
        )
        if early_stopping_patience is not None:
            callbacks.append(EarlyStopping(monitor=monitor_loss, patience=early_stopping_patience))
    if len(loggers) and args.checkpoints_dir is not None:
        logger = loggers[0]
        if slurm_job_id is not None:
            save_dir = Path(args.checkpoints_dir) / "checkpoints"
        else:
            save_dir = Path(args.checkpoints_dir) / str(logger.name) / str(logger.version) / "checkpoints"
        callbacks.append(ModelCheckpoint(dirpath=save_dir, save_top_k=1, mode="min", monitor=monitor_loss))

    _trainer_args = {
        "default_root_dir": args.checkpoints_dir,
        "fast_dev_run": args.fast_dev_run,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "strategy": (args.distributed_backend if args.devices > 1 else None),
        "logger": loggers,
        "callbacks": callbacks,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "max_epochs": args.max_epochs,
        "max_steps": args.max_steps,
        "multiple_trainloader_mode": "min_size",
    }
    if trainer_args is not None:
        _trainer_args.update(trainer_args)

    return Trainer(**_trainer_args)
