import os
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from bim_gw.modules import VAE, ShapesLM, ActionModule
from bim_gw.modules.language_model import ShapesAttributesLM
from bim_gw.modules.workspace_module import PassThroughWM
from bim_gw.utils.loggers import get_loggers


def get_domain(name, domain_name, args, data):
    if domain_name in ["v", "v_f"]:
        domain = VAE.load_from_checkpoint(
            args.global_workspace.vae_checkpoint,
            mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
            kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
        )
    elif domain_name in ["t", "t_f"]:
        domain = ShapesLM.load_from_checkpoint(
            args.global_workspace.lm_checkpoint,
            bert_path=args.global_workspace.bert_path)
    elif domain_name in ["attr", "attr_f"]:
        domain = ShapesAttributesLM(len(data.classes), data.img_size)
    elif domain_name == "a":
        domain = ActionModule()
    else:
        raise ValueError(f"{domain_name} is not a valid domain name.")

    if args.global_workspace.use_pre_saved and domain_name in args.global_workspace.load_pre_saved_latents.keys():
        domain = PassThroughWM(domain)

    domain.eval()
    domain.freeze()
    return domain


def get_domains(args, data):
    return {
        name: get_domain(name, domain, args, data) for name, domain in args.global_workspace.selected_domains.items()
    }


def get_trainer(name, args, model, monitor_loss="val_total_loss", trainer_args=None):
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
        callbacks.append(EarlyStopping(monitor=monitor_loss, patience=5))
    if len(loggers) and args.checkpoints_dir is not None:
        logger = loggers[0]
        if slurm_job_id is not None:
            save_dir = Path(args.checkpoints_dir) / "checkpoints"
        else:
            save_dir = Path(args.checkpoints_dir) / str(logger.name) / str(logger.version) / "checkpoints"
        callbacks.append(ModelCheckpoint(dirpath=save_dir, save_top_k=2, mode="min", monitor=monitor_loss))

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
        "multiple_trainloader_mode": "min_size",
    }
    if trainer_args is not None:
        _trainer_args.update(trainer_args)

    return Trainer(**_trainer_args)
