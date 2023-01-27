import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.domain_modules.simple_shapes.downstream import UnpairedClassifierAttributes
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers
from bim_gw.utils.utils import get_checkpoint_path

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    args.global_workspace.selected_domains = OmegaConf.create(["attr"])

    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    assert args.downstream.unpaired_cls.checkpoint is not None, "You must provide a checkpoint for this script."
    checkpoint_path = get_checkpoint_path(args.downstream.unpaired_cls.checkpoint)
    domain_mods = get_domains(args, data.img_size)
    global_workspace = GlobalWorkspace.load_from_checkpoint(checkpoint_path, domain_mods=domain_mods, strict=False)

    args.losses.coefs = OmegaConf.create({
        "translation": global_workspace.hparams['loss_coef_translation'],
        "cycles": global_workspace.hparams['loss_coef_cycles'],
        "demi_cycles": global_workspace.hparams['loss_coef_demi_cycles'],
        "contrastive": global_workspace.hparams['loss_coef_contrastive'],
    })

    model = UnpairedClassifierAttributes(
        global_workspace, args.downstream.unpaired_cls.optimizer.lr,
        args.downstream.unpaired_cls.optimizer.weight_decay,
    )

    slurm_job_id = os.getenv("SLURM_JOBID", None)

    tags = None
    version = args.run_name
    if slurm_job_id is not None:
        tags = ["slurm"]
    source_files = ['../**/*.py', '../readme.md',
                    '../requirements.txt', '../**/*.yaml']
    loggers = get_loggers("train_odd_image", version, args.loggers, model, args, tags, source_files)

    trainer = Trainer(
        default_root_dir=args.checkpoints_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=(args.distributed_backend if args.devices > 1 else None),
        max_epochs=args.max_epochs,
        logger=loggers,
    )

    if not args.downstream.unpaired_cls.random_regressor:
        trainer.fit(model, data)

    trainer.validate(global_workspace, data, "best" if not args.downstream.unpaired_cls.random_regressor else None)
    trainer.test(global_workspace, data, "best" if not args.downstream.unpaired_cls.random_regressor else None)
