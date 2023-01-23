import os

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.unpaired_classifier import UnpairedClassifier
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    args.global_workspace.selected_domains = OmegaConf.create(["v", "t"])

    data = load_dataset(args, args.global_workspace, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    domain_mods = get_domains(args, data.img_size)
    global_workspace = GlobalWorkspace.load_from_checkpoint(args.checkpoint, domain_mods=domain_mods, strict=False)

    model = UnpairedClassifier(global_workspace)

    slurm_job_id = os.getenv("SLURM_JOBID", None)

    tags = None
    version = None
    if slurm_job_id is not None:
        tags = ["slurm", slurm_job_id]
        version = "-".join(tags)
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

    trainer.fit(model, data)
