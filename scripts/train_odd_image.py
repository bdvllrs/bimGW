import os

from bim_gw.modules.gw import GlobalWorkspace
from pytorch_lightning import Trainer

from bim_gw.datasets.odd_image.data_module import OddImageDataModule
from bim_gw.modules.odd_classifier import OddClassifier
from bim_gw.modules.utils import DomainEncoder
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    data = OddImageDataModule(args.simple_shapes_path, args.global_workspace.load_pre_saved_latents,
                              args.odd_image.batch_size, args.dataloader.num_workers)

    if args.odd_image.encoder_path is None:
        encoder = DomainEncoder(args.vae.z_size, args.global_workspace.hidden_size, args.global_workspace.z_size,
                                args.global_workspace.n_layers.encoder)
    else:
        global_workspace = GlobalWorkspace.load_from_checkpoint(args.odd_image.encoder_path,
                                                                domain_mods=get_domains(args, data), strict=False)
        global_workspace.freeze()
        global_workspace.eval()
        encoder = global_workspace.encoders["v"]

    model = OddClassifier(encoder, args.global_workspace.z_size,
                args.odd_image.optimizer.lr, args.odd_image.optimizer.weight_decay)

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
        fast_dev_run=True,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=(args.distributed_backend if args.devices > 1 else None),
        logger=loggers,
    )

    trainer.fit(model, data)
