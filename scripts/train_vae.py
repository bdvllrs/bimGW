import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from bim_gw.loggers.neptune import NeptuneLogger

from bim_gw.datasets import ImageNetData
from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args


def main(args):
    debug_mode = int(os.getenv("DEBUG", 0))
    seed_everything(args.seed)

    data = ImageNetData(args.image_net_path, args.batch_size, args.img_size, args.dataloader.num_workers)

    vae = VAE(
        data.img_size, data.num_channels, args.kernel_num, args.z_size,
        args.optim.lr, args.optim.weight_decay
    )

    logger = None
    if not debug_mode:
        logger = NeptuneLogger(
            api_key=args.neptune.api_token,
            project_name=args.neptune.project_name,
            experiment_name="train_vae",
            params=dict(args),
            upload_source_files=['../bim_gw/**/*.py', '../scripts/**/.py',
                                 '../*.py', '../readme.md',
                                 '../requirements.txt', '../**/*.yaml']
        )

    model_checkpoints = ModelCheckpoint(args.checkpoints_dir, save_top_k=-1, mode="min", monitor="val_total_loss")
    trainer = Trainer(
        fast_dev_run=True,
        gpus=args.gpus, logger=logger,
        checkpoint_callback=model_checkpoints,
        resume_from_checkpoint=args.resume_from_checkpoint,
        distributed_backend=(args.distributed_backend if args.gpus > 1 else None),
        max_epochs=args.max_epochs
    )

    trainer.fit(vae, data)


if __name__ == '__main__':
    main(get_args())
