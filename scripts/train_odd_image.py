import os
from pathlib import Path

import pandas as pd
from torch import nn

from bim_gw.modules.gw import GlobalWorkspace
from pytorch_lightning import Trainer

from bim_gw.datasets.odd_image.data_module import OddImageDataModule
from bim_gw.modules.odd_classifier import OddClassifier
from bim_gw.modules.utils import DomainEncoder
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers


def get_name(x):
    name = ""
    if x['parameters/losses/coefs/supervision'] > 0:
        name += "+sup"
    if x['parameters/losses/coefs/demi_cycles'] > 0:
        name += "+dcy"
    if x['parameters/losses/coefs/cycles'] > 0:
        name += "+cy"
    return name

def get_csv_data(df, args):
    df = df.loc[df['parameters/losses/coefs/contrastive'] == args.losses.coefs.contrastive]
    df = df.loc[df['parameters/losses/coefs/cycles'] == args.losses.coefs.cycles]
    df = df.loc[df['parameters/losses/coefs/demi_cycles'] == args.losses.coefs.demi_cycles]
    df = df.loc[df['parameters/losses/coefs/supervision'] == args.losses.coefs.supervision]
    df = df.loc[df['parameters/global_workspace/prop_labelled_images'] == args.global_workspace.prop_labelled_images]
    item = df.iloc[0].to_dict()
    return item
    # df['slug'] = df.apply(get_name, axis=1)
    # min_idx = df.groupby(["parameters/global_workspace/prop_labelled_images", 'slug'])["min"].idxmin()
    # df = df.loc[min_idx]
    # for index, row in df.iterrows():
    #     slurm_id = row["name"].split("-")[1]
    #     prop_labelled_images = row["parameters/global_workspace/prop_labelled_images"]
    #     contrastive_coef = row['parameters/losses/coefs/contrastive']
    #     cycles_coef = row['parameters/losses/coefs/cycles']
    #     demi_cycles_coef = row['parameters/losses/coefs/demi_cycles']
    #     supervision_coef = row['parameters/losses/coefs/supervision']


def find_best_epoch(ckpt_folder):
    ckpt_folder = Path(ckpt_folder)
    files = [(str(p), int(str(p).split('/')[-1].split('-')[0][6:])) for p in ckpt_folder.iterdir()]
    return sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    # epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    # return max(epochs)

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    item = get_csv_data(pd.read_csv(args.odd_image.csv_ids), args)
    args.odd_image.slurm_id = item['name'].split("-")[1]

    data = OddImageDataModule(args.simple_shapes_path, args.global_workspace.load_pre_saved_latents,
                              args.odd_image.batch_size, args.dataloader.num_workers)

    if args.odd_image.encoder_path is None or args.odd_image.encoder_path == "random":
        encoder = nn.Sequential(DomainEncoder(args.vae.z_size, args.global_workspace.hidden_size, args.global_workspace.z_size,
                                args.global_workspace.n_layers.encoder), nn.Tanh())
        if args.odd_image.encoder_path == "random":
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
    elif args.odd_image.encoder_path == "identity":
        encoder = lambda x: x[0]
    else:
        path = find_best_epoch(args.odd_image.encoder_path)
        global_workspace = GlobalWorkspace.load_from_checkpoint(path,
                                                                domain_mods=get_domains(args, data), strict=False)
        global_workspace.freeze()
        global_workspace.eval()
        encoder = nn.Sequential(global_workspace.encoders["v"], nn.Tanh())

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
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=(args.distributed_backend if args.devices > 1 else None),
        max_epochs=args.max_epochs,
        logger=loggers,
    )

    trainer.fit(model, data)
