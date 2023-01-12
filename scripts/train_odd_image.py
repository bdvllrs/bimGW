import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from torch import nn

from bim_gw.modules.gw import GlobalWorkspace
from pytorch_lightning import Trainer

from bim_gw.datasets.odd_image.data_module import OddImageDataModule
from bim_gw.modules.odd_classifier import OddClassifier
from bim_gw.modules.utils import DomainEncoder
from bim_gw.scripts.utils import get_domains
from bim_gw.utils import get_args
from bim_gw.utils.loggers import get_loggers
from bim_gw.utils.visualization import update_df_for_legacy_code


def get_name(x):
    name = ""
    if x['parameters/losses/coefs/supervision'] > 0:
        name += "+sup"
    if x['parameters/losses/coefs/demi_cycles'] > 0:
        name += "+dcy"
    if x['parameters/losses/coefs/cycles'] > 0:
        name += "+cy"
    return name


def get_csv_data(df, args, csv_row=None):
    if csv_row is None:
        df = df.loc[df['parameters/losses/coefs/contrastive'] == args.losses.coefs.contrastive]
        df = df.loc[df['parameters/losses/coefs/cycles'] == args.losses.coefs.cycles]
        df = df.loc[df['parameters/losses/coefs/demi_cycles'] == args.losses.coefs.demi_cycles]
        df = df.loc[df['parameters/losses/coefs/translation'] == args.losses.coefs.translation]
        df = df.loc[df['parameters/global_workspace/prop_labelled_images'] == args.global_workspace.prop_labelled_images]
        item = df.iloc[0].to_dict()
    else:
        item = df.iloc[csv_row].to_dict()

    args.losses.coefs.demi_cycles = item['parameters/losses/coefs/demi_cycles']
    args.losses.coefs.cycles = item['parameters/losses/coefs/cycles']
    args.losses.coefs.contrastive = item['parameters/losses/coefs/contrastive']
    args.losses.coefs.translation = item['parameters/losses/coefs/translation']
    args.global_workspace.prop_labelled_images = item['parameters/global_workspace/prop_labelled_images']
    args.global_workspace.selected_domains = item['parameters/global_workspace/selected_domains']
    if 'parameters/seed' in item:
        args.seed = item['parameters/seed']
    if 'Name' in item:
        item['name'] = item['Name']
    return item


def find_best_epoch(ckpt_folder):
    ckpt_folder = Path(ckpt_folder)
    files = [(str(p), int(str(p).split('/')[-1].split('-')[0][6:])) for p in ckpt_folder.iterdir()]
    return sorted(files, key=lambda x: x[0], reverse=True)[0][0]
    # epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    # return max(epochs)

class IdentityModule(nn.Module):
    def forward(self, x):
        return x[0]

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))

    if args.odd_image.csv_ids is not None:
        df = update_df_for_legacy_code(pd.read_csv(args.odd_image.csv_ids))
        item = get_csv_data(df, args, args.odd_image.csv_row)
        args.odd_image.slurm_id = item['name'].split("-")[1]

    load_domains = []

    if args.odd_image.encoder_path is None or args.odd_image.encoder_path == "random":
        encoder = nn.Sequential(DomainEncoder(args.vae.z_size, args.global_workspace.hidden_size, args.global_workspace.z_size,
                                args.global_workspace.n_layers.encoder), nn.Tanh())
        if args.odd_image.encoder_path == "random":
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
        load_domains = ["v"]
        encoders = {name: encoder for name in load_domains}
    elif args.odd_image.encoder_path == "identity":
        encoder = IdentityModule()
        load_domains = ["v"]
        encoders = {name: encoder for name in load_domains}
    else:
        path = args.odd_image.encoder_path
        if not os.path.isfile(path) and os.path.isdir(path):
            path = find_best_epoch(path)
        global_workspace = GlobalWorkspace.load_from_checkpoint(path,
                                                                domain_mods=get_domains(args, 32), strict=False)
        load_domains = global_workspace.domain_names
        global_workspace.freeze()
        global_workspace.eval()
        encoders = {name: nn.Sequential(global_workspace.encoders[name], nn.Tanh()) for name in load_domains}

    args.global_workspace.selected_domains = OmegaConf.create([name for name in load_domains])

    if args.odd_image.resume_csv is not None:
        item = get_csv_data(pd.read_csv(args.odd_image.resume_csv), args)
        args.odd_image.slurm_id = item['Name'].split("-")[1]
        path = args.odd_image.encoder_path
        if not os.path.isfile(path) and os.path.isdir(path):
            path = find_best_epoch(path)
        model = OddClassifier.load_from_checkpoint(path,
                                                   unimodal_encoders=get_domains(args, 32),
                                                   encoders=encoders)
        for logger in args.loggers:
            logger.args.version = item['ID']
            logger.args.id = item['ID']
            logger.args.resume = True
    else:
        model = OddClassifier(get_domains(args, 32), encoders, args.global_workspace.z_size,
                              args.odd_image.optimizer.lr, args.odd_image.optimizer.weight_decay)

    data = OddImageDataModule(args.simple_shapes_path, args.global_workspace.load_pre_saved_latents,
                              args.odd_image.batch_size, args.dataloader.num_workers,
                              args.global_workspace.selected_domains, args.fetchers.t.bert_latents)

    if 'attr' in model.unimodal_encoders.keys():
        model.unimodal_encoders['attr'].output_dims = [len(data.classes), data.img_size]

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
