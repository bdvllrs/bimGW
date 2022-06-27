import os

import torch
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.datasets.simple_shapes.data_modules import SimpleShapesDataModule
from bim_gw.modules import VAE, AE, GlobalWorkspace, ShapesLM
from bim_gw.scripts.utils import get_domains, get_trainer


def train_gw(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    global_workspace = GlobalWorkspace(get_domains(args, data), args.global_workspace.z_size,
                                       args.global_workspace.hidden_size,
                                       args.global_workspace.n_layers.encoder,
                                       args.global_workspace.n_layers.decoder,
                                       args.global_workspace.n_layers.decoder_head,
                                       len(data.classes),
                                       args.losses.coefs.demi_cycles, args.losses.coefs.cycles,
                                       args.losses.coefs.supervision, args.losses.coefs.cosine,
                                       args.losses.coefs.contrastive,
                                       args.global_workspace.optim.lr, args.global_workspace.optim.weight_decay,
                                       args.global_workspace.scheduler.mode, args.global_workspace.scheduler.interval,
                                       args.global_workspace.scheduler.step, args.global_workspace.scheduler.gamma,
                                       args.losses.schedules, data.domain_examples,
                                       args.global_workspace.monitor_grad_norms)

    trainer = get_trainer("train_gw", args, global_workspace, monitor_loss="val/in_dist/total_loss", trainer_args={
        # "val_check_interval": args.global_workspace.prop_labelled_images
    })
    trainer.fit(global_workspace, data)

    for logger in trainer.loggers:
        logger.save_images(True)
    trainer.validate(global_workspace, data)


def train_lm(args):
    seed_everything(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "1"

    args.lm.prop_labelled_images = 1.

    args.lm.split_ood = False
    args.lm.selected_domains = {"a": "attr", "t": "t"}
    args.lm.data_augmentation = False

    data = load_dataset(args, args.lm)
    data.prepare_data()
    data.setup(stage="fit")

    domain_examples = {d: data.domain_examples["in_dist"][0][d][1:] for d in data.domain_examples["in_dist"][0].keys()}

    if "checkpoint" in args:
        lm = ShapesLM.load_from_checkpoint(args.checkpoint, strict=False,
                                           bert_path=args.global_workspace.bert_path,
                                           validation_domain_examples=domain_examples)
    else:
        lm = ShapesLM(args.lm.z_size, len(data.classes), data.img_size, args.global_workspace.bert_path,
                      args.lm.optim.lr, args.lm.optim.weight_decay, args.lm.scheduler.step, args.lm.scheduler.gamma,
                      domain_examples)

    trainer = get_trainer("train_lm", args, lm, monitor_loss="val/total_loss")
    trainer.fit(lm, data)
    trainer.validate(lm, data)


def train_ae(args):
    seed_everything(args.seed)

    args.vae.prop_sync_domains = {"all": 1.}
    args.vae.split_ood = False
    args.vae.selected_domains = {"v": "v"}

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")

    ae = AE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size,
        args.n_validation_examples,
        args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
        data.domain_examples["in_dist"][0]["v"][1]
    )

    trainer = get_trainer("train_lm", args, ae, monitor_loss="val_total_loss")
    trainer.fit(ae, data)
    trainer.validate(ae, data)


def train_vae(args):
    seed_everything(args.seed)

    args.vae.prop_labelled_images = 1.

    args.vae.split_ood = False
    args.vae.selected_domains = {"v": "v"}

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")
    data.compute_inception_statistics(32, torch.device("cuda" if args.accelerator == "gpu" else "cpu"))

    if "checkpoint" in args:
        vae = VAE.load_from_checkpoint(args.checkpoint, strict=False,
                                       n_validation_examples=args.n_validation_examples,
                                       validation_reconstruction_images=data.domain_examples["in_dist"][0]["v"][1])
    else:
        vae = VAE(
            data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size, args.vae.beta, args.vae.type,
            args.n_validation_examples,
            args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
            data.domain_examples["in_dist"][0]["v"][1], args.vae.n_FID_samples
        )

    trainer = get_trainer("train_vae", args, vae, monitor_loss="val_total_loss")
    trainer.fit(vae, data)
    # vae.n_FID_samples = data.val_dataset_size  # all the dataset
    trainer.validate(vae, data)
