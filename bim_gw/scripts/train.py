import torch
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesText
from bim_gw.utils.scripts import get_domains, get_trainer
from bim_gw.utils.utils import get_checkpoint_path, loggers_save_images


def train_gw(args, mode="train"):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    if "checkpoint" in args and args.checkpoint is not None:
        checkpoint_path = get_checkpoint_path(args.checkpoint)
        global_workspace = GlobalWorkspace.load_from_checkpoint(
            checkpoint_path,
            domain_mods=get_domains(args, data.img_size),
            domain_examples=data.domain_examples, )
    else:
        global_workspace = GlobalWorkspace(
            get_domains(args, data.img_size), args.global_workspace.z_size,
            args.global_workspace.hidden_size,
            args.global_workspace.n_layers.encoder,
            args.global_workspace.n_layers.decoder,
            args.global_workspace.n_layers.decoder_head, len(data.classes),
            args.losses.coefs.demi_cycles, args.losses.coefs.cycles,
            args.losses.coefs.translation,
            args.losses.coefs.cosine, args.losses.coefs.contrastive,
            args.global_workspace.optim.lr,
            args.global_workspace.optim.weight_decay,
            args.global_workspace.scheduler.mode,
            args.global_workspace.scheduler.interval,
            args.global_workspace.scheduler.step,
            args.global_workspace.scheduler.gamma, args.losses.schedules,
            data.domain_examples,
            args.global_workspace.monitor_grad_norms,
            args.global_workspace.remove_sync_domains
        )

    trainer = get_trainer(
        "train_gw", args, global_workspace,
        monitor_loss="val/in_dist/total_loss",
        early_stopping_patience=args.global_workspace.early_stopping_patience,
        trainer_args={
            # "val_check_interval": args.global_workspace.prop_labelled_images
        }
    )

    best_checkpoint = None
    if mode == "train":
        trainer.fit(global_workspace, data)
        best_checkpoint = "best" if not args.fast_dev_run else None

    loggers_save_images(trainer.loggers, True)
    if mode in ["train", "eval"]:
        trainer.validate(global_workspace, data, best_checkpoint)
    if mode in ["train", "test"]:
        trainer.test(global_workspace, data, best_checkpoint)


def train_lm(args):
    seed_everything(args.seed)

    data = load_dataset(
        args, args.lm, add_unimodal=False, selected_domains=["attr", "t"]
    )
    data.prepare_data()
    data.setup(stage="fit")

    if "checkpoint" in args and args.checkpoint is not None:
        checkpoint_path = get_checkpoint_path(args.checkpoint)
        lm = SimpleShapesText.load_from_checkpoint(
            checkpoint_path, strict=False,
            bert_path=args.global_workspace.bert_path,
            domain_examples=data.domain_examples,
            train_vae=args.lm.train_vae,
            train_attr_decoders=args.lm.train_attr_decoders,
            optimize_vae_with_attr_regression=args.lm
            .optimize_vae_with_attr_regression,
            ceof_attr_loss=args.lm.coef_attr_loss,
            ceof_vae_loss=args.lm.coef_vae_loss,
        )
    else:
        lm = SimpleShapesText(
            args.lm.z_size, args.lm.hidden_size, args.lm.beta,
            len(data.classes), data.img_size, args.global_workspace.bert_path,
            args.lm.optim.lr, args.lm.optim.weight_decay,
            args.lm.scheduler.step,
            args.lm.scheduler.gamma, data.domain_examples,
            args.lm.train_vae, args.lm.train_attr_decoders,
            args.lm.optimize_vae_with_attr_regression, args.lm.coef_attr_loss,
            args.lm.coef_vae_loss,
        )

    trainer = get_trainer(
        "train_lm", args, lm, monitor_loss="val/total_loss",
        early_stopping_patience=args.lm.early_stopping_patience
    )
    trainer.fit(lm, data)
    best_checkpoint = "best" if not args.fast_dev_run else None

    loggers_save_images(trainer.loggers, True)
    trainer.validate(lm, data, best_checkpoint)
    trainer.test(lm, data, best_checkpoint)


def train_vae(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.vae, selected_domains=["v"])

    data.prepare_data()
    data.setup(stage="fit")
    data.compute_inception_statistics(
        32, torch.device("cuda" if args.accelerator == "gpu" else "cpu")
    )

    if "checkpoint" in args and args.checkpoint is not None:
        checkpoint_path = get_checkpoint_path(args.checkpoint)
        validation_images = data.domain_examples["val"][0]["v"][1]
        vae = VAE.load_from_checkpoint(
            checkpoint_path, strict=False,
            n_validation_examples=args.n_validation_examples,
            validation_reconstruction_images=validation_images,
        )
    else:
        vae = VAE(
            data.img_size, data.num_channels, args.vae.ae_size,
            args.vae.z_size, args.vae.beta, args.vae.type,
            args.n_validation_examples, args.vae.optim.lr,
            args.vae.optim.weight_decay, args.vae.scheduler.step,
            args.vae.scheduler.gamma, data.domain_examples["val"][0]["v"][1],
            args.vae.n_fid_samples
        )

    trainer = get_trainer(
        "train_vae", args, vae, monitor_loss="val_total_loss",
        early_stopping_patience=args.vae.early_stopping_patience
    )
    trainer.fit(vae, data)
    # vae.n_FID_samples = data.val_dataset_size  # all the dataset
    best_checkpoint = "best" if not args.fast_dev_run else None
    loggers_save_images(trainer.loggers, True)
    trainer.validate(vae, data, best_checkpoint)
    trainer.test(vae, data, best_checkpoint)
