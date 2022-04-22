import torch
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.datasets.simple_shapes import SimpleShapesData
from bim_gw.modules import VAE, AE, GlobalWorkspace, ShapesLM
from bim_gw.scripts.utils import get_domains, get_trainer


def train_gw(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace)
    data.prepare_data()
    data.setup(stage="fit")

    global_workspace = GlobalWorkspace(get_domains(args, data), args.global_workspace.z_size,
                                       args.global_workspace.hidden_size, len(data.classes),
                                       args.losses.coefs.demi_cycles, args.losses.coefs.cycles,
                                       args.losses.coefs.supervision, args.losses.coefs.cosine,
                                       args.global_workspace.optim.lr, args.global_workspace.optim.weight_decay,
                                       args.global_workspace.scheduler.mode, args.global_workspace.scheduler.interval,
                                       args.global_workspace.scheduler.step, args.global_workspace.scheduler.gamma,
                                       args.losses.schedules, data.domain_examples,
                                       args.global_workspace.monitor_grad_norms)

    trainer = get_trainer("train_gw", args, global_workspace, monitor_loss="val_in_dist_total_loss", trainer_args={
        # "val_check_interval": args.global_workspace.prop_labelled_images
    })
    trainer.fit(global_workspace, data)


def train_lm(args):
    seed_everything(args.seed)

    data = SimpleShapesData(args.simple_shapes_path, args.lm.batch_size, args.dataloader.num_workers, False, 1.,
                            args.lm.n_validation_examples, False, {"a": "attr", "t": "t"})
    data.prepare_data()
    data.setup(stage="fit")

    lm = ShapesLM(args.lm.z_size, len(data.classes), data.img_size, args.global_workspace.bert_path,
                  args.lm.optim.lr, args.lm.optim.weight_decay, args.lm.scheduler.step, args.lm.scheduler.gamma,
                  data.domain_examples["in_dist"])

    trainer = get_trainer("train_lm", args, lm, monitor_loss="val_total_loss")
    trainer.fit(lm, data)


def train_ae(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")

    ae = AE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size,
        args.n_validation_examples,
        args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
        data.domain_examples["in_dist"]["v"]
    )

    trainer = get_trainer("train_lm", args, ae, monitor_loss="val_total_loss")
    trainer.fit(ae, data)


def train_vae(args):
    seed_everything(args.seed)

    args.vae.prop_labelled_images = 1.
    args.vae.split_ood = False
    args.vae.selected_domains = {"v": "v"}

    data = load_dataset(args, args.vae)

    data.prepare_data()
    data.setup(stage="fit")
    data.compute_inception_statistics(32, torch.device("cuda" if args.gpus >= 1 else "cpu"))

    vae = VAE(
        data.img_size, data.num_channels, args.vae.ae_size, args.vae.z_size, args.vae.beta, args.vae.type,
        args.n_validation_examples,
        args.vae.optim.lr, args.vae.optim.weight_decay, args.vae.scheduler.step, args.vae.scheduler.gamma,
        data.domain_examples["in_dist"]["v"], args.vae.n_FID_samples
    )

    trainer = get_trainer("train_vae", args, vae, monitor_loss="val_total_loss")
    trainer.fit(vae, data)

    vae.n_FID_samples = data.val_dataset_size  # all the dataset
    trainer.validate(data.val_dataloader())
