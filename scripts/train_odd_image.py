import os

from omegaconf import OmegaConf
from ruamel.yaml import YAML
from torch import nn

from bim_gw.datasets.odd_image.data_module import OddImageDataModule
from bim_gw.modules.domain_modules.domain_module import DomainSpecs
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.modules.odd_classifier import OddClassifier
from bim_gw.modules.workspace_encoders import DomainEncoder
from bim_gw.utils import get_args
from bim_gw.utils.scripts import get_domains, get_trainer
from bim_gw.utils.utils import (
    find_best_epoch, get_checkpoint_path,
    get_runs_dataframe
)

yaml = YAML(typ='safe')

coef_param = 'parameters/losses/coefs'
coef_const = f"{coef_param}/contrastive"
coef_demi_cycles = f"{coef_param}/demi_cycles"
coef_cycles = f"{coef_param}/cycles"
coef_translation = f"{coef_param}/translation"
prop_labelled = 'parameters/global_workspace/prop_labelled_images'


def filter_coefs(df, args):
    df = df.loc[df[coef_const] == args.losses.coefs.contrastive]
    df = df.loc[df[coef_demi_cycles] == args.losses.coefs.demi_cycles]
    df = df.loc[df[coef_cycles] == args.losses.coefs.cycles]
    df = df.loc[df[coef_translation] == args.losses.coefs.translation]
    df = df.loc[
        df[prop_labelled] == args.global_workspace.prop_labelled_images]
    df = df.loc[df["parameters/seed"] == args.seed]
    return df


def get_selected_domains(item):
    assert 'parameters/global_workspace/selected_domains' in item
    if isinstance(item['parameters/global_workspace/selected_domains'], str):
        return yaml.load(item['parameters/global_workspace/selected_domains'])
    return item['parameters/global_workspace/selected_domains']


def update_args_from_selected_run(
    df, args, select_row_from_index=None,
    select_row_from_current_coefficients=False
):
    if select_row_from_index is not None:
        item = df.iloc[select_row_from_index].to_dict()
    elif select_row_from_current_coefficients:
        df = filter_coefs(df, args)
        item = df.iloc[0].to_dict()
    else:
        raise ValueError(
            'select_row_from_index or select_row_from_current_coefficients '
            'must be set.'
        )

    args.losses.coefs.demi_cycles = item[coef_demi_cycles]
    args.losses.coefs.cycles = item[coef_cycles]
    args.losses.coefs.contrastive = item[coef_const]
    args.losses.coefs.translation = item[coef_translation]
    args.global_workspace.prop_labelled_images = item[prop_labelled]
    args.global_workspace.selected_domains = get_selected_domains(item)
    if 'parameters/seed' in item:
        args.seed = item['parameters/seed']

    if args.odd_image.encoder.selected_id_key not in item:
        raise ValueError("selected_id_key not in item.")

    item['selected_id'] = item[args.odd_image.encoder.selected_id_key]
    return item


class ExtractFirstItemModule(nn.Module):
    def forward(self, x):
        return x[0]


if __name__ == "__main__":
    args = get_args(debug=bool(os.getenv("DEBUG", False)))

    if args.odd_image.encoder.load_from is not None:
        df = get_runs_dataframe(args.odd_image.encoder)
        item = update_args_from_selected_run(
            df, args, args.odd_image.select_row_from_index,
            args.odd_image.select_row_from_current_coefficients
        )
        args.odd_image.encoder.selected_id = item['selected_id']

    load_domains = []

    if (args.odd_image.encoder.path is None
            or args.odd_image.encoder.path == "random"):
        encoder = nn.Sequential(
            DomainEncoder(
                DomainSpecs(
                    output_dims={"z_img": args.vae.z_size},
                    input_keys=["img"],
                    latent_keys=["z_img"],
                    decoder_activation_fn={},
                    losses={},
                ),
                args.global_workspace.hidden_size.encoder.v,
                args.global_workspace.z_size,
                args.global_workspace.n_layers.encoder.v
            ), nn.Tanh()
        )
        if args.odd_image.encoder.path == "random":
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad_(False)
        load_domains = ["v"]
        encoders = {name: encoder for name in load_domains}
    elif args.odd_image.encoder.path == "identity":
        encoder = ExtractFirstItemModule()
        load_domains = ["v"]
        encoders = {name: encoder for name in load_domains}
    else:
        path = args.odd_image.encoder.path
        if not os.path.isfile(path) and os.path.isdir(path):
            path = find_best_epoch(path)
        global_workspace = GlobalWorkspace.load_from_checkpoint(
            path,
            domain_mods=get_domains(args, args.img_size),
            strict=False
        )
        load_domains = global_workspace.domains.names
        global_workspace.freeze()
        global_workspace.eval()
        encoders = {
            name: nn.Sequential(global_workspace.encoders[name], nn.Tanh()) for
            name in load_domains}

    args.global_workspace.selected_domains = OmegaConf.create(
        [name for name in load_domains]
    )

    if args.resume_from_checkpoint is not None:
        path = get_checkpoint_path(args.resume_from_checkpoint)
        model = OddClassifier.load_from_checkpoint(
            path,
            unimodal_encoders=get_domains(args, args.img_size),
            encoders=encoders
        )
        if args.logger_resume_id is not None:
            for logger in args.loggers:
                logger.args.version = args.logger_resume_id
                logger.args.id = args.logger_resume_id
                logger.args.resume = True
    else:
        model = OddClassifier(
            get_domains(args, 32), encoders, args.global_workspace.z_size,
            args.odd_image.optimizer.lr, args.odd_image.optimizer.weight_decay
        )

    data = OddImageDataModule(
        args.simple_shapes_path, args.global_workspace.load_pre_saved_latents,
        args.odd_image.batch_size, args.dataloader.num_workers,
        args.global_workspace.selected_domains,
        args.domain_loader.t.bert_latents
    )

    if 'attr' in model.unimodal_encoders.keys():
        model.unimodal_encoders['attr'].output_dims = [len(data.classes),
                                                       data.img_size]

    trainer = get_trainer(
        "train_odd_image", args, model,
        monitor_loss="val_loss",
        early_stopping_patience=args.global_workspace.early_stopping_patience,
    )

    trainer.fit(model, data)
