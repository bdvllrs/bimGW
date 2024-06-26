# -*- coding: utf-8 -*-

from pathlib import Path

from bim_gw.scripts.train import train_gw, train_lm, train_vae
from bim_gw.utils import get_args

tests_folder = Path(__file__).absolute().parent.parent


def test_train_vae():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[tests_folder / "configs/test_base.yaml"],
    )
    train_vae(args)


def test_train_lm():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[tests_folder / "configs/test_base.yaml"],
    )
    train_lm(args)


def test_train_gw_with_attributes():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/load_pretrained_models.yaml",
            tests_folder / "configs/test_gw_with_attributes.yaml",
        ],
    )
    train_gw(args)


def test_train_gw_with_text():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/load_pretrained_models.yaml",
            tests_folder / "configs/test_gw_with_text.yaml",
        ],
    )
    train_gw(args)


def test_train_gw_with_unsync_examples():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/load_pretrained_models.yaml",
            tests_folder / "configs/test_gw_with_unsync_examples.yaml",
        ],
    )
    train_gw(args)


def test_train_gw_with_split_ood():
    args = get_args(
        use_local=False,
        cli=False,
        additional_config_files=[
            tests_folder / "configs/test_base.yaml",
            tests_folder / "configs/load_pretrained_models.yaml",
            tests_folder / "configs/test_gw_with_split_ood.yaml",
        ],
    )
    train_gw(args)
