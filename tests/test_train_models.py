import os
from pathlib import Path

from bim_gw.scripts.train import train_gw, train_lm, train_vae
from bim_gw.utils import get_args


def test_train_vae():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml")
    ])
    args.fast_dev_run = True
    train_vae(args)


def test_train_lm():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml")
    ])
    args.fast_dev_run = True
    train_lm(args)


def test_train_gw_with_attributes():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml"),
        Path("configs/test_gw_with_attributes.yaml")
    ])
    args.fast_dev_run = True
    train_gw(args)


def test_train_gw_with_text():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml"),
        Path("configs/test_gw_with_text.yaml")
    ])
    args.fast_dev_run = True
    train_gw(args)


def test_train_gw_with_unsync_examples():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml"),
        Path("configs/test_gw_with_unsync_examples.yaml")
    ])
    args.fast_dev_run = True
    train_gw(args)


def test_train_gw_with_split_ood():
    args = get_args(debug=int(os.getenv("DEBUG", 0)), additional_config_files=[
        Path("configs/test_base.yaml"),
        Path("configs/test_gw_with_split_ood.yaml")
    ])
    args.fast_dev_run = True
    train_gw(args)
