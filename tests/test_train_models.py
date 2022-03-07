import os

from bim_gw.utils import get_args
from bim_gw.scripts.train import train_gw, train_lm, train_vae


def test_train_vae():
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    args.fast_dev_run = True
    train_vae(args)


def test_train_lm():
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    args.fast_dev_run = True
    train_lm(args)


def test_train_gw():
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    args.fast_dev_run = True
    train_gw(args)
