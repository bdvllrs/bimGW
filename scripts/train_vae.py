import os

from bim_gw.scripts.train import train_vae
from bim_gw.utils import get_args

if __name__ == "__main__":
    train_vae(get_args(debug=bool(int(os.getenv("DEBUG", 0)))))