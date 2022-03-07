import os

from bim_gw.scripts.train import train_ae
from bim_gw.utils import get_args

if __name__ == "__main__":
    train_ae(get_args(debug=int(os.getenv("DEBUG", 0))))
