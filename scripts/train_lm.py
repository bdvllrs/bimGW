import os

from bim_gw.scripts.train import train_lm
from bim_gw.utils import get_args

if __name__ == "__main__":
    train_lm(get_args(debug=int(os.getenv("DEBUG", 0))))
