import os

import matplotlib

from bim_gw.scripts.train import train_gw
from bim_gw.utils import get_args

matplotlib.use("Agg")

if __name__ == "__main__":
    train_gw(get_args(debug=bool(int(os.getenv("DEBUG", 0)))))
