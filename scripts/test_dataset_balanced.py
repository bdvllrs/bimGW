import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
import torchvision
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.datasets.utils import get_lm
from bim_gw.modules.gw import GlobalWorkspace
from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args
from bim_gw.utils.shapes import log_shape_fig


def test_dataset_balanced(args):
    seed_everything(args.seed)

    data = load_dataset(args, args.global_workspace, bimodal=True)
    data.prepare_data()
    data.setup(stage="fit")

    labels = np.array(data.train_datasets["t"].labels)
    cls = 2
    labels = labels[labels[:, 0] == cls]
    plt.hist(labels[:, 1], 200, density=True)
    plt.title(f"x cls {cls}")
    plt.show()
    plt.hist(labels[:, 2], 200, density=True)
    plt.title(f"y cls {cls}")
    plt.show()
    plt.hist(labels[:, 3], 200, density=True)
    plt.title(f"scale cls {cls}")
    plt.show()
    plt.hist(labels[:, 4], 200, density=True)
    plt.title(f"rotation cls {cls}")
    plt.show()
    plt.hist(labels[:, 5], 200, density=True)
    plt.title(f"r cls {cls}")
    plt.show()
    plt.hist(labels[:, 6], 200, density=True)
    plt.title(f"g cls {cls}")
    plt.show()
    plt.hist(labels[:, 7], 200, density=True)
    plt.title(f"b cls {cls}")
    plt.show()
    print('ok')


if __name__ == "__main__":
    test_dataset_balanced(get_args(debug=int(os.getenv("DEBUG", 0))))
