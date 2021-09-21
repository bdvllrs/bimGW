import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

from bim_gw.modules.vae import VAE
from bim_gw.utils import get_args


def explore_vae(args):
    seed_everything(args.seed)

    device = torch.device('cuda')

    vae = VAE.load_from_checkpoint(args.global_workspace.vae_checkpoint).to(device).eval()
    vae.freeze()

    print("Z size", vae.z_size)

    n = 13
    start = -3
    end = 3
    imsize = vae.image_size + 2

    for dim_i in range(vae.z_size - 1):
        for dim_j in range(dim_i + 1, vae.z_size):
            # dim_i = 0
            # dim_j = 2

            z = torch.zeros(n, n, vae.z_size).to(device)
            # z[:, :, 4] = -1.5
            for i in range(n):
                step = start + (end - start) * float(i) / float(n-1)
                z[i, :, dim_i] = step
            for j in range(n):
                step = start + (end - start) * float(j) / float(n-1)
                z[:, j, dim_j] = step

            sampled_images = vae.decoder(z.reshape(-1, vae.z_size))

            sampled_images = sampled_images - sampled_images.min()
            sampled_images = sampled_images / sampled_images.max()
            img_grid = torchvision.utils.make_grid(sampled_images, nrow=n)
            img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
            plt.imshow(img_grid)
            plt.xlabel(f"dim {dim_j}")
            plt.ylabel(f"dim {dim_i}")
            plt.gca().set_xticks(imsize * np.arange(n) + imsize // 2)
            plt.gca().set_xticklabels(list(map(lambda x: f"{x:.1f}", np.linspace(start, end, n))))
            plt.gca().set_yticks(imsize * np.arange(n) + imsize // 2)
            plt.gca().set_yticklabels(list(map(lambda x: f"{x:.1f}", np.linspace(start, end, n))))
            plt.show()

if __name__ == "__main__":
    explore_vae(get_args(debug=int(os.getenv("DEBUG", 0))))
