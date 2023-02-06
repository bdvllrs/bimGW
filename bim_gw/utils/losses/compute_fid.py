import os

import numpy as np
import torch

from bim_gw.utils.inception import InceptionV3
from bim_gw.utils.losses.fid import (
    calculate_frechet_distance, get_activations_from_generation, get_activations_from_loader, output_mse
)


def compute_dataset_statistics(dataset, stats_path, dataset_name, batch_size, device):
    """
    Compute the statistics of the dataset to compute the FID
    Args:
        dataset: dataset to compute the stats
        stats_path: path to save the statistics in.
        dataset_name: name of the dataset. The stat file will be `inception_stats_{dataset_name}.npy`
        batch_size:
        device:
    """
    if not os.path.exists(os.path.join(stats_path, f'inception_stats_{dataset_name}.npy')):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        activation_model = InceptionV3([block_idx])
        activation_model.to(device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        os.makedirs(stats_path, exist_ok=True)
        print('Compute the FID statistics on the dataset', dataset_name)

        _, mu, sigma = get_activations_from_loader(loader, activation_model, device)
        np.save(os.path.join(stats_path, f'inception_stats_{dataset_name}.npy'), {'mu': mu, 'sigma': sigma})
    return os.path.join(stats_path, f'inception_stats_{dataset_name}.npy')


def compute_FID(stats_path, dataloader, model, z_size, input_size, device, n_fid_samples=None):
    model.eval()

    mse = output_mse(dataloader, model, device)
    mse = mse.mean(dim=0).item() / np.prod(input_size)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    inception_model = InceptionV3([block_idx])
    inception_model.to(device)

    inception_model.eval()

    # stats_path = os.path.join(args.dataset_root, args.dataset, 'inception_train_statistics.npy')

    stat = np.load(stats_path, allow_pickle=True).item()
    mu_dataset = stat['mu']
    sigma_dataset = stat['sigma']

    _, mu_model, sigma_model = get_activations_from_generation(
        model, inception_model, z_size, device,
        n_fid_samples=n_fid_samples
    )

    fid_value = calculate_frechet_distance(mu_dataset, sigma_dataset, mu_model, sigma_model)

    return fid_value, mse
