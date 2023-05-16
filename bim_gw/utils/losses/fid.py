"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d


def to_image(tensor, normalize=False):
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    image = tensor.mul(256).clamp(max=255).long()
    if normalize:
        image = image.float() / 255
    return image


def torch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print("Imaginary component {}".format(m))
            return np.inf
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


def get_activations_from_loader(
    data_loader,
    activation_model,
    device,
    z_size=None,
    batch_stop=None,
    verbose=False,
    generation_model=None,
    reconstruction_model=None,
):
    activation_model.eval()
    if generation_model is not None:
        generation_model.eval()

    activations = []

    for i, (batch, _) in enumerate(data_loader):
        if generation_model is None:
            batch = batch.to(device)
            if reconstruction_model is not None:
                batch, _, _, _, _, _ = reconstruction_model(batch)
        else:
            assert z_size is not None
            samples = torch.randn(batch.size(0), z_size)
            samples = samples.to(device)
            batch = generation_model.generate(samples)

        pred = activation_model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        activations.append(pred.view(pred.size(0), -1).detach().cpu())

        if (batch_stop is not None) and (i == batch_stop):
            break

    activations = torch.cat(activations, 0)

    mu = torch.mean(activations, dim=0)
    sigma = torch_cov(activations, rowvar=False)

    return activations.numpy(), mu.numpy(), sigma.numpy()


def get_activations_from_generation(
    model, inception_model, z_size, device, n_fid_samples=1000, batch_size=128
):
    activations = []
    i = 0
    while i < n_fid_samples:
        b_s = min(n_fid_samples - i, batch_size)

        # samples = torch.randn(b_s, generation_model.z_size)
        samples = torch.randn(b_s, z_size)
        samples = samples.to(device)
        generated = model.generate(samples)

        generated = generated[:, :3]

        generated = to_image(generated, normalize=True)

        pred = inception_model(generated)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        activations.append(pred.view(pred.size(0), -1).detach().cpu())
        i += b_s

    activations = torch.cat(activations, 0)

    mu = torch.mean(activations, dim=0)
    sigma = torch_cov(activations, rowvar=False)

    return activations.numpy(), mu.numpy(), sigma.numpy()


def output_mse(data_loader, generation_model, device):
    generation_model.eval()
    all_mse = []
    for i, batch in enumerate(data_loader):
        batch = batch["v"]
        batch = batch.to(device)

        _, reco = generation_model(batch)

        reco = reco[:, :3]

        reco = to_image(reco, normalize=True)
        batch = to_image(batch, normalize=True)
        mse = F.mse_loss(
            reco.view(reco.size(0), -1),
            batch.view(batch.size(0), -1),
            reduction="none",
        )

        all_mse.append(mse.sum(dim=1).detach().cpu())

    all_mse = torch.cat(all_mse, dim=0)
    return all_mse
