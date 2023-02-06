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
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import datasets, transforms

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from bim_gw.utils.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


# parser.add_argument('path', type=str, nargs=2,
#                    help=('Path to the generated images or '
#                          'to .npz statistic files'))
# parser.add_argument('--batch-size', type=int, default=50,
#                    help='Batch size to use')
# parser.add_argument('--dims', type=int, default=2048,
#                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                    help=('Dimensionality of Inception features to use. '
#                          'By default, uses pool3 features'))
# parser.add_argument('-c', '--gpu', default='', type=str,
#                    help='GPU to use (leave blank for CPU only)')

def to_image(tensor, normalize=False):
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    image = tensor.mul(256).clamp(max=255).long()
    if normalize:
        image = image.float() / 255
    return image


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def imread_custom(filename, dataset):
    print(dataset)
    if filename.endswith('.npy'):
        data = np.load(filename, allow_pickle=True)
        # data = data[:,:,2:-2,2:-2]

        data = ((data / np.max(data)) * 255 - 127) / 127
        # data/np.max(data)
        if dataset == 'MNIST':
            data = np.repeat(data.reshape(data.shape[0], 32, 32, 1), 3, 3)
        elif dataset == 'CIFAR':
            data = data.reshape(10000, 32, 32, 3)
        elif dataset == 'CelebA':
            data = data.reshape(-1, 64, 64, 3)
    else:
        if dataset == 'MNIST':
            data = torch.load(filename)[0].numpy()
            data = np.pad(data, [[0], [2], [2]], mode='constant')
            # data = data.reshape(data.shape[0], 28, 28, 1)
            data = ((data / np.max(data)) * 255 - 127) / 127
            data = np.repeat(data.reshape(data.shape[0], 32, 32, 1), 3, 3)
        elif dataset == 'CIFAR':
            data = unpickle(filename).astype(np.uint8)
            data = ((data / np.max(data)) * 255 - 127) / 127
            data = data.reshape(data.shape[0], 32, 32, 3)
        elif dataset == 'CelebA':
            data = torch.utils.data.DataLoader(
                datasets.CelebA(
                    filename, split='test', download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            # transforms.Normalize((0.5, 0.5, 0.5),
                            #                     (0.5, 0.5, 0.5)),
                        ]
                    )
                ),
                batch_size=50, shuffle=False
            )
    return data


def imread_custom_CIFAR(filename, dataset=True):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    if dataset == True:
        return unpickle(filename).astype(np.uint8)
    else:
        return torch.load(filename).numpy().astype(np.uint8)


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    fo.close()
    return X


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


def imread_custom_MNIST(filename, dataset=True):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return torch.load(filename)[0].astype(np.uint8)


def output_mse(data_loader, generation_model, device):
    generation_model.eval()
    all_mse = []
    for i, batch in enumerate(data_loader):
        batch = batch["v"]
        batch = batch.to(device)

        # reco, _, _, _, _, _ = generation_model(batch)
        _, reco = generation_model(batch)

        reco = reco[:, :3]

        reco = to_image(reco, normalize=True)
        batch = to_image(batch, normalize=True)
        # print('data', batch.shape, batch.min(), batch.max())
        # print('reco', reco.shape, reco.min(), reco.max())
        mse = F.mse_loss(reco.view(reco.size(0), -1), batch.view(batch.size(0), -1), reduction='none')

        all_mse.append(mse.sum(dim=1).detach().cpu())

    all_mse = torch.cat(all_mse, dim=0)
    return all_mse


# def generate_data(data_loader, generation_model, args):
#     generation_model.eval()
#     all_generation = []
#     for i, (batch, _) in enumerate(data_loader):

#         if 'energy' in args.flow:
#             _, _, generated = model.prior(batch.size(0))
#         else:
#             samples = torch.randn(batch.size(0), args.z_size)
#             if args.cuda:
#                 samples = samples.cuda()
#             generated = generation_model.generate(samples)

#         all_generation.append(generated.detach().cpu().numpy())

#     all_generation = np.concatenate(all_generation, axis=0)
#     return all_generation

# args.dataset, model, args

def generate_data(generation_model, activation_model, args, n_fid_samples=None, batch_size=128):
    generation_model.eval()

    if n_fid_samples is None:
        n_fid_samples = args.n_fid_samples if 'n_fid_samples' in args.__dict__ else 50000

    # for i, (batch, _) in enumerate(data_loader):
    all_generation = []
    i = 0
    while i < n_fid_samples:
        # for i in range(nb_batch):
        # samples = torch.randn(batch.size(0), args.z_size)
        b_s = min(n_fid_samples - i, batch_size)

        if 'energy' in args.flow:
            _, _, generated = model.prior(b_s)

        else:
            # samples = torch.randn(b_s, generation_model.z_size)
            samples = torch.randn(b_s, args.z_size)
            if args.cuda:
                samples = samples.cuda()
            generated = generation_model.generate(samples)

        if generation_model.input_type == 'multinomial':
            # Find largest class logit
            tmp = generated.view(-1, 256, *args.input_size).max(dim=1)[1]
            # tmp = generated.view(-1, 256, *generation_model.input_size).max(dim=1)[1]
            generated = tmp.float() / (256 - 1.)

        elif generated.min() < 0:
            generated = (generated + 1) / 2

        all_generation.append(generated)
        i += min(n_fid_samples - i, batch_size)

    all_generation = torch.cat(all_generation, 0)

    all_generation = all_generation.numpy()

    return all_generation


def get_activations_from_generation_(data_loader, generation, activation_model, args):
    activation_model.eval()
    generation = torch.tensor(generation)
    if args.cuda:
        generation = generation.cuda()

    for i, (batch, _) in enumerate(data_loader):
        batch_size = batch.size(0)
        data = generation[i * batch_size:(i + 1) * batch_size]
        pred = activation_model(data)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        if i == 0:
            pred_arr = pred.view(pred.size(0), -1).cpu().data
        else:
            pred_arr = torch.cat((pred_arr, pred.view(pred.size(0), -1).cpu().data), dim=0)

    return pred_arr.numpy()


# def to_image(tensor, args, normalize=False):
#     if args.input_type == 'multinomial':
#         # data is already between 0 and 1
#         num_classes = 256
#         # Find largest class logit
#         image = tensor.view(-1, num_classes, *args.input_size).max(dim=1)[1]
#         # recon_mean = tmp.float() / (num_classes - 1.)
#         if normalize:
#             image = image.float() / num_classes
#         return image
#     elif args.input_type == 'gaussian':
#         if ('normalize_pixels' in args.__dict__ and args.normalize_pixels) or tensor.min()<0:
#             tensor = (tensor+1)/2

#         image = tensor.mul(255).long()
#         if normalize:
#             image = image.float() / 255

#         return image

#     elif args.input_type == 'logistic':
#         if ('normalize_pixels' in args.__dict__ and args.normalize_pixels) or tensor.min()<0:
#             tensor = (tensor+1)/2

#         image = tensor.mul(255).long()
#         if normalize:
#             image = image.float() / 255

#         return image

#     elif args.input_type == 'binary':
#         image = tensor.mul(255).long()

#         if normalize:
#             image = image.float() / 255
#         return image
#     else:
#         raise ValueError('Invalid input type {}'.format(args.input_type))

def get_activations_from_generation(model, inception_model, z_size, device, n_fid_samples=1000, batch_size=128):
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


def get_activations_from_loader(
        data_loader, activation_model, device, z_size=None, batch_stop=None, verbose=False,
        generation_model=None, reconstruction_model=None
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


def get_activations(
        file, model, batch_size=50, dims=2048,
        cuda=False, verbose=False, dataset='CelebA', loader=True
):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    data_arr = imread_custom(file, dataset)
    if dataset == 'CelebA' and (type(data_arr) is torch.utils.data.dataloader.DataLoader):
        pred_arr = np.empty((len(data_arr) * 50, dims))
        # Model evaluation
        for i, (batch, _) in tqdm(enumerate(data_arr)):
            if cuda:
                batch = batch.cuda()
            # batch = ((batch/torch.max(batch))*255-127)/127
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            if pred.size(0) == 50:
                pred_arr[i:i + pred.size(0)] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        if verbose:
            print(' done')
    else:
        print(data_arr.shape)
        data_arr = data_arr.transpose((0, 3, 1, 2))
        pred_arr = np.empty((data_arr.shape[0], dims))

        # Model evaluation
        for i in tqdm(range(0, len(data_arr), batch_size)):
            batch = torch.from_numpy(data_arr[i:i + batch_size]).type(torch.FloatTensor)
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[i:i + batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        if verbose:
            print(' done')

    return pred_arr


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

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print('Imaginary component {}'.format(m))
            return np.inf
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(
        file, model, batch_size=50,
        dims=2048, cuda=False, verbose=False, loader=False
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(file, model, batch_size, dims, cuda, verbose, loader=loader)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, loader=False):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(path, model, batch_size, dims, cuda, loader)
    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims, model, m1=None, s1=None):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    if cuda:
        model.cuda()
    if m1 is None and s1 is None:
        print('activation of set 1')
        m1, s1 = _compute_statistics_of_path(
            paths[0], model, batch_size,
            dims, cuda, loader=True
        )
    else:
        print('received m1 and s1')
    print('activation of set 2')
    m2, s2 = _compute_statistics_of_path(
        paths[1], model, batch_size,
        dims, cuda
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, m1, s1, m2, s2


# paths = ['/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/AE_CIFAR_100_good_conv_2',
#         '/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/AE_CIFAR_baseline_236',
#        '/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/CIFAR_100_final_2']

def FID_comp(paths):
    for path in paths:
        FIDS = []
        for k in np.logspace(-3, 1, 10):
            print('testing model : ', k)
            args.path = ['/media/data_cifs/mchalvid/Project_synchrony/CIFAR/cifar-10-batches-py/test_batch',
                         path + '/0/10000_samples_KD_{}.npy'.format(k)]
            fid_value = calculate_fid_given_paths(
                args.path,
                args.batch_size,
                args.gpu != '',
                args.dims,
                model
            )
            FIDS.append(fid_value)
            print('FID : ', fid_value)
        np.save(path + '/FID_array.npy', np.array(FIDS))


if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = 50
    args.gpu = True
    args.dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx])
    comp = '___'
    if comp == 'AE':
        # args.path = ['/media/data_cifs/mchalvid/Project_synchrony/MNIST/MNIST/processed/test.pt',
        #         '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/AE_encoder_vanilla_100epochs_rec_run2/recos_100.npy']
        args.path = ['/media/data_cifs/mchalvid/Project_synchrony/CIFAR/cifar-10-batches-py/test_batch',
                     '/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/AE_CIFAR_100_good_conv_2/0/recos_20.npy']
    elif comp == 'ODE':
        # args.path = ['/media/data_cifs/mchalvid/Project_synchrony/MNIST/MNIST/processed/test.pt',
        #        '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/MNIST_t=10_activations_100epochs/recos_10.npy']
        # args.path = ['/media/data_cifs/mchalvid/Project_synchrony/MNIST/MNIST/processed/test.pt',
        #         '/media/data_cifs/mchalvid/Project_synchrony/MNIST/results/MNIST_t=10_init_gaussian/5/10000_samples_GMM.npy']
        args.path = ['/media/data_cifs/mchalvid/Project_synchrony/CIFAR/cifar-10-batches-py/test_batch',
                     '/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/CIFAR_100_final_2/0/recos_100.npy']
    elif comp == 'search':
        FIDS = []
        for i in [int(np.ceil(i)) for i in np.logspace(1, 4, 10)]:
            print('testing model : ', i)
            args.path = ['/media/data_cifs/mchalvid/Project_synchrony/CIFAR/cifar-10-batches-py/test_batch',
                         '/media/data_cifs/mchalvid/Project_synchrony/CIFAR/results/CIFAR_100_final_2/0/10000_samples_GMM_{}_components_search.npy'.format(
                             i
                         )]
            fid_value = calculate_fid_given_paths(
                args.path,
                args.batch_size,
                args.gpu != '',
                args.dims,
                model
            )
            FIDS.append(fid_value)
            print('FID : ', fid_value)
    elif comp == 'CelebA_debug':
        args.path = ['/media/data_cifs/mchalvid/Project_synchrony/CelebA/2/',
                     '/media/data_cifs/mchalvid/Project_synchrony/CelebA/2/results/CelebA_100_debug_4_nodatanorm/0/recos_40.npy']
