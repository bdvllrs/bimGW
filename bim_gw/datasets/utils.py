from bim_gw.datasets import CIFARData
from bim_gw.datasets import ImageNetData
from bim_gw.datasets.simple_shapes import SimpleShapesData


def load_vae_dataset(args):
    if args.vae.visual_dataset == "imagenet":
        print("Loading ImageNet.")
        return ImageNetData(args.image_net_path, args.vae.batch_size, args.img_size,
                            args.dataloader.num_workers, args.vae.data_augmentation)
    elif args.vae.visual_dataset == "cifar10":
        print("Loading CIFAR10.")
        return CIFARData(args.cifar10_path, args.vae.batch_size,
                         args.dataloader.num_workers, args.vae.data_augmentation)
    elif args.vae.visual_dataset == "shapes":
        print("Loading Shapes.")
        return SimpleShapesData(args.simple_shapes_path, args.vae.batch_size,
                                args.dataloader.num_workers, args.vae.data_augmentation)
    else:
        raise ValueError("The requested dataset is not implemented.")
