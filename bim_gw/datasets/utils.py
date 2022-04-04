from bim_gw.datasets import CIFARData
from bim_gw.datasets import ImageNetData
from bim_gw.datasets.simple_shapes import SimpleShapesData


def load_dataset(args, local_args, **kwargs):
    if args.visual_dataset == "imagenet":
        print("Loading ImageNet.")
        return ImageNetData(args.image_net_path, local_args.batch_size, args.img_size,
                            args.dataloader.num_workers, local_args.data_augmentation,
                            args.global_workspace.prop_labelled_images,
                            args.global_workspace.classes_labelled_images, **kwargs)
    elif args.visual_dataset == "cifar10":
        print("Loading CIFAR10.")
        if "bimodal" in kwargs:
            raise ValueError("CIFAR is not yet ready for GW training...")
        return CIFARData(args.cifar10_path, local_args.batch_size,
                         args.dataloader.num_workers, local_args.data_augmentation)
    elif args.visual_dataset == "shapes":
        print("Loading Shapes.")
        pre_saved_latent_paths = None
        if "use_pre_saved" in local_args and local_args.use_pre_saved:
            pre_saved_latent_paths = args.global_workspace.load_pre_saved_latents
        return SimpleShapesData(args.simple_shapes_path, local_args.batch_size,
                                args.dataloader.num_workers, local_args.data_augmentation,
                                local_args.prop_labelled_images,
                                args.n_validation_examples, local_args.split_ood,
                                local_args.selected_domains,
                                pre_saved_latent_paths, args.global_workspace.sync_uses_whole_dataset)
    else:
        raise ValueError("The requested dataset is not implemented.")
