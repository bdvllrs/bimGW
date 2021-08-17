from bim_gw.datasets import CIFARData
from bim_gw.datasets import ImageNetData
from bim_gw.datasets.simple_shapes import SimpleShapesData
from bim_gw.modules import SkipGramLM
from bim_gw.modules.language_model import ShapesLM


def load_dataset(args, **kwargs):
    if args.visual_dataset == "imagenet":
        print("Loading ImageNet.")
        return ImageNetData(args.image_net_path, args.vae.batch_size, args.img_size,
                            args.dataloader.num_workers, args.vae.data_augmentation,
                            args.global_workspace.prop_labelled_images,
                            args.global_workspace.classes_labelled_images, **kwargs)
    elif args.visual_dataset == "cifar10":
        print("Loading CIFAR10.")
        if "bimodal" in kwargs:
            raise ValueError("CIFAR is not yet ready for GW training...")
        return CIFARData(args.cifar10_path, args.vae.batch_size,
                         args.dataloader.num_workers, args.vae.data_augmentation)
    elif args.visual_dataset == "shapes":
        print("Loading Shapes.")
        return SimpleShapesData(args.simple_shapes_path, args.vae.batch_size,
                                args.dataloader.num_workers, args.vae.data_augmentation,
                                args.global_workspace.prop_labelled_images,
                                **kwargs)
    else:
        raise ValueError("The requested dataset is not implemented.")


def get_lm(args, data):
    if args.visual_dataset == "shapes":
        lm = ShapesLM(len(data.classes), data.img_size)
    else:
        lm = SkipGramLM(args.gensim_model_path, data.classes, args.word_embeddings).eval()
    return lm
