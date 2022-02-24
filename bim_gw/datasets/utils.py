from bim_gw.datasets import CIFARData
from bim_gw.datasets import ImageNetData
from bim_gw.datasets.simple_shapes import SimpleShapesData
from bim_gw.modules import SkipGramLM
from bim_gw.modules.language_model import ShapesLM, ShapesAttributesLM


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
        return SimpleShapesData(args.simple_shapes_path, local_args.batch_size,
                                args.dataloader.num_workers, local_args.data_augmentation,
                                local_args.prop_sync_domains,
                                args.n_validation_examples, local_args.split_ood,
                                local_args.selected_domains)
    else:
        raise ValueError("The requested dataset is not implemented.")


def get_lm(args, data, **kwargs):
    if args.global_workspace.text_domain == "attributes":
        lm = ShapesAttributesLM(len(data.classes), data.img_size)
    elif args.global_workspace.text_domain == "bert":
        lm = ShapesLM.load_from_checkpoint(
            args.global_workspace.lm_checkpoint,
            bert_path=args.global_workspace.bert_path)
    else:
        lm = SkipGramLM(args.gensim_model_path, data.classes, args.word_embeddings).eval()
    return lm
