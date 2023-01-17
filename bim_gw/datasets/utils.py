from bim_gw.modules.domain_modules.simple_shapes.text import SimpleShapesText
from bim_gw.modules.domain_modules.simple_shapes.attributes import SimpleShapesAttributes


def load_dataset(args, local_args, **kwargs):
    if args.visual_dataset == "shapes":
        from bim_gw.datasets.simple_shapes.data_modules import SimpleShapesDataModule

        print("Loading Shapes.")
        pre_saved_latent_paths = None
        sync_uses_whole_dataset = False
        if "use_pre_saved" in local_args and local_args.use_pre_saved:
            pre_saved_latent_paths = args.global_workspace.load_pre_saved_latents
        if "sync_uses_whole_dataset" in local_args and local_args.sync_uses_whole_dataset:
            sync_uses_whole_dataset = True
        return SimpleShapesDataModule(args.simple_shapes_path, local_args.batch_size,
                                      args.dataloader.num_workers,
                                      local_args.prop_labelled_images,
                                      local_args.remove_sync_domains,
                                      args.n_validation_examples, local_args.split_ood,
                                      local_args.selected_domains,
                                      pre_saved_latent_paths,
                                      sync_uses_whole_dataset, fetcher_params=args.fetchers, **kwargs)
    # TODO: finish cmu_mosei. But how to handle sequences?
    # elif args.visual_dataset == "cmu_mosei":
    #     from bim_gw.datasets.cmu_mosei.data_module import CMUMOSEIDataModule
    #
    #     print("Loading CMU MOSEI.")
    #     return CMUMOSEIDataModule(args.cmu_mosei.path, local_args.batch_size, args.dataloader.num_workers,
    #                               local_args.selected_domains, args.cmu_mosei.validate,
    #                               args.cmu_mosei.seq_length, **kwargs)
    else:
        raise ValueError("The requested dataset is not implemented.")


def get_lm(args, data, **kwargs):
    if args.global_workspace.text_domain == "attributes":
        lm = SimpleShapesAttributes(len(data.classes), data.img_size)
    elif args.global_workspace.text_domain == "bert":
        lm = SimpleShapesText.load_from_checkpoint(
            args.global_workspace.lm_checkpoint,
            bert_path=args.global_workspace.bert_path)
    return lm
