from bim_gw.utils.registers import DatasetRegister


def load_simple_shapes_dataset(args, local_args, **kwargs):
    from bim_gw.datasets.simple_shapes.data_modules import SimpleShapesDataModule

    print("Loading Shapes.")
    pre_saved_latent_paths = None
    sync_uses_whole_dataset = False
    if "use_pre_saved" in local_args and local_args.use_pre_saved:
        pre_saved_latent_paths = args.global_workspace.load_pre_saved_latents
    if "sync_uses_whole_dataset" in local_args and local_args.sync_uses_whole_dataset:
        sync_uses_whole_dataset = True
    return SimpleShapesDataModule(
        args.simple_shapes_path, local_args.batch_size,
        args.dataloader.num_workers,
        local_args.prop_labelled_images,
        local_args.remove_sync_domains,
        args.n_validation_examples, local_args.split_ood,
        local_args.selected_domains,
        pre_saved_latent_paths,
        sync_uses_whole_dataset, fetcher_params=args.fetchers, **kwargs
    )


def load_cmu_mosei_dataset(args, local_args, **kwargs):
    from bim_gw.datasets.cmu_mosei.data_module import CMUMOSEIDataModule

    # TODO: finish cmu_mosei. But how to handle sequences?
    print("Loading CMU MOSEI.")
    return CMUMOSEIDataModule(
        args.cmu_mosei.path, local_args.batch_size, args.dataloader.num_workers,
        local_args.selected_domains, args.cmu_mosei.validate,
        args.cmu_mosei.seq_length
    )


def load_dataset(args, local_args, **kwargs):
    DatasetRegister().add("shapes", load_simple_shapes_dataset)
    # DatasetRegister().add("cmu_mosei", load_cmu_mosei_dataset)

    try:
        dataset = DatasetRegister().get(args.current_dataset)(args, local_args, **kwargs)
    except KeyError:
        raise ValueError("The requested dataset is not implemented.")
    return dataset


def get_lm(args, data, **kwargs):
    raise NotImplementedError("Use get_domains instead.")
