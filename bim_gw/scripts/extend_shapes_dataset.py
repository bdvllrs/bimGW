from pathlib import Path

import numpy as np

from bim_gw.utils.shapes import (
    generate_dataset,
    generate_transformations,
    load_labels,
    merge_labels,
    save_dataset,
    save_labels,
)


def extend_shapes_dataset(
    simple_shapes_path,
    seed,
    image_size,
    n_train,
    n_val,
    n_test,
    possible_categories,
    min_x,
    max_x,
    min_y,
    max_y,
    min_scale,
    max_scale,
    min_rotation,
    max_rotation,
    min_lightness,
    max_lightness,
    min_hue,
    max_hue,
):
    dataset_location = Path(simple_shapes_path)

    assert dataset_location.is_dir()

    size_train_set = n_train
    size_val_set = n_val
    size_test_set = n_test

    np.random.seed(seed)

    # np.save(
    #     dataset_location / "val_labels.npy",
    #     np.load(dataset_location / "val_labels.npy")[:50000],
    # )
    # np.save(
    #     dataset_location / "test_labels.npy",
    #     np.load(dataset_location / "test_labels.npy")[:50000],
    # )

    train_labels, train_transfo = load_labels(
        dataset_location / "train_labels.npy"
    )
    val_labels, val_transfo = load_labels(dataset_location / "val_labels.npy")
    test_labels, test_transfo = load_labels(
        dataset_location / "test_labels.npy"
    )
    len_train = train_labels["classes"].shape[0]
    len_val = val_labels["classes"].shape[0]
    len_test = test_labels["classes"].shape[0]

    new_train_labels = generate_dataset(
        size_train_set - len_train,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        image_size,
    )
    new_train_transfo_labels, new_train_transfo = generate_transformations(
        new_train_labels,
        generate_dataset(
            size_train_set - len_train,
            min_scale,
            max_scale,
            min_lightness,
            max_lightness,
            possible_categories,
            min_hue,
            max_hue,
            min_rotation,
            max_rotation,
            min_x,
            max_x,
            min_y,
            max_y,
            image_size,
        ),
    )
    new_val_labels = generate_dataset(
        size_val_set - len_val,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        image_size,
    )
    new_val_transfo_labels, new_val_transfo = generate_transformations(
        new_val_labels,
        generate_dataset(
            size_val_set - len_val,
            min_scale,
            max_scale,
            min_lightness,
            max_lightness,
            possible_categories,
            min_hue,
            max_hue,
            min_rotation,
            max_rotation,
            min_x,
            max_x,
            min_y,
            max_y,
            image_size,
        ),
    )
    new_test_labels = generate_dataset(
        size_test_set - len_test,
        min_scale,
        max_scale,
        min_lightness,
        max_lightness,
        possible_categories,
        min_hue,
        max_hue,
        min_rotation,
        max_rotation,
        min_x,
        max_x,
        min_y,
        max_y,
        image_size,
    )
    new_test_transfo_labels, new_test_transfo = generate_transformations(
        new_test_labels,
        generate_dataset(
            size_test_set - len_test,
            min_scale,
            max_scale,
            min_lightness,
            max_lightness,
            possible_categories,
            min_hue,
            max_hue,
            min_rotation,
            max_rotation,
            min_x,
            max_x,
            min_y,
            max_y,
            image_size,
        ),
    )

    total_train_labels = merge_labels(train_labels, new_train_labels)
    total_val_labels = merge_labels(val_labels, new_val_labels)
    total_test_labels = merge_labels(test_labels, new_test_labels)

    total_train_transfo = merge_labels(train_transfo, new_train_transfo)
    total_val_transfo = merge_labels(val_transfo, new_val_transfo)
    total_test_transfo = merge_labels(test_transfo, new_test_transfo)

    print("Save labels...")
    save_labels(
        dataset_location / "train_labels.npy",
        total_train_labels,
        total_train_transfo,
    )
    save_labels(
        dataset_location / "val_labels.npy",
        total_val_labels,
        total_val_transfo,
    )
    save_labels(
        dataset_location / "test_labels.npy",
        total_test_labels,
        total_test_transfo,
    )

    print(new_val_labels["classes"].shape)

    print("Saving training set...")
    (dataset_location / "transformed").mkdir(exist_ok=True)
    (dataset_location / "train").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "train", new_train_labels, image_size, len_train
    )
    (dataset_location / "transformed" / "train").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "train",
        new_train_transfo_labels,
        image_size,
        len_train,
    )
    print("Saving validation set...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_dataset(dataset_location / "val", new_val_labels, image_size, len_val)
    (dataset_location / "transformed" / "val").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "val",
        new_val_transfo_labels,
        image_size,
        len_val,
    )
    print("Saving test set...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "test", new_test_labels, image_size, len_test
    )
    (dataset_location / "transformed" / "test").mkdir(exist_ok=True)
    save_dataset(
        dataset_location / "transformed" / "test",
        new_test_transfo_labels,
        image_size,
        len_test,
    )

    # print("Saving captions...")
    #
    # lengths = {"train": len_train, "val": len_val, "test": len_test}
    #
    # for split in ["train", "val", "test"]:
    #     labels = np.load(str(dataset_location / f"{split}_labels.npy"))[
    #         lengths[split] :
    #     ]
    #     new_captions = []
    #     new_choices = []
    #     for k in tqdm(range(labels.shape[0]), total=labels.shape[0]):
    #         caption, choice = composer(
    #             {
    #                 "shape": int(labels[k][0]),
    #                 "rotation": labels[k][4],
    #                 "color": (labels[k][5], labels[k][6], labels[k][7]),
    #                 "size": labels[k][3],
    #                 "location": (labels[k][1], labels[k][2]),
    #             }
    #         )
    #         new_captions.append(caption)
    #         new_choices.append(choice)
    #     old_captions = np.load(str(dataset_location / f"{split}_captions.npy"))
    #     old_choices = np.load(
    #         str(dataset_location / f"{split}_caption_choices.npy"),
    #         allow_pickle=True,
    #     )
    #     captions = np.concatenate([old_captions, new_captions], axis=0)
    #     choices = {
    #         k: np.concatenate([old_choices[k], new_choices[k]], axis=0)
    #         for k in old_choices.keys()
    #     }
    #
    #     np.save(str(dataset_location / f"{split}_captions.npy"), captions)
    #     np.save(
    #         str(dataset_location / f"{split}_caption_choices.npy"), choices
    #     )
    #
    # print("Extracting BERT features...")
    # bert_latents = args.domain_loader.t.bert_latents
    # args.domain_loader.t.bert_latents = None
    # args.global_workspace.use_pre_saved = False
    # args.global_workspace.prop_labelled_images = 1.0
    # args.global_workspace.split_ood = False
    # args.global_workspace.sync_uses_whole_dataset = True
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.global_workspace.selected_domains = OmegaConf.create(["t"])
    # data = load_dataset(args, args.global_workspace, add_unimodal=False)
    # data.prepare_data()
    # data.setup(stage="fit")
    # save_bert_latents(
    #     data,
    #     args.global_workspace.bert_path,
    #     bert_latents,
    #     args.simple_shapes_path,
    #     device,
    # )
    #
    # print("done!")
