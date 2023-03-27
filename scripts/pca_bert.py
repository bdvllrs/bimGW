import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    shapes_path = Path(args.simple_shapes_path)
    domain_args = args.domain_loader

    bert_data_train = np.load(
        str(shapes_path / f"train_{domain_args.t.bert_latents}")
    )
    bert_data_val = np.load(
        str(shapes_path / f"val_{domain_args.t.bert_latents}")
    )
    bert_data_test = np.load(
        str(shapes_path / f"test_{domain_args.t.bert_latents}")
    )
    bert_mean = np.load(shapes_path / f"mean_{domain_args.t.bert_latents}")
    bert_std = np.load(shapes_path / f"std_{domain_args.t.bert_latents}")
    bert_data_train = (bert_data_train - bert_mean) / bert_std
    bert_data_val = (bert_data_val - bert_mean) / bert_std
    bert_data_test = (bert_data_test - bert_mean) / bert_std
    for k in [16, 32, 64, 128, 256, 512]:
        pca = PCA(n_components=k)
        pca.fit(bert_data_train[:500_000])
        bert_reduced_train = pca.transform(bert_data_train)
        bert_reduced_val = pca.transform(bert_data_val)
        bert_reduced_test = pca.transform(bert_data_test)
        np.save(
            str(
                shapes_path / f"train_reduced_{k}_"
                              f"{domain_args.t.bert_latents}"
            ), bert_reduced_train
        )
        np.save(
            str(
                shapes_path / f"val_reduced_{k}_{domain_args.t.bert_latents}"
            ), bert_reduced_val
        )
        np.save(
            str(
                shapes_path / f"tes"
                              f"t_reduced_{k}_{domain_args.t.bert_latents}"
            ), bert_reduced_test
        )
        mean = bert_reduced_train.mean(axis=0)
        std = bert_reduced_train.std(axis=0)
        np.save(
            str(
                shapes_path / f"mean_reduced_{k}_"
                              f"{domain_args.t.bert_latents}"
            ), mean
        )
        np.save(
            str(
                shapes_path / f"std_reduced_{k}_{domain_args.t.bert_latents}"
            ), std
        )
