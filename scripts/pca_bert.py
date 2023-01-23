import os
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA, KernelPCA

from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    shapes_path = Path(args.simple_shapes_path )

    bert_data_train = np.load(str(shapes_path / f"train_{args.fetchers.t.bert_latents}"))
    bert_data_val = np.load(str(shapes_path / f"val_{args.fetchers.t.bert_latents}"))
    bert_data_test = np.load(str(shapes_path / f"test_{args.fetchers.t.bert_latents}"))
    bert_mean = np.load(shapes_path / f"mean_{args.fetchers.t.bert_latents}")
    bert_std = np.load(shapes_path / f"std_{args.fetchers.t.bert_latents}")
    bert_data_train = (bert_data_train - bert_mean) / bert_std
    bert_data_val = (bert_data_val - bert_mean) / bert_std
    bert_data_test = (bert_data_test - bert_mean) / bert_std
    for k in [16, 32, 64, 128, 256, 512]:
        pca = PCA(n_components=k)
        pca.fit(bert_data_train[:500_000])
        bert_reduced_train = pca.transform(bert_data_train)
        bert_reduced_val = pca.transform(bert_data_val)
        bert_reduced_test = pca.transform(bert_data_test)
        np.save(str(shapes_path / f"train_reduced_{k}_{args.fetchers.t.bert_latents}"), bert_reduced_train)
        np.save(str(shapes_path / f"val_reduced_{k}_{args.fetchers.t.bert_latents}"), bert_reduced_val)
        np.save(str(shapes_path / f"test_reduced_{k}_{args.fetchers.t.bert_latents}"), bert_reduced_test)

