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
    pca = PCA(n_components=args.lm.z_size)
    pca.fit(bert_data_train[:500_000])
    # print(pca.explained_variance_ratio_)
    # print(sum(pca.explained_variance_ratio_))
    bert_reduced_train = pca.transform(bert_data_train)
    bert_reduced_val = pca.transform(bert_data_val)
    bert_reduced_test = pca.transform(bert_data_test)
    np.save(str(shapes_path / f"train_reduced_{args.fetchers.t.bert_latents}"), bert_reduced_train)
    np.save(str(shapes_path / f"val_reduced_{args.fetchers.t.bert_latents}"), bert_reduced_val)
    np.save(str(shapes_path / f"test_reduced_{args.fetchers.t.bert_latents}"), bert_reduced_test)

