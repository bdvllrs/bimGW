import csv
import os
from pathlib import Path

import numpy as np

from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    root_path = Path(args.simple_shapes_path)

    for split in ["train", "val", "test"]:
        labels = []
        ids = []

        with open(root_path / f"{split}_labels.csv", "r") as f:
            reader = csv.reader(f)
            for k, line in enumerate(reader):
                if k > 0:
                    labels.append(list(map(float, line)))
        labels = np.array(labels, dtype=np.float32)
        np.save(str(root_path / f"{split}_labels.npy"), labels)
