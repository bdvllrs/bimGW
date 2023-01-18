import os

from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm

from bim_gw.datasets.simple_shapes import SimpleShapesData
from bim_gw.modules import ShapesLM
from bim_gw.utils import get_args
from bim_gw.utils.text_composer.writers import LocationQuantizer, RotationQuantizer, SizeQuantizer

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    seed_everything(args.seed)

    data = SimpleShapesData(args.simple_shapes_path, args.lm.batch_size, args.dataloader.num_workers, False, 1.,
                            args.lm.n_validation_examples, False, {"a": "attr", "t": "t"})
    data.prepare_data()
    data.setup(stage="fit")
    n_samples = len(data.train_set["a"])
    # n_samples = 10_000

    quantizers = {
        "location": LocationQuantizer(0),
        "rotation": RotationQuantizer(0),
        "size": SizeQuantizer(0),
    }
    indices = {
        "location": [0, 1],
        "rotation": [3, 4],
        "size": [2]
    }

    for quantizer_name, quantizer in quantizers.items():
        if type(quantizer.y_axis_labels[0][0]) is list:
            labels = [l for labels in quantizer.y_axis_labels[0] for l in labels]
        else:
            labels = [l for l in quantizer.y_axis_labels[0]]

        location_counts = {l: 0 for l in labels}
        for k in tqdm(range(n_samples)):
            cls, attrs = data.train_set["a"][k]
            inputs =  [attrs[idx] for idx in indices[quantizer_name]]
            location = quantizer(*inputs)
            location_counts[location] += 1

        bar_x = []
        bar_y = []
        for k in range(len(labels)):
            bar_x.append(k * 0.7)
            bar_y.append(location_counts[labels[k]])
        p = plt.bar(bar_x, bar_y, 0.2)
        plt.gca().bar_label(p)
        plt.gca().set_xticks(bar_x, labels=labels)
        plt.title(quantizer_name)
        plt.show()
