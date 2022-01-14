import os

import matplotlib.pyplot as plt
import seaborn as sn
import torch
from transformers import BertModel, BertTokenizer

from bim_gw.datasets.simple_shapes import get_preprocess, SimpleShapesDataset
from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    transformer_model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = SimpleShapesDataset(args.simple_shapes_path, "train",
                                  get_preprocess(False),
                                  lambda v, t: t,
                                  textify=True)
    num_examples_per_class = 8
    num_classes = 32

    text_rep = []

    for i in range(num_classes):
        sentences = [dataset[i][0] for k in range(num_examples_per_class)]
        tokens = tokenizer(sentences, return_tensors='pt', padding=True)
        latent_rep = transformer_model(**tokens)["last_hidden_state"][:, 0]
        text_rep.append(latent_rep)

    text_rep = torch.stack(text_rep, dim=0)
    distances = torch.zeros(num_classes, num_classes, num_examples_per_class, num_examples_per_class)

    for i in range(num_classes):
        for j in range(num_classes):
            for k in range(num_examples_per_class):
                for p in range(num_examples_per_class):
                    distances[i, j, k, p] = torch.nn.functional.mse_loss(text_rep[i, k], text_rep[j, p])
                    if i == j:
                        factor = (num_examples_per_class * num_examples_per_class)
                        factor /= factor - num_examples_per_class
                        distances[i, j, k, p] *= factor
    avg_distantes = distances.mean((2, 3)).detach().cpu().numpy()
    sn.heatmap(avg_distantes)
    # for i in range(num_classes):
    #     for j in range(num_classes):
    #         text = plt.text(j, i, f"{avg_distantes[i, j]:.2f}", ha="center", va="center", color="w")
    plt.title(f"Intra and inter distances between BERT vectors")
    plt.xlabel("Images")
    plt.ylabel("Images")
    plt.show()
    print('ok')

    # tokens = tokenizer(sentence, return_tensors='pt', padding=True)
    # encoding = transformer_model(**tokens)["last_hidden_state"][:, 0]
