import os

import torch
from transformers import BertModel, BertTokenizer

from bim_gw.utils import get_args

from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.modules import ShapesLM

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.lm.prop_labelled_images = 1.

    args.lm.split_ood = False
    args.lm.selected_domains = {"a": "attr", "t": "t"}
    args.lm.data_augmentation = False

    data = load_dataset(args, args.lm, add_unimodal=False)
    data.prepare_data()
    data.setup(stage="fit")

    domain_examples = {d: data.domain_examples["in_dist"][0][d][1:] for d in data.domain_examples["in_dist"][0].keys()}

    lm = ShapesLM.load_from_checkpoint(args.checkpoint, strict=False,
                                       bert_path=args.global_workspace.bert_path,
                                       validation_domain_examples=domain_examples)
    lm.eval()
    lm.freeze()
    lm.to(device)

    transformer = BertModel.from_pretrained(args.global_workspace.bert_path)
    transformer.eval()
    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(args.global_workspace.bert_path)

    sentences = [
        "There is an object in the image.",
        "There is an object in the bottom-right hand corner of the image.",
        "A diamond is at the top.",
        "A diamond is at the bottom.",
        "A diamond. It is at the bottom.",
        "A diamond is at the very bottom.",
        "A diamond. It is at the very bottom.",
        "A big red egg.",
        "A big red color egg.",
        "Purple.",
        "Purple color.",
        "A medium four-sided shape in bright red color. It is in the upper side, slightly left and is pointing to the left top-left corner.",
        "A medium four-sided shape in bright red color.",
        "The image represents an egg shape. It is in the upper side, slightly left."
    ]

    sentences += ["" for k in range(32 - len(sentences))]

    tokens = tokenizer(sentences, return_tensors='pt', padding=True).to(device)
    x = transformer(**tokens)["last_hidden_state"][:, 0]

    lm.log_domain(None, [x, sentences], "")

