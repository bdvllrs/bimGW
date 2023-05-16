from typing import Union

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

ImageType = Union[torch.Tensor, plt.Figure, Image.Image]


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def to_pil_image(image: ImageType):
    if isinstance(image, torch.Tensor):
        return torchvision.transforms.ToPILImage(mode="RGB")(image.cpu())
    elif isinstance(image, plt.Figure):
        return fig2img(image)
    elif isinstance(image, Image.Image):
        return image


def text_from_column_data(x):
    column, data = x
    if isinstance(data, float):
        return f"{x[0]}: {x[1]: .4f}"
    return f"{x[0]}: {x[1]}"


def text_from_table(columns, data):
    text = ""
    for k in range(len(data)):
        text += (
            f"{k + 1} - "
            + ", ".join(map(text_from_column_data, zip(columns, data[k])))
            + "\n"
        )
    text += "---- \n"
    return text
