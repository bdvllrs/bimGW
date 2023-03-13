import logging
from typing import List, Optional, Union

import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.loggers import \
    TensorBoardLogger as TensorBoardLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, text_from_table


class TensorBoardLogger(TensorBoardLoggerBase):
    def __init__(
        self, *params, save_images=True, save_last_images=True, **kwargs
    ):
        super(TensorBoardLogger, self).__init__(*params, **kwargs)
        self.do_save_images = save_images
        self.do_save_last_images = save_last_images
        if not self.do_save_images:
            logging.warning(
                "TensorBoardLogger will not save the images. Set "
                "`save_images' to true to log them."
            )

    def set_summary(self, name, mode="max"):
        pass

    def save_images(self, mode=True):
        self.do_save_images = mode

    def save_tables(self, mode=True):
        pass

    @rank_zero_only
    def log_image(
        self, log_name: str, image: ImageType, step: Optional[int] = None
    ) -> None:
        if self.do_save_images:
            if isinstance(image, Image.Image):
                image = torchvision.transforms.ToTensor()(image)
            if isinstance(image, torch.Tensor):
                self.experiment.add_image(log_name, image, step)
            elif isinstance(image, plt.Figure):
                self.experiment.add_figure(log_name, image, step)

    @rank_zero_only
    def log_text(
        self, log_name: str, text: Union[List, str], step: Optional[int] = None
    ) -> None:
        if isinstance(text, list):
            text = "  \n".join(text)
        self.experiment.add_text(log_name, text, step)

    @rank_zero_only
    def log_table(
        self, log_name: str, columns: List[str], data: List[List[str]],
        step: Optional[int] = None
    ):
        self.log_text(log_name, text_from_table(columns, data), step)


def get_tensor_board_logger(
    name, version, log_args, model, conf, tags, source_files
):
    args = OmegaConf.to_object(log_args.args)
    args['name'] = name
    args['version'] = version
    args['save_images'] = log_args.save_images
    args['save_last_images'] = log_args.save_last_images
    logger = TensorBoardLogger(
        **args
    )
    hparams = {"parameters": OmegaConf.to_object(conf)}
    if tags is not None:
        hparams["tags"] = tags
    logger.log_hyperparams(hparams)
    # logger.experiment.add_graph(model)
    # TODO: add source_files
    return logger
