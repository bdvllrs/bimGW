import logging
from typing import Optional, Union, List

from aim import Image as AimImage, Text as AimText
from aim.sdk.adapters.pytorch_lightning import AimLogger as AimLoggerBase
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, text_from_table


class AimLogger(AimLoggerBase):
    def __init__(self, *aim_params, save_images=True, **aim_run_kwargs):
        super().__init__(*aim_params, **aim_run_kwargs)

        self._save_images = save_images
        if not self._save_images:
            logging.warning("AimLogger will not save the images. Set `save_images' to true to log them.")

    def set_summary(self, name, mode="max"):
        pass

    def save_images(self, mode=True):
        self._save_images = mode

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._save_images:
            image = AimImage(image)
            self.experiment.track(image, name=log_name, step=step)

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if isinstance(text, list):
            text = "\n".join(text)
        text = AimText(text)
        self.experiment.track(text, name=log_name, step=step)

    @rank_zero_only
    def log_table(self, log_name: str, columns: List[str], data: List[List[str]], step: Optional[int] = None):
        self.log_text(log_name, text_from_table(columns, data), step)


def get_aim_logger(name, version, log_args, model, conf, tags, source_files):
    logger = AimLogger(
        save_images=log_args.save_images,
        experiment=name,
        **OmegaConf.to_object(log_args.args)
    )
    if tags is not None:
        for tag in tags:
            logger.experiment.add_tag(tag)

    logger.experiment["parameters"] = OmegaConf.to_object(conf)
    return logger
