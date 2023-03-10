import logging
from typing import List, Optional, Union

from aim import Image as AimImage, Text as AimText
from aim.sdk.adapters.pytorch_lightning import AimLogger as AimLoggerBase
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, text_from_table


class AimLogger(AimLoggerBase):
    def __init__(self, *aim_params, save_images=True, save_last_images=True, **aim_run_kwargs):
        super().__init__(*aim_params, **aim_run_kwargs)

        self.do_save_images = save_images
        self.do_save_last_images = save_last_images
        if not self.do_save_images:
            logging.warning("AimLogger will not save the images. Set `save_images' to true to log them.")

    def set_summary(self, name, mode="max"):
        pass

    def save_images(self, mode=True):
        self.do_save_images = mode

    def save_tables(self, mode=True):
        pass

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self.do_save_images:
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
        save_last_images=log_args.save_last_images,
        experiment=name,
        **OmegaConf.to_object(log_args.args)
    )
    if tags is not None:
        for tag in tags:
            logger.experiment.add_tag(tag)

    logger.experiment["parameters"] = OmegaConf.to_object(conf)
    return logger
