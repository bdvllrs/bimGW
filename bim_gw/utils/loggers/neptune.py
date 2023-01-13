import logging
from typing import Optional, Union, List

from neptune.new.exceptions import MissingFieldException
from neptune.new.types import File
from omegaconf import OmegaConf
from pytorch_lightning.loggers import NeptuneLogger as NeptuneLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, to_pil_image, text_from_table


class NeptuneLogger(NeptuneLoggerBase):
    def __init__(self, save_images=True, **neptune_run_kwargs):
        super().__init__(**neptune_run_kwargs)

        self._save_images = save_images
        if not self._save_images:
            logging.warning("NeptuneLogger will not save the images. Set `save_images' to true to log them.")

    def set_summary(self, name, mode="max"):
        pass

    def save_images(self, mode=True):
        self._save_images = mode

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._save_images:
            image = to_pil_image(image)
            self.experiment[log_name].log(File.as_image(image), step=step)

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if not isinstance(text, list):
            text = [text]
        self.experiment[log_name].log(text)

    @rank_zero_only
    def log_table(self, log_name: str, columns: List[str], data: List[List[str]], step: Optional[int] = None):
        self.log_text(log_name, text_from_table(columns, data), step)


def get_neptune_logger(name, version, log_args, model, conf, tags, source_files):
    logger = NeptuneLogger(
        save_images=log_args.save_images,
        name=name,
        log_model_checkpoints=False,
        tags=tags,
        source_files=source_files,
        **OmegaConf.to_object(log_args.args)
    )

    for k in range(5):
        try:
            logger.experiment["parameters"] = OmegaConf.to_object(conf)
        except MissingFieldException as e:
            print("Error, retrying")
        else:
            break

    logger.log_model_summary(model=model, max_depth=-1)
    return logger
