import logging
import os
from typing import Optional, Union, List

from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger as CSVLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, to_pil_image, text_from_table


class CSVLogger(CSVLoggerBase):
    def __init__(self, *params, save_images=True, save_last_images=True, image_location="images", text_location="texts",
                 source_location="sources", **kwargs):
        super(CSVLogger, self).__init__(*params, **kwargs)
        self._image_location = image_location
        self._text_location = text_location
        self._source_location = source_location
        self._texts = {}
        self._images = []
        self.do_save_images = save_images
        self.do_save_last_images = save_last_images
        self._image_last_step = {}
        self._text_last_step = {}
        if not self.do_save_images:
            logging.warning("CSVLogger will not save the images. Set `save_images' to true to log them.")

    def set_summary(self, name, mode="max"):
        pass

    def save_images(self, mode=True):
        self.do_save_images = mode

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self.do_save_images:
            if step is None:
                if log_name not in self._image_last_step:
                    self._image_last_step[log_name] = 0
                step = self._image_last_step[log_name]
                self._image_last_step[log_name] += 1
            path = os.path.join(self.log_dir, f"{self._image_location}/{log_name}/{log_name}_step={step}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image = to_pil_image(image)
            self._images.append((path, image))
            path = os.path.join(self.log_dir, f"{self._image_location}/{log_name}/{log_name}_step=last.png")
            self._images.append((path, image))

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if step is None:
            if log_name not in self._text_last_step:
                self._text_last_step[log_name] = 0
            step = self._text_last_step[log_name]
            self._text_last_step[log_name] += 1
        if isinstance(text, list):
            text = "\n".join(text)
        path = os.path.join(self.log_dir, f"{self._text_location}/{log_name}/{log_name}.txt")
        text = f"# step={step}\n{text}\n"
        if path not in self._texts:
            self._texts[path] = ""
        self._texts[path] += text

    @rank_zero_only
    def log_table(self, log_name: str, columns: List[str], data: List[List[str]], step: Optional[int] = None):
        self.log_text(log_name, text_from_table(columns, data), step)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        for path, text in self._texts.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(text)
        for path, image in self._images:
            image.save(path)
        self._texts = {}
        self._images = []


def get_csv_logger(name, version, log_args, model, conf, tags, source_files):
    args = OmegaConf.to_object(log_args.args)
    args['name'] = name
    args['version'] = version
    args['save_images'] = log_args.save_images
    args['save_last_images'] = log_args.save_last_images
    logger = CSVLogger(
        **args
    )
    logger.experiment.log_hparams({
        "parameters": OmegaConf.to_object(conf),
        "tags": tags
    })
    # TODO: add source_files
    return logger
