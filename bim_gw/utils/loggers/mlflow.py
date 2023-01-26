import logging
from typing import Optional, Union, List

from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType, to_pil_image, text_from_table


class MLFlowLogger(MLFlowLoggerBase):
    def __init__(self, *params, image_location="images", text_location="texts", save_images=True, save_last_images=True,
                 **kwargs):
        super(MLFlowLogger, self).__init__(*params, **kwargs)
        self._image_location = image_location
        self._text_location = text_location
        self.do_save_images = save_images
        self.do_save_last_images = save_last_images

        self._image_last_step = {}
        self._text_last_step = {}

        if not self.do_save_images:
            logging.warning("MLFLowLogger will not save the images. Set `save_images' to true to log them.")

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
            path = f"{self._image_location}/{log_name}/{log_name}_step={step}.png"
            image = to_pil_image(image)
            self.experiment.log_image(self.run_id, image, path)
            path = f"{self._image_location}/{log_name}/{log_name}_step=last.png"
            self.experiment.log_image(self.run_id, image, path)

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if step is None:
            if log_name not in self._text_last_step:
                self._text_last_step[log_name] = 0
            step = self._text_last_step[log_name]
            self._text_last_step[log_name] += 1
        if isinstance(text, list):
            text = "  \n".join(text)
        path = f"{self._text_location}/{log_name}/{log_name}.txt"
        text = f"====  \nstep={step}  \n====  \n{text}  \n"
        self.experiment.log_text(self.run_id, text, path)

    @rank_zero_only
    def log_table(self, log_name: str, columns: List[str], data: List[List[str]], step: Optional[int] = None):
        self.log_text(log_name, text_from_table(columns, data), step)


def get_ml_flow_logger(name, version, log_args, model, conf, tags, source_files):
    if tags is not None:
        tags = {tag: 1 for tag in tags}
    logger = MLFlowLogger(
        save_images=log_args.save_images,
        run_name=version,
        experiment_name=name,
        tags=tags,
        **OmegaConf.to_object(log_args.args)
    )
    logger.log_hyperparams({
        "parameters": OmegaConf.to_object(conf),
    })
    # TODO: add source_files
    return logger
