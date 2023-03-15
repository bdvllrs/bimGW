import logging
from typing import List, Optional, Union

import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger as WandbLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from bim_gw.utils.loggers.utils import ImageType


class WandbLogger(WandbLoggerBase):
    def __init__(
        self, *params, save_images=True, save_last_images=True,
        save_last_tables=True, save_tables=True, **kwargs
    ):
        super().__init__(*params, **kwargs)

        self.do_save_images = save_images
        self.do_save_last_images = save_last_images
        self.do_save_tables = save_tables
        self.do_save_last_tables = save_last_tables
        if not self.do_save_images:
            logging.warning(
                "WandbLogger will not save the images. Set `save_images' to "
                "true to log them."
            )
        if not self.do_save_tables:
            logging.warning(
                "WandbLogger will not save the tables. Set `save_tables' to "
                "true to log them."
            )

    def set_summary(self, name, mode="min"):
        wandb.define_metric(name, summary=mode)
        # pass

    def save_images(self, mode=True):
        self.do_save_images = mode

    def save_tables(self, mode=True):
        self.do_save_tables = mode

    @rank_zero_only
    def log_image(
        self, log_name: str, image: ImageType, step: Optional[int] = None
    ) -> None:
        if self.do_save_images:
            super(WandbLogger, self).log_image(
                key=log_name, images=[image], step=step
            )

    @rank_zero_only
    def log_text(
        self, log_name: str, text: Union[List, str], step: Optional[int] = None
    ) -> None:
        if self.do_save_tables:
            if not isinstance(text, list):
                text = [[text]]
            else:
                text = list(map(lambda x: [x], text))
            super(WandbLogger, self).log_text(
                key=log_name, columns=["text"], data=text, step=step
            )

    @rank_zero_only
    def log_table(
        self, log_name: str, columns: List[str], data: List[List[str]],
        step: Optional[int] = None
    ):
        if self.do_save_tables:
            super(WandbLogger, self).log_table(
                key=log_name, columns=columns, data=data, step=step
            )


def get_wandb_logger(name, version, log_args, model, conf, tags, source_files):
    logger = WandbLogger(
        save_images=log_args.save_images,
        save_last_images=log_args.save_last_images,
        save_last_tables=log_args.save_last_tables,
        save_tables=log_args.save_tables if "save_tables" in log_args else
        True,
        tags=tags,
        **OmegaConf.to_container(log_args.args, resolve=True)
    )
    if version is not None:
        logger.experiment.name = version
    # logger.experiment.log_code("../",
    #                            include_fn=lambda path: path.endswith(
    #                            ".py") or path.endswith(".yaml") or
    #                            path.endswith(
    #                                ".txt"))
    conf["_script_name"] = name
    logger.log_hyperparams(
        {"parameters": OmegaConf.to_container(conf, resolve=True)}
    )
    if "watch_model" in log_args and log_args.watch_model:
        logger.experiment.watch(model)
    return logger
