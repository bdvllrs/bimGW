import logging
import os
from typing import Optional, Union, List

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from neptune.new.exceptions import MissingFieldException
from neptune.new.types import File
from omegaconf import OmegaConf
from pytorch_lightning.loggers import (
    NeptuneLogger as NeptuneLoggerBase,
    TensorBoardLogger as TensorBoardLoggerBase,
    MLFlowLogger as MLFlowLoggerBase,
    CSVLogger as CSVLoggerBase
)
from pytorch_lightning.utilities import rank_zero_only

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
        return torchvision.transforms.ToPILImage(mode='RGB')(image.cpu())
    elif isinstance(image, plt.Figure):
        return fig2img(image)
    elif isinstance(image, Image.Image):
        return image


class NeptuneLogger(NeptuneLoggerBase):
    def __init__(self, save_images=True, **neptune_run_kwargs):
        super().__init__(**neptune_run_kwargs)

        self._log_images = save_images
        if not self._log_images:
            logging.warning("NeptuneLogger will not save the images. Set `save_images' to true to log them.")

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._log_images:
            image = to_pil_image(image)
            self.experiment[log_name].log(File.as_image(image), step=step)

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if not isinstance(text, list):
            text = [text]
        self.experiment[log_name].log(text)


class TensorBoardLogger(TensorBoardLoggerBase):
    def __init__(self, *params, save_images=True, **kwargs):
        super(TensorBoardLogger, self).__init__(*params, **kwargs)
        self._log_images = save_images
        if not self._log_images:
            logging.warning("TensorBoardLogger will not save the images. Set `save_images' to true to log them.")

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._log_images:
            if isinstance(image, Image.Image):
                image = torchvision.transforms.ToTensor()(image)
            if isinstance(image, torch.Tensor):
                self.experiment.add_image(log_name, image, step)
            elif isinstance(image, plt.Figure):
                self.experiment.add_figure(log_name, image, step)

    @rank_zero_only
    def log_text(self, log_name: str, text: Union[List, str], step: Optional[int] = None) -> None:
        if isinstance(text, list):
            text = "  \n".join(text)
        self.experiment.add_text(log_name, text, step)


class MLFlowLogger(MLFlowLoggerBase):
    def __init__(self, *params, image_location="images", text_location="texts", save_images=True, **kwargs):
        super(MLFlowLogger, self).__init__(*params, **kwargs)
        self._image_location = image_location
        self._text_location = text_location
        self._save_images = save_images
        self._image_last_step = {}
        self._text_last_step = {}

        if not self._save_images:
            logging.warning("MLFLowLogger will not save the images. Set `save_images' to true to log them.")

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._save_images:
            if step is None:
                if log_name not in self._image_last_step:
                    self._image_last_step[log_name] = 0
                step = self._image_last_step[log_name]
                self._image_last_step[log_name] += 1
            path = f"{self._image_location}/{log_name}/{log_name}_step={step}.png"
            image = to_pil_image(image)
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


class CSVLogger(CSVLoggerBase):
    def __init__(self, *params, save_images=True, image_location="images", text_location="texts",
                 source_location="sources", **kwargs):
        super(CSVLogger, self).__init__(*params, **kwargs)
        self._image_location = image_location
        self._text_location = text_location
        self._source_location = source_location
        self._texts = {}
        self._images = []
        self._save_images = save_images
        self._image_last_step = {}
        self._text_last_step = {}
        if not self._save_images:
            logging.warning("CSVLogger will not save the images. Set `save_images' to true to log them.")

    @rank_zero_only
    def log_image(self, log_name: str, image: ImageType, step: Optional[int] = None) -> None:
        if self._save_images:
            if step is None:
                if log_name not in self._image_last_step:
                    self._image_last_step[log_name] = 0
                step = self._image_last_step[log_name]
                self._image_last_step[log_name] += 1
            path = os.path.join(self.log_dir, f"{self._image_location}/{log_name}/{log_name}_step={step}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image = to_pil_image(image)
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


def get_tensor_board_logger(name, version, log_args, model, conf, tags, source_files):
    args = OmegaConf.to_object(log_args.args)
    args['name'] = name
    args['version'] = version
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


def get_csv_logger(name, version, log_args, model, conf, tags, source_files):
    args = OmegaConf.to_object(log_args.args)
    args['name'] = name
    args['version'] = version
    logger = CSVLogger(
        **args
    )
    logger.experiment.log_hparams({
        "parameters": OmegaConf.to_object(conf),
        "tags": tags
    })
    # TODO: add source_files
    return logger

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

def get_loggers(name, version, args, model, conf, tags, source_files):
    loggers = []
    for logger in args:
        if logger.logger == "NeptuneLogger":
            loggers.append(get_neptune_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "CSVLogger":
            loggers.append(get_csv_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "TensorBoardLogger":
            loggers.append(get_tensor_board_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "MLFlowLogger":
            loggers.append(get_ml_flow_logger(name, version, logger, model, conf, tags, source_files))
        else:
            raise ValueError(f"Logger: {logger.logger} is not yet available.")
        # TODO: implement for the other loggers
    return loggers
