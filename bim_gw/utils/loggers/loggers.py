from bim_gw.utils.types import AvailableLoggers


def get_loggers(name, version, args, model, conf, tags, source_files):
    loggers = []
    for logger in args:
        if logger.logger == AvailableLoggers.WandbLogger:
            from .wandb import get_wandb_logger

            loggers.append(
                get_wandb_logger(
                    name, version, logger, model, conf, tags, source_files
                )
            )
        elif logger.logger == AvailableLoggers.CSVLogger:
            from .csv import get_csv_logger

            loggers.append(
                get_csv_logger(
                    name, version, logger, model, conf, tags, source_files
                )
            )
        elif logger.logger == AvailableLoggers.TensorBoardLogger:
            from .tensorboard import get_tensor_board_logger

            loggers.append(
                get_tensor_board_logger(
                    name, version, logger, model, conf, tags, source_files
                )
            )
        else:
            raise ValueError(f"Logger: {logger.logger} is not yet available.")
    return loggers
