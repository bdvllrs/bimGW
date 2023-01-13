def get_loggers(name, version, args, model, conf, tags, source_files):
    loggers = []
    for logger in args:
        if logger.logger == "NeptuneLogger":
            from bim_gw.utils.loggers.neptune import get_neptune_logger
            loggers.append(get_neptune_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "WandbLogger":
            from bim_gw.utils.loggers.wandb import get_wandb_logger
            loggers.append(get_wandb_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "CSVLogger":
            from bim_gw.utils.loggers.csv import get_csv_logger
            loggers.append(get_csv_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "TensorBoardLogger":
            from bim_gw.utils.loggers.tensorboard import get_tensor_board_logger
            loggers.append(get_tensor_board_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "MLFlowLogger":
            from bim_gw.utils.loggers.mlflow import get_ml_flow_logger
            loggers.append(get_ml_flow_logger(name, version, logger, model, conf, tags, source_files))
        elif logger.logger == "AimLogger":
            from bim_gw.utils.loggers.aim import get_aim_logger
            loggers.append(get_aim_logger(name, version, logger, model, conf, tags, source_files))
        else:
            raise ValueError(f"Logger: {logger.logger} is not yet available.")
    return loggers
