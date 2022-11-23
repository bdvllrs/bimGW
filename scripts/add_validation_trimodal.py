import os

import pandas as pd
import wandb
from pytorch_lightning import seed_everything

from bim_gw.datasets import load_dataset
from bim_gw.modules.gw import GlobalWorkspace

from bim_gw.scripts.utils import get_trainer, get_domains
from bim_gw.utils import get_args
from bim_gw.utils.utils import find_best_epoch

if __name__ == "__main__":
    args = get_args(debug=int(os.getenv("DEBUG", 0)))
    df = pd.read_csv(args.csv_file)
    for idx, row in df.iterrows():
    # for idx, row in enumerate(df):
        slurm_id = row['Name'].split('-')[1]
        wandb_id = row['ID']
        checkpoint_path = args.path_to_checkpoint.format(id=slurm_id)
        checkpoint_path = find_best_epoch(checkpoint_path)
        print(row)
        seed_everything(row['parameters/seed'])

        data = load_dataset(args, args.global_workspace)
        data.prepare_data()
        data.setup(stage="fit")

        global_workspace = GlobalWorkspace.load_from_checkpoint(checkpoint_path, domain_mods=get_domains(args, data),
                                                                domain_examples=data.domain_examples)

        for logger in args.loggers:
            logger.args.version = wandb_id
            logger.args.id = wandb_id
            logger.args.resume = True

        trainer = get_trainer("train_gw", args, global_workspace,
                              monitor_loss="val/in_dist/total_loss",
                              early_stopping_patience=args.global_workspace.early_stopping_patience,
                              trainer_args={
                                  # "val_check_interval": args.global_workspace.prop_labelled_images
                              })

        for logger in trainer.loggers:
            logger.save_images(True)
        trainer.validate(global_workspace, data)
        trainer.test(global_workspace, data)
        wandb.finish()
