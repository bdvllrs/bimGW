import os
from pathlib import Path

from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    work_dir = Path(__file__).absolute().parent.parent

    args = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False)
    cli_args = OmegaConf.from_cli()
    args = OmegaConf.merge(args, cli_args)
    OmegaConf.resolve(args)

    for logger in args.loggers:
        if logger.logger == "WandbLogger" and "args" in logger.logger and "save_dir" in logger.logger.args:
            logs_location = Path(logger.logger.args.save_dir) / "wandb"
            if logs_location.is_dir():
                for file in logs_location.iterdir():
                    if file.is_dir() and (file.name.startswith("run-") or file.name.startswith("offline-run-")):
                        print(f"Syncing {file.name}...")
                        os.system(f"wandb sync {file}")
                        print(f"Done syncing {file.name}.")
            if "-d" in args or "--delete" in args:
                print("Done syncing all runs.")
                print("Cleaning local logs...")
                for file in logs_location.iterdir():
                    os.system(f"wandb sync --clean")
