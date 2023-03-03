import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    args = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False)
    cli_args = OmegaConf.from_cli()
    args = OmegaConf.merge(args, cli_args)
    OmegaConf.resolve(args)

    assert "run_work_directory" in args.slurm, "You must specify a run_work_directory in your slurm config."

    run_work_directory = Path(args.slurm.run_work_directory)
    parent_directory = run_work_directory.parent
    files_to_compress = []
    for file in run_work_directory.iterdir():
        if file.is_dir() and file.name.isdigit():
            files_to_compress.append(file.resolve().as_posix())
    print(f"Compressing {len(files_to_compress)} directories...")
    time = datetime.now()
    command = f"tar -czvf {parent_directory}/compressed_{time}.tar.gz {'/* '.join(files_to_compress)}/*"
    if "--dry-run" not in args:
        os.system(command)
    else:
        print(command)
        print(f"Dry run, not compressing {len(files_to_compress)} runs.")
    print(f"Done compressing {len(files_to_compress)} directories.")
    if "-d" in args or "--delete" in args:
        print("Cleaning runs...")
        for file in files_to_compress:
            if "--dry-run" not in args:
                os.system(f"rm -rf {file}")
            else:
                print("[dry-run mode] would deleting", file)
