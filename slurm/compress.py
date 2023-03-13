import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    args = OmegaConf.from_cli()

    if "--help" in args:
        print("Compresses runs in a run_work_directory.")
        print(
            "Usage: python compress.py --slurm.run_work_directory <path> ["
            "'--before=<int>'] ['--after=<int>'] ['-n=<name>'] [-d] ["
            "--dry-run]"
        )
        print("Options:")
        print(
            "  '--before=<int>'  Compresses run IDs before the given run "
            "number."
        )
        print(
            "  '--after=<int>'  Compresses run IDs after the given run number."
        )
        print("  '-n=<name>'  Name of the compressed file.")
        print("   -d  Deletes the experiment folders after compressing them.")
        print(
            "   --dry-run  Performs a dry run, does not compress or delete "
            "runs."
        )
        exit(0)

    conf = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False, verbose=False)
    OmegaConf.resolve(conf)

    assert "run_work_directory" in conf.slurm, "You must specify a " \
                                               "run_work_directory in your " \
                                               "slurm config."

    run_work_directory = Path(conf.slurm.run_work_directory)
    parent_directory = run_work_directory.parent
    files_to_compress = []
    for file in run_work_directory.iterdir():
        if file.is_dir() and file.name.isdigit() and (
                not "--before" in args or int(file.name) < int(
            args["--before"]
        )) and (
                not "--after" in args or int(file.name) > int(
            args["--after"]
        )):
            files_to_compress.append(file.resolve().as_posix())
    print(f"Compressing {len(files_to_compress)} directories...")
    time = str(datetime.now()).replace(" ", "_").replace(":", "-").replace(
        ".", "-"
    )
    compressed_filename = f"compressed_{time}" if "-n" not in args else args[
        "-n"]
    command = f"tar -czvf {parent_directory}/{compressed_filename}.tar.gz " \
              f"{' '.join(files_to_compress)}"
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
