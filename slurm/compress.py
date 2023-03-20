import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from bim_gw.utils import get_args
from bim_gw.utils.cli import parse_args


def should_keep_file(file_path: Path, args: DictConfig):
    return (
            file_path.is_dir() and file_path.name.isdigit()
            and (args.before is None
                 or int(file_path.name) < args.before)
            and (args.after is None
                 or int(file_path.name) > args.after)
    )


@dataclass
class CompressArgs:
    help: bool = field(
        default=False, metadata={
            "cli_names": ["--help", "-h"],
        }
    )
    before: Optional[int] = field(
        default=None, metadata={
            "cli_names": ["--before"],
        }
    )
    after: Optional[int] = field(
        default=None, metadata={
            "cli_names": ["--after"],
        }
    )
    name: Optional[str] = field(
        default=None, metadata={
            "cli_names": ["-n", "--name"],
        }
    )
    delete: bool = field(
        default=False, metadata={
            "cli_names": ["-d", "--delete"],
        }
    )
    dry_run: bool = field(
        default=False, metadata={
            "cli_names": ["--dry-run"],
        }
    )


if __name__ == '__main__':
    args = parse_args(CompressArgs)

    if args.help:
        print("Compresses runs in a run_work_directory.")
        print(
            "Usage: python compress.py --slurm.run_work_directory <path> "
            "[--before <int>] [--after <int>] [-n <name>] [-d] "
            "[--dry-run]"
        )
        print("Options:")
        print(
            "  '--before <int>'  Compresses run IDs before the given run "
            "number."
        )
        print(
            "  '--after <int>'  Compresses run IDs after the given run number."
        )
        print("  '-n <name>'  Name of the compressed file.")
        print("   -d  Deletes the experiment folders after compressing them.")
        print(
            "   --dry-run  Performs a dry run, does not compress or delete "
            "runs."
        )
        exit(0)

    conf = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False, verbose=False)
    OmegaConf.resolve(conf)

    if "run_work_directory" not in conf.slurm:
        raise ValueError(
            "You must specify a run_work_directory in your slurm config."
        )

    run_work_directory = Path(conf.slurm.run_work_directory)
    parent_directory = run_work_directory.parent
    files_to_compress = []
    for file in run_work_directory.iterdir():
        if should_keep_file(file, args):
            files_to_compress.append(file.resolve().as_posix())
    print(f"Compressing {len(files_to_compress)} directories...")
    time = str(datetime.now()).replace(" ", "_")
    time = time.replace(":", "-").replace(".", "-")
    compressed_filename = f"compressed_{time}"
    if args.name is not None:
        compressed_filename = args.name

    command = f"tar -czvf {parent_directory}/{compressed_filename}.tar.gz "
    command += f"{' '.join(files_to_compress)}"
    if not args.dry_run:
        os.system(command)
    else:
        print(command)
        print(f"Dry run, not compressing {len(files_to_compress)} runs.")
    print(f"Done compressing {len(files_to_compress)} directories.")
    if args.delete:
        print("Cleaning runs...")
        for file in files_to_compress:
            if not args.dry_run:
                os.system(f"rm -rf {file}")
            else:
                print("[dry-run mode] would deleting", file)
