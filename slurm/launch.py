import os
from pathlib import Path

from auto_sbatch import ExperimentHandler, SBatch
from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    work_dir = Path(__file__).absolute().parent.parent

    args = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False, use_schema=False)
    cli_args = OmegaConf.from_cli()
    args = OmegaConf.merge(args, cli_args)
    OmegaConf.resolve(args)

    script_location = f"scripts/{args.slurm.script}.py"

    handler = ExperimentHandler(
        script_location,
        str(work_dir.absolute()),
        args.slurm.run_work_directory,
        args.slurm.python_environment,
        pre_modules=args.slurm.pre_modules,
        run_modules=args.slurm.run_modules,
        setup_experiment=False,
        exclude_in_rsync=["images", "tests"],
    )

    sbatch = SBatch(args.slurm.slurm, cli_args,
        grid_search=args.slurm.grid_search,
        experiment_handler=handler)
    sbatch(
        args.slurm.command,
        schedule_all_tasks=True
    )
