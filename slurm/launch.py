import os
from pathlib import Path

from auto_sbatch import ExperimentHandler, GridSearch, SBatch
from omegaconf import OmegaConf

from bim_gw.utils import get_args


def grid_search_exclusion_from_past_search(dotlist):
    def walk_omegaconf(cfg, idx, prefix=""):
        result = {}
        for key, value in cfg.items():
            if OmegaConf.is_dict(value):
                result.update(
                    walk_omegaconf(value, idx, prefix=f"{prefix}{key}.")
                )
            else:
                result[f"{prefix}{key}"] = value[idx]
        return result

    args = OmegaConf.from_dotlist(dotlist)
    grid_search_params = [x.split("=")[0] for x in dotlist]
    grid_search = GridSearch(grid_search_params)
    n, past_search = grid_search.get_combinations(args)
    return [walk_omegaconf(past_search, k) for k in range(n)]


def main(args, cli_args):
    work_dir = Path(__file__).absolute().parent.parent

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

    extra_args = cli_args
    grid_search = None
    # Add all grid search parameters as parameters to auto_sbatch
    if args.slurm.grid_search is not None:
        extra_args = OmegaConf.unsafe_merge(
            OmegaConf.from_dotlist(
                [
                    str(arg)
                    + "="
                    + str(OmegaConf.select(args, arg, throw_on_missing=True))
                    for arg in args.slurm.grid_search
                ]
            ),
            extra_args,
        )

        grid_search = GridSearch(
            args.slurm.grid_search, args.slurm.grid_search_exclude
        )

    sbatch = SBatch(
        args.slurm.slurm,
        extra_args,
        grid_search=grid_search,
        experiment_handler=handler,
    )
    sbatch.run(args.slurm.command, schedule_all_tasks=True)


if __name__ == "__main__":
    main(
        get_args(
            debug=int(os.getenv("DEBUG", 0)), cli=False, use_schema=False
        ),
        OmegaConf.from_cli(),
    )
