import os
from pathlib import Path
from auto_sbatch import SBatch, ExperimentHandler
from omegaconf import OmegaConf

from bim_gw.utils import get_args

if __name__ == '__main__':
    work_dir = Path(__file__).absolute().parent.parent
    script_location = "scripts/train.py"

    args = get_args(debug=int(os.getenv("DEBUG", 0)), cli=False)
    cli_args = OmegaConf.from_cli()
    OmegaConf.resolve(args)

    handler = ExperimentHandler(
        script_location,
        str(work_dir.absolute()),
        args.slurm.run_work_directory,
        args.slurm.python_environment,
        args.slurm.run_registry_path,
        pre_modules=["python/3.8.5"],
        run_modules=["python/3.8.5", "cuda/11.5"],
        additional_scripts=[
            "pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html"
        ]
    )

    args = args.slurm
    args = OmegaConf.merge(args, cli_args)
    slurm_args = args.slurm
    del args.slurm

    sbatch = SBatch(slurm_args, args, handler)
    sbatch('python "{script_name}" "devices={num_gpus}" {all_params} "checkpoints_dir=\'{checkpoints_dir}\'"')

